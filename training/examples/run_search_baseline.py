"""三档搜索 AI 基线对战（进程内，不依赖 HTTP）。

运行示例（仓库根目录）：

  PYTHONPATH=training training/.venv/bin/python training/examples/run_search_baseline.py
  PYTHONPATH=training training/.venv/bin/python training/examples/run_search_baseline.py --games 20 --output data/search_baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from quoridor_lab.rules import State
from quoridor_lab.search import DIFFICULTY_PRESETS, SearchAgent

MATCHUPS = (("easy", "medium"), ("easy", "hard"), ("medium", "hard"))
MAX_PLIES = 256


def game_seed(base_seed: int, matchup: tuple[str, str], game_index: int) -> int:
    """Stable across Python processes; do not use randomized hash()."""
    return base_seed + MATCHUPS.index(matchup) * 1_000_000 + game_index * 4


def make_opening(seed: int, plies: int) -> tuple[str, ...]:
    """Use the same seeded legal opening for both color assignments."""
    rng = random.Random(seed)
    state = State()
    for _ in range(plies):
        moves = [a for a in state.legal_actions()
                 if a.kind == "move" and state.apply(a).winner is None]
        if not moves:
            break
        state = state.apply(rng.choice(moves))
    return state.history


def play_game(red: SearchAgent, blue: SearchAgent, *, seed: int,
              opening: tuple[str, ...] = (), max_plies: int = MAX_PLIES) -> dict:
    state = State.from_moves(opening)
    agents = {1: red, 2: blue}
    latencies = {1: [], 2: []}
    steps = []

    while state.winner is None and len(state.history) < max_plies:
        player = state.turn
        decision = agents[player].choose(state)
        if state.error(decision.action):
            raise RuntimeError(
                f"Illegal action {decision.action.notation} at ply {len(state.history) + 1} (seed={seed})"
            )
        latencies[player].append(decision.latency_seconds)
        steps.append({"ply": len(state.history) + 1, "player": player,
                      "notation": decision.action.notation,
                      "latency_seconds": decision.latency_seconds,
                      "source": decision.source, "metadata": decision.metadata})
        state = state.apply(decision.action)

    if state.winner == 1:
        result = "red"
    elif state.winner == 2:
        result = "blue"
    else:
        result = "truncated"

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    return {
        "result": result,
        "truncated": state.winner is None,
        "plies": len(state.history),
        "moves": json.dumps(state.history),
        "_steps": steps,
        "illegal_moves": 0,
        "red_avg_latency": avg(latencies[1]),
        "blue_avg_latency": avg(latencies[2]),
        "red_plies": len(latencies[1]),
        "blue_plies": len(latencies[2]),
    }


def matchup_summary(rows: list[dict], a: str, b: str) -> dict:
    subset = [r for r in rows if r["matchup"] == f"{a}_vs_{b}"]
    wins = {a: 0, b: 0}
    truncated = 0
    for row in subset:
        if row["result"] == "truncated":
            truncated += 1
        elif row["result"] == "red":
            wins[row["red"]] += 1
        else:
            wins[row["blue"]] += 1
    completed = len(subset) - truncated
    return {
        "pair": f"{a}_vs_{b}",
        "games": len(subset),
        "completed_games": completed,
        "truncated_games": truncated,
        "wins": wins,
        f"{a}_win_rate": round(wins[a] / completed, 4) if completed else None,
        f"{b}_win_rate": round(wins[b] / completed, 4) if completed else None,
    }


def build_conclusion(summaries: list[dict]) -> str:
    lookup = {s["pair"]: s for s in summaries}
    lines: list[str] = []
    for low, high, label in (
        ("easy", "medium", "medium 对 easy"),
        ("easy", "hard", "hard 对 easy"),
        ("medium", "hard", "hard 对 medium"),
    ):
        item = lookup.get(f"{low}_vs_{high}")
        if item is None or not item["games"]:
            continue
        completed = item["completed_games"]
        if completed:
            rate = item["wins"].get(high, 0) / completed
            lines.append(f"{label}: 已完成对局胜率 {rate:.1%}，完成 {completed} 局，"
                         f"截断 {item['truncated_games']} 局。")
        else:
            lines.append(f"{label}: 无完成对局，截断 {item['truncated_games']} 局，无法计算胜率。")
    if lines:
        lines.append("结果仅描述这些开局下的样本，不足以证明三档棋力递增。")
    return "\n".join(lines) if lines else "无对阵数据。"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search AI difficulty baseline")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--output", default="data/search_baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=MAX_PLIES)
    parser.add_argument("--hard-time-limit-seconds", type=float, default=2.0,
                        help="Use 0 to disable wall-clock cutoffs for reproducible search")
    args = parser.parse_args()
    if args.games < 1:
        raise SystemExit("--games must be >= 1")
    if not 0 <= args.opening_plies < args.max_plies:
        raise SystemExit("Require 0 <= --opening-plies < --max-plies")
    if args.hard_time_limit_seconds < 0:
        raise SystemExit("--hard-time-limit-seconds must be >= 0")

    configs = dict(DIFFICULTY_PRESETS)
    configs["hard"] = replace(configs["hard"],
                              time_limit_seconds=args.hard_time_limit_seconds or None)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict] = []

    for level_a, level_b in MATCHUPS:
        for game_idx in range(args.games):
            opening_seed = game_seed(args.seed, (level_a, level_b), game_idx)
            opening = make_opening(opening_seed, args.opening_plies)
            for swap in (False, True):
                red_level, blue_level = (level_b, level_a) if swap else (level_a, level_b)
                seed = opening_seed + (2 if swap else 0)
                red = SearchAgent(seed=seed, config=configs[red_level])
                blue = SearchAgent(seed=seed + 1, config=configs[blue_level])
                result = play_game(red, blue, seed=seed, opening=opening, max_plies=args.max_plies)
                rows.append(
                    {
                        "matchup": f"{level_a}_vs_{level_b}",
                        "game_index": game_idx,
                        "swap_colors": swap,
                        "seed": seed,
                        "opening_seed": opening_seed,
                        "opening_moves": json.dumps(opening),
                        "red": red_level,
                        "blue": blue_level,
                        **result,
                    }
                )

    summaries = [matchup_summary(rows, a, b) for a, b in MATCHUPS]
    conclusion = build_conclusion(summaries)

    steps_path = output / f"search_baseline_{stamp}_steps.jsonl"
    with steps_path.open("w", encoding="utf-8") as file:
        for row in rows:
            identity = {k: row[k] for k in ("matchup", "game_index", "swap_colors", "red", "blue")}
            for step in row["_steps"]:
                file.write(json.dumps({**identity, **step}, ensure_ascii=False) + "\n")

    csv_path = output / f"search_baseline_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "matchup",
                "game_index",
                "swap_colors",
                "seed",
                "opening_seed",
                "opening_moves",
                "red",
                "blue",
                "result",
                "truncated",
                "plies",
                "moves",
                "illegal_moves",
                "red_avg_latency",
                "blue_avg_latency",
                "red_plies",
                "blue_plies",
            ],
        )
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "_steps"} for row in rows)

    json_path = output / f"search_baseline_{stamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "date_utc": stamp,
                "games_per_color": args.games,
                "max_plies": args.max_plies,
                "seed": args.seed,
                "opening_plies": args.opening_plies,
                "reproducibility_note": "限时搜索的完成深度可能随机器速度变化；"
                    "设 --hard-time-limit-seconds 0 可禁用该截断。"
                    "--opening-plies 0 会重复标准初始局面，确定性策略的重复对局不增加独立样本。",
                "difficulty_config": {k: asdict(v) for k, v in configs.items()},
                "matchup_summary": summaries,
                "conclusion": conclusion,
                "csv": csv_path.name,
                "steps_jsonl": steps_path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(conclusion)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
