"""三档搜索 AI 基线对战（进程内，不依赖 HTTP）。

运行示例（仓库根目录）：

  PYTHONPATH=training training/.venv/bin/python training/examples/run_search_baseline.py
  PYTHONPATH=training training/.venv/bin/python training/examples/run_search_baseline.py --games 20 --output data/search_baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from quoridor_lab.rules import State
from quoridor_lab.search import DIFFICULTY_PRESETS, SearchAgent

MATCHUPS = (("easy", "medium"), ("easy", "hard"), ("medium", "hard"))
MAX_PLIES = 256


def play_game(red: SearchAgent, blue: SearchAgent, *, seed: int) -> dict:
    state = State()
    agents = {1: red, 2: blue}
    latencies = {1: [], 2: []}

    while state.winner is None and len(state.history) < MAX_PLIES:
        player = state.turn
        decision = agents[player].choose(state)
        if state.error(decision.action):
            raise RuntimeError(
                f"Illegal action {decision.action.notation} at ply {len(state.history) + 1} (seed={seed})"
            )
        latencies[player].append(decision.latency_seconds)
        state = state.apply(decision.action)

    if state.winner == 1:
        result = "red"
    elif state.winner == 2:
        result = "blue"
    else:
        result = "draw"

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    return {
        "result": result,
        "plies": len(state.history),
        "illegal_moves": 0,
        "red_avg_latency": avg(latencies[1]),
        "blue_avg_latency": avg(latencies[2]),
        "red_plies": len(latencies[1]),
        "blue_plies": len(latencies[2]),
    }


def matchup_summary(rows: list[dict], a: str, b: str) -> dict:
    subset = [r for r in rows if r["matchup"] == f"{a}_vs_{b}"]
    wins = {a: 0, b: 0, "draw": 0}
    for row in subset:
        if row["result"] == "draw":
            wins["draw"] += 1
        elif row["result"] == "red":
            wins[row["red"]] += 1
        else:
            wins[row["blue"]] += 1
    return {
        "pair": f"{a}_vs_{b}",
        "games": len(subset),
        "wins": wins,
        f"{a}_win_rate": round(wins[a] / len(subset), 4) if subset else None,
        f"{b}_win_rate": round(wins[b] / len(subset), 4) if subset else None,
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
        rate = item["wins"].get(high, 0) / item["games"]
        if rate > 0.5:
            verdict = "强于对方（>50%）"
        elif rate == 0.5:
            verdict = "胜负接近（=50%）"
        else:
            verdict = "未体现强度递增（<50%）"
        lines.append(f"{label}: 胜率 {rate:.1%}，{verdict}，样本 {item['games']} 局。")
    return "\n".join(lines) if lines else "无对阵数据。"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search AI difficulty baseline")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--output", default="data/search_baseline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.games < 1:
        raise SystemExit("--games must be >= 1")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict] = []

    for level_a, level_b in MATCHUPS:
        for game_idx in range(args.games):
            for swap in (False, True):
                red_level, blue_level = (level_b, level_a) if swap else (level_a, level_b)
                seed = args.seed + game_idx * 4 + (1 if swap else 0) + hash((level_a, level_b)) % 10_000
                red = SearchAgent(red_level, seed=seed)
                blue = SearchAgent(blue_level, seed=seed + 1)
                result = play_game(red, blue, seed=seed)
                rows.append(
                    {
                        "matchup": f"{level_a}_vs_{level_b}",
                        "game_index": game_idx,
                        "swap_colors": swap,
                        "red": red_level,
                        "blue": blue_level,
                        **result,
                    }
                )

    summaries = [matchup_summary(rows, a, b) for a, b in MATCHUPS]
    conclusion = build_conclusion(summaries)

    csv_path = output / f"search_baseline_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "matchup",
                "game_index",
                "swap_colors",
                "red",
                "blue",
                "result",
                "plies",
                "illegal_moves",
                "red_avg_latency",
                "blue_avg_latency",
                "red_plies",
                "blue_plies",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = output / f"search_baseline_{stamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "date_utc": stamp,
                "games_per_color": args.games,
                "max_plies": MAX_PLIES,
                "difficulty_config": {k: asdict(v) for k, v in DIFFICULTY_PRESETS.items()},
                "matchup_summary": summaries,
                "conclusion": conclusion,
                "csv": csv_path.name,
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