"""Round-robin baseline: easy vs medium vs hard SearchAgent, swap colors each game."""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

from quoridor_lab.rules import State
from quoridor_lab.search import DIFFICULTY_PRESETS, SearchAgent


def play_game(red: SearchAgent, blue: SearchAgent, max_plies: int) -> dict:
    state = State()
    agents = {1: red, 2: blue}
    step_records = []
    illegal = 0

    while state.winner is None and len(state.history) < max_plies:
        agent = agents[state.turn]
        decision = agent.choose(state)
        reason = state.error(decision.action)
        if reason:
            illegal += 1
            break
        step_records.append(
            {
                "ply": len(state.history) + 1,
                "player": state.turn,
                "notation": decision.action.notation,
                "latency_seconds": decision.latency_seconds,
                "source": decision.source,
            }
        )
        state = state.apply(decision.action)

    winner_side = None
    if state.winner == 1:
        winner_side = "red"
    elif state.winner == 2:
        winner_side = "blue"

    return {
        "winner": winner_side,
        "plies": len(state.history),
        "truncated": state.winner is None and len(state.history) >= max_plies,
        "illegal_moves": illegal,
        "steps": step_records,
        "total_latency": sum(s["latency_seconds"] for s in step_records),
    }


def summarize(rows: list[dict]) -> dict:
    def wins(level: str, color: str) -> int:
        return sum(
            1
            for r in rows
            if r["winner"] == color
            and (r["red"] == level if color == "red" else r["blue"] == level)
        )

    out = {}
    for level in DIFFICULTY_PRESETS:
        played = [r for r in rows if level in (r["red"], r["blue"])]
        out[level] = {
            "games": len(played),
            "wins_as_red": wins(level, "red"),
            "wins_as_blue": wins(level, "blue"),
            "avg_plies": round(sum(r["plies"] for r in played) / len(played), 2) if played else 0,
            "illegal_moves": sum(r["illegal_moves"] for r in played),
        }
    return out


def conclusion(summary: dict) -> str:
    """Short text comparing difficulty tiers by win counts in head-to-head matchups."""
    pairs = [
        ("easy", "medium"),
        ("easy", "hard"),
        ("medium", "hard"),
    ]
    lines = []
    for a, b in pairs:
        a_w = sum(
            1
            for r in rows_cache
            if (r["red"] == a and r["winner"] == "red" or r["blue"] == a and r["winner"] == "blue")
            and b in (r["red"], r["blue"])
        )
        b_w = sum(
            1
            for r in rows_cache
            if (r["red"] == b and r["winner"] == "red" or r["blue"] == b and r["winner"] == "blue")
            and a in (r["red"], r["blue"])
        )
        total = a_w + b_w
        if total == 0:
            lines.append(f"{a} vs {b}: 无胜负局，无法判断强弱。")
            continue
        stronger = b if b_w > a_w else a if a_w > b_w else None
        if stronger:
            lines.append(f"{a} vs {b}: {stronger} 胜率更高 ({max(a_w, b_w)}/{total})。")
        else:
            lines.append(f"{a} vs {b}: 胜负接近 ({a_w}-{b_w})。")
    if summary["hard"]["illegal_moves"] or summary["medium"]["illegal_moves"] or summary["easy"]["illegal_moves"]:
        lines.append("警告：实验中出现非法行动，应检查搜索实现。")
    else:
        lines.append("全部对局未出现非法行动。")
    return "\n".join(lines)


rows_cache: list[dict] = []


def main():
    parser = argparse.ArgumentParser(description="Search AI difficulty baseline")
    parser.add_argument("--games", type=int, default=20, help="Games per color assignment per matchup")
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument("--output", default="data/search_baseline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    matchups = [
        ("easy", "medium"),
        ("easy", "hard"),
        ("medium", "hard"),
    ]
    global rows_cache
    rows_cache = []

    for level_a, level_b in matchups:
        for game_idx in range(args.games):
            for swap in (False, True):
                red_level, blue_level = (level_b, level_a) if swap else (level_a, level_b)
                seed = args.seed + game_idx * 10 + (1 if swap else 0) + hash((level_a, level_b)) % 1000
                red = SearchAgent(red_level, seed=seed)
                blue = SearchAgent(blue_level, seed=seed + 1)
                started = time.time()
                result = play_game(red, blue, args.max_plies)
                row = {
                    "_steps": result["steps"],
                    "matchup": f"{level_a}_vs_{level_b}",
                    "game_index": game_idx,
                    "swap_colors": swap,
                    "red": red_level,
                    "blue": blue_level,
                    "winner": result["winner"],
                    "plies": result["plies"],
                    "truncated": result["truncated"],
                    "illegal_moves": result["illegal_moves"],
                    "total_latency_seconds": result["total_latency"],
                    "wall_clock_seconds": time.time() - started,
                }
                rows_cache.append(row)

    steps_path = output / f"baseline_{stamp}_steps.jsonl"
    with steps_path.open("w", encoding="utf-8") as file:
        for row in rows_cache:
            for step in row.get("_steps", []):
                file.write(
                    json.dumps(
                        {
                            "matchup": row["matchup"],
                            "game_index": row["game_index"],
                            "red": row["red"],
                            "blue": row["blue"],
                            **step,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    csv_path = output / f"baseline_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        export_rows = [{k: v for k, v in r.items() if k != "_steps"} for r in rows_cache]
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "matchup",
                "game_index",
                "swap_colors",
                "red",
                "blue",
                "winner",
                "plies",
                "truncated",
                "illegal_moves",
                "total_latency_seconds",
                "wall_clock_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(export_rows)

    summary = summarize(rows_cache)
    text = conclusion(summary)
    config_blob = {name: asdict(DIFFICULTY_PRESETS[name]) for name in DIFFICULTY_PRESETS}
    json_path = output / f"baseline_{stamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "date_utc": stamp,
                "games_per_side": args.games,
                "max_plies": args.max_plies,
                "seed": args.seed,
                "difficulty_config": config_blob,
                "summary": summary,
                "conclusion": text,
                "csv": csv_path.name,
                "steps_jsonl": steps_path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(text)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()