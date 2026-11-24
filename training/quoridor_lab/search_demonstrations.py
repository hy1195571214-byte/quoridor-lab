"""Generate versioned Quoridor demonstrations using local search only."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import random

from .dataset_import import reflection_group
from .rules import Action, State
from .search import DIFFICULTY_PRESETS, SearchAgent


SEARCH_LEVELS = {
    "easy": DIFFICULTY_PRESETS["easy"],
    "medium": replace(DIFFICULTY_PRESETS["medium"], max_depth=2, max_wall_candidates=8),
    "hard": replace(DIFFICULTY_PRESETS["hard"], min_depth=3, max_depth=3,
                    max_wall_candidates=8, time_limit_seconds=None),
}


def make_opening(seed: int, plies: int) -> tuple[str, ...]:
    rng = random.Random(seed)
    state = State()
    for _ in range(plies):
        moves = [Action("move", *square) for square in sorted(state.legal_destinations())]
        safe_moves = [move for move in moves if state.apply(move).winner is None]
        state = state.apply(rng.choice(safe_moves))
    return state.history


def opening_pool(seed: int, count: int, plies: int,
                 excluded: set[str] | None = None) -> list[tuple[str, ...]]:
    seen = set(excluded or ())
    openings = []
    for index in range(100 * count):
        opening = make_opening(seed + index, plies)
        group = reflection_group(opening)
        if group not in seen:
            seen.add(group)
            openings.append(opening)
        if len(openings) == count:
            return openings
    raise ValueError("Could not generate enough distinct openings. Increase the opening length.")


def teacher_game(job: tuple) -> dict:
    opening, levels, seed, split, max_plies = job
    agents = {player: SearchAgent(config=SEARCH_LEVELS[level], seed=seed + player)
              for player, level in enumerate(levels, 1)}
    state = State.from_moves(opening)
    while not state.winner and len(state.history) < max_plies:
        state = state.apply(agents[state.turn].choose(state).action)
    return {
        "seed": seed,
        "split": split,
        "opening": opening,
        "levels": levels,
        "moves": state.history,
        "winner": state.winner,
        "truncated": state.winner is None,
    }


def generate(output: Path, pairs: int, seed: int, workers: int, max_plies: int) -> None:
    openings = opening_pool(seed, pairs, plies=6)
    assignments = [("easy", "medium"), ("medium", "easy")]
    jobs = [
        (opening, levels, seed + 2 * (2 * pair + side), split, max_plies)
        for pair, opening in enumerate(openings)
        for split in ("validation" if pair % 5 == 0 else "train",)
        for side, levels in enumerate(assignments)
    ]
    output.mkdir(parents=True, exist_ok=True)
    target = output / "demonstrations.jsonl"
    if target.exists():
        raise FileExistsError(target)
    config = {
        "schema": 1,
        "seed": seed,
        "pairs": pairs,
        "opening_plies": 6,
        "max_plies": max_plies,
        "teachers": {level: asdict(SEARCH_LEVELS[level]) for level in ("easy", "medium")},
    }
    (output / "demonstrations-config.json").write_text(json.dumps(config, indent=2) + "\n")
    partial = target.with_suffix(".partial")
    seen = set()
    count = 0
    with ProcessPoolExecutor(max_workers=workers) as pool, partial.open("x") as report:
        for index, game in enumerate(pool.map(teacher_game, jobs), 1):
            group = reflection_group(game["moves"])
            if group not in seen:
                seen.add(group)
                report.write(json.dumps(game) + "\n")
                report.flush()
                count += 1
            if index % 8 == 0 or index == len(jobs):
                print(f"Search games: {index}/{len(jobs)} ({count} unique)", flush=True)
    partial.replace(target)
    print(f"Saved {count} demonstration games to {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-plies", type=int, default=256)
    args = parser.parse_args()
    if min(args.pairs, args.workers) < 1 or args.pairs < 5 or args.max_plies <= 6:
        parser.error("Require at least 5 opening pairs, a positive worker count, and max-plies > 6")
    generate(args.output, args.pairs, args.seed, args.workers, args.max_plies)


if __name__ == "__main__":
    main()
