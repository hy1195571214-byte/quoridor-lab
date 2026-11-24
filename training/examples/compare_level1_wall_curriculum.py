"""Compare Masked PPO with random-wall starts and optional BFS potential shaping."""
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from quoridor_lab.dataset_import import reflection_group
from quoridor_lab.fast_observation import fast_action_mask
from quoridor_lab.level1_arena import Game, model_elo, passed
from quoridor_lab.phase1_monitor import read_json, read_rows
from quoridor_lab.rules import State
from quoridor_lab.train_level1 import Experience, evaluate, sha256, update_policy
from quoridor_lab.train_phase1 import ActorCritic, record, save
from quoridor_lab.training_monitor import atomic_json


def wall_opening_bank(arena, count=768):
    """Use training-only pawn prefixes and legal walls from the local rules engine."""
    rng = random.Random(3_401_000)
    depths = (4, 8, 12, 16)
    heldout = {reflection_group(opening) for opening in arena["eval_openings"]}
    states = {}
    for i in range(count * 3):
        prefix = rng.choice(arena["training_openings"])
        if reflection_group(prefix) in heldout:
            raise ValueError("A training opening overlaps the frozen evaluation")
        state = State.from_moves(prefix)
        depth = depths[i % len(depths)]
        for _ in range(depth):
            candidates = np.flatnonzero(fast_action_mask(state)[81:])
            if not len(candidates):
                break
            state = state.apply(81 + int(rng.choice(candidates)))
        if len(state.walls) != depth:
            continue
        key = state.pawns, state.walls, state.turn
        states.setdefault(key, state.history)
        if len(states) == count:
            return list(states.values())
    raise RuntimeError("Could not generate enough distinct legal random-wall starts")


class WallExperience(Experience):
    def __init__(self, arena, config, snapshot, curriculum, walls, wall_fraction):
        super().__init__(arena["training_openings"], config["lanes"], config["seed"], snapshot, curriculum)
        self.walls = walls
        self.wall_fraction = wall_fraction
        self.wall_pairs = 0
        self.new_pairs = 0

    def next_game(self, curriculum_fraction=0.):
        fresh_pair = self.started % 2 == 0
        rng = random.Random(3_412_000 + self.seed + self.started // 2)
        use_walls = fresh_pair and rng.random() < self.wall_fraction
        game = super().next_game(0. if use_walls else curriculum_fraction)
        self.new_pairs += fresh_pair
        if not use_walls:
            return game
        self.wall_pairs += 1
        self.paired_opening = rng.choice(self.walls)
        return Game.start(self.paired_opening, game.player, game.opponent, game.seed, game.index)


def train_arm(args, arm, checkpoint, source, arena, curriculum, bank):
    config, previous = checkpoint["config"], checkpoint["state"]
    model = ActorCritic(args.architecture).to(args.device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=previous["hyper"]["lr"])
    optimizer.load_state_dict(deepcopy(checkpoint["optimizer"]))
    hyper = dict(previous["hyper"])
    if arm == "wall_pbrs":
        hyper["pbrs_scale"] = args.pbrs_scale
    for group in optimizer.param_groups:
        group["lr"] = hyper["lr"]
    fraction = args.wall_fraction if arm != "control" else 0.
    experience = WallExperience(arena, config, previous["rollout"], curriculum, bank, fraction)
    folder = args.output / args.architecture / arm
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {"source_update": previous["update"], "source_sha256": sha256(source),
                "wall_fraction": fraction, "updates": args.updates, "architecture": args.architecture,
                "device": args.device, "hyperparameters": hyper,
                "opponent_pool": config["opponent_pool"], "arena_sha256": config["arena_sha256"]}
    if arm == "wall_pbrs":
        metadata["pbrs"] = {"potential": "negative BFS distance to learner goal edge",
                            "scale": args.pbrs_scale, "gamma": hyper["gamma"], "terminal_potential": 0.}
    atomic_json(folder / "config.json", metadata)
    completed = Counter()
    shaping = Counter()
    started = time.perf_counter()
    for offset in range(1, args.updates + 1):
        update = previous["update"] + offset
        torch.manual_seed(config["seed"] + 1_000_000 + update)
        batch, outcomes, by_opponent, *_ = experience.collect(model, hyper, args.device)
        diagnostics = update_policy(model, optimizer, batch, hyper, args.device)
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise RuntimeError("PPO produced non-finite training diagnostics")
        completed.update(outcomes)
        shaping_metrics = {}
        if arm == "wall_pbrs":
            shaping_metrics = {key: sum(row.get(key, 0) for row in by_opponent.values())
                               for key in ("pbrs_reward", "pbrs_nonzero")}
            shaping.update(shaping_metrics)
        record(folder / "metrics.jsonl", {"architecture": args.architecture, "arm": arm, "update": offset,
               "cumulative_update": update, "actions": offset * hyper["rollout_actions"],
               "new_pairs": experience.new_pairs, "wall_pairs": experience.wall_pairs,
               "rollout_results": outcomes, **shaping_metrics, **diagnostics})
    if args.device == "mps":
        torch.mps.synchronize()
    seconds = time.perf_counter() - started
    weights = model.state_dict()
    actor_delta = math.sqrt(sum(
        (value.cpu() - checkpoint["model"][key]).square().sum().item()
        for key, value in weights.items() if key.startswith(("encoder.", "actor."))
    ))
    if not math.isfinite(actor_delta) or actor_delta == 0:
        raise RuntimeError("The policy parameters did not update")
    scores, _ = evaluate(model, arena, args.device, hyper.get("action_family_balance", 0.))
    result = {**metadata, "learner_actions": args.updates * hyper["rollout_actions"],
              "new_pairs": experience.new_pairs, "wall_pairs": experience.wall_pairs,
              "actor_encoder_l2_change": actor_delta, "training_seconds": seconds,
              "rollout_outcomes": dict(completed), "formal_eval": scores,
              "elo": model_elo(scores, arena["anchor_ratings"]), "gate_passed": passed(scores)}
    if arm == "wall_pbrs":
        result["shaping_totals"] = dict(shaping)
    save(folder / "latest.pt", model=weights, optimizer=optimizer.state_dict(), experiment=result,
         rollout=experience.snapshot())
    atomic_json(folder / "result.json", result)
    print("PURE NETWORK RESULT", args.architecture, arm, result, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=("mlp", "cnn32"), required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--wall-fraction", type=float, default=.35)
    parser.add_argument("--arm", choices=("control", "wall_curriculum", "wall_pbrs"))
    parser.add_argument("--pbrs-scale", type=float, default=.02)
    args = parser.parse_args()
    if not (args.updates > 0 and args.threads > 0 and 0 < args.wall_fraction <= 1):
        parser.error("Positive updates and threads and a wall fraction in (0, 1] are required")
    if not math.isfinite(args.pbrs_scale) or args.pbrs_scale <= 0:
        parser.error("--pbrs-scale must be positive and finite")
    torch.set_num_threads(args.threads)
    arena = read_json(args.source / "arena.json")
    curriculum = read_json(args.source / "curriculum.json")["openings"]
    source = args.source / f"{args.architecture}-seed0" / f"checkpoint-{args.checkpoint:04d}.pt"
    checkpoint = torch.load(source, weights_only=True, map_location="cpu")
    if (checkpoint["state"]["update"] != args.checkpoint or checkpoint["config"]["architecture"] != args.architecture
        or sha256(args.source / "arena.json") != checkpoint["config"]["arena_sha256"]):
        raise ValueError("Checkpoint or Arena does not match the requested comparison")
    bank = wall_opening_bank(arena)
    for arm in ((args.arm,) if args.arm else ("control", "wall_curriculum")):
        train_arm(args, arm, checkpoint, source, arena, curriculum, bank)


if __name__ == "__main__":
    main()
