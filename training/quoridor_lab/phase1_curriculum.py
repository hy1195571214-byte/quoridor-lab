"""Resume the later Phase 1 curriculum levels in 500-game units."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
from random import Random
import signal
import time

import numpy as np
import torch

from .dataset_import import reflection_group
from .fast_observation import fast_action_mask, training_observation
from .level1_arena import (ANCHORS, DEEP_SEARCH, balance_action_families, bot_action, model_elo, passed)
from .phase1_monitor import PROGRESS, phase1_progress, read_json, read_rows
from .rules import Action, State
from .search_demonstrations import opening_pool
from .train_level1 import (Experience, archive, evaluate, sha256, update_policy, write_games)
from .train_phase1 import ActorCritic, record, save
from .training_monitor import atomic_json


LEVEL_NAMES = ("低深度搜索 AI", "中深度搜索 AI", "高深度搜索 AI", "混合搜索对手池", "历史神经网络与搜索 AI")
GAMES_PER_UNIT = 500
REGRESSION_OPPONENTS = ANCHORS
LEVEL3_PRACTICE = "level3_practice"
LEVEL_POOLS = {
    2: {"level2": 4, "random": 1},
    3: {"level3": 4, "random": 1},
    4: {"level1": 3, "level2": 3, "level3": 2, "random": 2},
    5: {"history": 5, "level1": 1, "level2": 1, "level3": 1, "random": 2},
}
LEVEL_GATES = {
    2: {"level2": .70},
    3: {"level3": .70},
    4: {"level1": .70, "level2": .70, "level3": .70},
    5: {"history": .70, "level3": .70},
}


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def local_stage_paths(plan, runs_root):
    """Locate a restored course without changing the authenticated stage plan."""
    arena = Path(plan["arena"])
    original_root = arena.parent.parent
    runs_root = runs_root.resolve()
    if not (runs_root / arena.relative_to(original_root)).is_file():
        return plan

    def relocate(item):
        path = Path(item["checkpoint"])
        if path.is_relative_to(original_root):
            path = runs_root / path.relative_to(original_root)
        return {**item, "checkpoint": str(path)}

    return {**plan, "arena": str(runs_root / arena.relative_to(original_root)),
            "sources": {key: relocate(value) for key, value in plan["sources"].items()},
            "historical": {key: relocate(value) for key, value in plan["historical"].items()}}


def search_reply(request):
    return bot_action(*request)


def resolve_practice(state, opponent, seed):
    if opponent == LEVEL3_PRACTICE:
        draw = Random(seed ^ (len(state.history) * 2654435761)).random()
        opponent = "bfs" if draw < .65 else "level3"
    return state, opponent, seed


class Opponents:
    """Frozen local search and checkpoint policies, shared across a paired opening."""

    def __init__(self, historical, search_pool=None):
        self.search_pool = search_pool
        self.models = {}
        for name, item in historical.items():
            path = Path(item["checkpoint"])
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Historical opponent changed: {path}")
            saved = checkpoint(path)
            model = ActorCritic(saved["config"]["architecture"])
            model.load_state_dict(saved["model"])
            self.models[name] = model.eval(), saved["state"]["hyper"].get("action_family_balance", 0.)
        self.history = tuple(sorted(self.models))

    def __call__(self, state, opponent, seed):
        state, opponent, seed = resolve_practice(state, opponent, seed)
        if opponent == "history":
            opponent = self.history[seed % len(self.history)]
        if opponent not in self.models:
            return bot_action(state, opponent, seed)
        return self.model_actions(opponent, [state])[0]

    def model_actions(self, opponent, states):
        model, balance = self.models[opponent]
        observations = [training_observation(state) for state in states]
        board = torch.from_numpy(np.stack([obs["board"] for obs in observations]))
        mask = torch.from_numpy(np.stack([obs["action_mask"] for obs in observations])).bool()
        with torch.inference_mode():
            logits, _ = model(board, mask)
            logits = balance_action_families(logits, mask, balance)
            actions = logits.argmax(dim=-1).tolist()
            if len(states) > 1:
                # Check near ties individually so rounding from batched matrix math
                # cannot change how a saved opponent breaks a tie.
                leading = logits.topk(2, dim=-1).values
                ties = (leading[:, 0] - leading[:, 1] <= 1e-4).nonzero().flatten().tolist()
                for index in ties:
                    actions[index] = self.model_actions(opponent, [states[index]])[0]
            return actions

    def batch(self, requests):
        requests = [resolve_practice(*request) for request in requests]
        deep = [(index, request) for index, request in enumerate(requests) if request[1] in DEEP_SEARCH]
        replies = [None] * len(requests)
        if self.search_pool is not None and deep:
            results = self.search_pool.map(search_reply, (request for _, request in deep), chunksize=2)
            for (index, _), result in zip(deep, results, strict=True):
                replies[index] = result
        model_requests = {}
        for index, (state, opponent, seed) in enumerate(requests):
            if replies[index] is not None:
                continue
            if opponent == "history":
                opponent = self.history[seed % len(self.history)]
            if opponent in self.models:
                model_requests.setdefault(opponent, []).append((index, state))
            else:
                replies[index] = bot_action(state, opponent, seed)
        for opponent, pending in model_requests.items():
            indices, states = zip(*pending)
            for index, action in zip(indices, self.model_actions(opponent, states), strict=True):
                replies[index] = action
        return replies


def frozen_arena(path, level1, pilot):
    if path.exists():
        return read_json(path)
    source = read_json(level1 / "arena.json")
    curriculum = read_json(level1 / "curriculum.json")
    seen = {reflection_group(opening) for opening in
            [*source["training_openings"], *source["eval_openings"], *curriculum["openings"]]}
    seen.update(reflection_group(game["opening"]) for game in read_rows(pilot / "demonstrations.jsonl"))
    # Three disjoint gate sets support the documented three consecutive passes.
    banks = []
    for index in range(4):
        bank = opening_pool(3_100_000 + index * 100_000, 100, 8, seen)
        banks.append(bank)
        seen.update(reflection_group(opening) for opening in bank)
    bundle = {"protocol": "phase1-curriculum-arena-v1", "level1_arena_sha256": sha256(level1 / "arena.json"),
              "training_openings": source["training_openings"], "curriculum_openings": curriculum["openings"],
              "gate_openings": banks[:3], "final_openings": banks[3], "anchor_ratings": source["anchor_ratings"],
              "eval_seed": 3_600_000, "search": {name: asdict(preset) for name, preset in DEEP_SEARCH.items()}}
    atomic_json(path, bundle)
    return read_json(path)


def evaluation_openings(arena, index):
    return {"eval_openings": arena["gate_openings"][index % len(arena["gate_openings"])],
            "eval_seed": arena["eval_seed"] + 10_000 * (index % len(arena["gate_openings"]))}


def gate_policy(folder, thresholds, *, original_only=False):
    original = {"wins": thresholds, "truncation_limit": .05}
    if original_only:
        return original, original, None
    adjustment = read_json(folder / "gate-adjustment.json")
    if not adjustment:
        return original, original, None
    adjusted = adjustment["adjusted"]
    limits = adjusted.get("truncation_limits", {})
    if (set(adjusted["wins"]) != set(original["wins"]) or
        any(not 0 < value <= original["wins"][name] for name, value in adjusted["wins"].items()) or
        not isinstance(limits, dict) or not limits.keys() <= thresholds.keys() or
        any(not original["truncation_limit"] <= limit < 1 for limit in limits.values()) or
        not original["truncation_limit"] <= adjusted["truncation_limit"] < 1 or not adjustment["reason"]):
        raise ValueError("Invalid gate adjustment")
    return original, adjusted, adjustment["reason"]


def training_adjustment(folder, hyper, update):
    request = read_json(folder / "curriculum-adjustment.json")
    if not request:
        return None
    after = request["fraction"]
    before = hyper.get("curriculum_fraction", 0.)
    if not isinstance(after, (int, float)) or not 0 <= after <= 1 or not request["reason"]:
        raise ValueError("Curriculum adjustments require a fraction in [0, 1] and a recorded reason")
    if before != after:
        hyper["curriculum_fraction"] = after
        return {"parameter": "curriculum_fraction", "before": before, "after": after,
                "reason": request["reason"], "from_update": update}
    before = hyper.get("pbrs_potential", "own_distance")
    after = request.get("pbrs_potential", before)
    if before != after:
        if after not in {"own_distance", "distance_margin"} or not request.get("pbrs_reason"):
            raise ValueError("PBRS potential adjustments require a supported potential and a recorded reason")
        hyper["pbrs_potential"] = after
        return {"parameter": "pbrs_potential", "before": before, "after": after,
                "reason": request["pbrs_reason"], "from_update": update}
    before = hyper.get("entropy_coef")
    after = request.get("entropy_coef", before)
    if before != after:
        if not isinstance(after, (int, float)) or not np.isfinite(after) or after < 0 or not request.get("entropy_reason"):
            raise ValueError("Entropy adjustments require a finite nonnegative coefficient and a recorded reason")
        hyper["entropy_coef"] = after
        return {"parameter": "entropy_coef", "before": before, "after": after,
                "reason": request["entropy_reason"], "from_update": update}
    before = hyper.get("level3_practice", False)
    after = request.get("level3_practice", before)
    if before != after:
        if not isinstance(after, bool) or not request.get("practice_reason"):
            raise ValueError("Level 3 practice requires an on/off setting and a recorded reason")
        hyper["level3_practice"] = after
        return {"parameter": "level3_practice", "before": before, "after": after,
                "reason": request["practice_reason"], "from_update": update}
    before = hyper.get("repeat_temperature", 1.)
    after = request.get("repeat_temperature", before)
    if before != after:
        if (not isinstance(after, (int, float)) or not np.isfinite(after) or after < 1.
                or not request.get("repeat_temperature_reason")):
            raise ValueError("Repeat exploration requires a finite temperature of at least one and a recorded reason")
        hyper["repeat_temperature"] = after
        return {"parameter": "repeat_temperature", "before": before, "after": after,
                "reason": request["repeat_temperature_reason"], "from_update": update}
    before = hyper.get("mirror_rollouts", False)
    after = request.get("mirror_rollouts", before)
    if before != after:
        if not isinstance(after, bool) or not request.get("mirror_reason"):
            raise ValueError("Mirrored rollouts require an on/off setting and a recorded reason")
        hyper["mirror_rollouts"] = after
        return {"parameter": "mirror_rollouts", "before": before, "after": after,
                "reason": request["mirror_reason"], "from_update": update}
    before = hyper.get("level3_retention", False)
    after = request.get("level3_retention", before)
    if before == after:
        return None
    if not isinstance(after, bool) or not request.get("retention_reason"):
        raise ValueError("Level 3 retention requires an on/off setting and a recorded reason")
    hyper["level3_retention"] = after
    return {"parameter": "level3_retention", "before": before, "after": after,
            "reason": request["retention_reason"], "from_update": update}


def make_red_finish_bank(arena, destination, count=148):
    """Generate paired finish practice from legal play on training-only openings."""
    base = arena["training_openings"]
    distinct = {}
    for index in range(4 * len(base)):
        position = State.from_moves(base[index % len(base)])
        rng = Random(6_410_000 + index)
        while position.winner is None and len(position.history) < 100:
            if position.turn == 2 and not position.remaining[1] and position.distance(2) >= 2 and position.distance(1) <= 2:
                replies = [position.apply(Action("move", row, col)) for row, col in position.legal_destinations()]
                if replies and all(child.winner is None and any(row == 0 for row, _ in child.legal_destinations())
                                   for child in replies):
                    distinct.setdefault(reflection_group(position.history), position.history)
                    break
            if position.turn == 2:
                walls = np.flatnonzero(fast_action_mask(position)[81:])
                action = 81 + int(rng.choice(walls)) if len(walls) else bot_action(position, "random", 6_430_000 + index)
            elif position.remaining[1] and any(row == 0 for row, _ in position.legal_destinations()):
                alternatives = sorted(row * 9 + col for row, col in position.legal_destinations() if row)
                if not alternatives:
                    break
                action = rng.choice(alternatives)
            else:
                action = bot_action(position, "bfs", 6_450_000 + index)
            position = position.apply(action)
        if len(distinct) == count:
            break
    if len(distinct) < count:
        raise RuntimeError("Not enough distinct training-only positions with a red finish")
    atomic_json(destination, {"generator": "training-only-red-finishes-v1", "seed": 6_410_000,
                "openings": list(distinct.values())})


def training_openings(folder, openings, state):
    profile = read_json(folder / "curriculum-adjustment.json")
    focus = profile.get("opening_focus", "all")
    if focus in {"red_ahead", "red_finish", "red_midgame"}:
        filtered = []
        for opening in openings:
            position = State.from_moves(opening)
            red, blue = position.distance(1), position.distance(2)
            if red <= 3 and blue >= red + 2:
                filtered.append(opening)
        openings = filtered
    elif focus in {"search_finishes", "search_wall_tactics", "search_finish_progression"}:
        bank = folder.parent / "search-finish-openings.json"
        if sha256(bank) != profile["search_finish_bank_sha256"]:
            raise ValueError("Search finish training opening bank changed")
        limit = profile["max_reference_plies"]
        openings = [item["opening"] for item in read_json(bank)["positions"]
                    if item["reference_plies_remaining"] <= limit]
        if focus != "search_finishes":
            walls = folder.parent / "search-wall-openings.json"
            if sha256(walls) != profile["search_wall_bank_sha256"]:
                raise ValueError("Search wall training opening bank changed")
            openings.extend(item["opening"] for item in read_json(walls)["positions"])
        if focus == "search_finish_progression":
            progression = folder.parent / "mlp-finish-progression-openings.json"
            if sha256(progression) != profile["finish_progression_bank_sha256"]:
                raise ValueError("Finish progression training opening bank changed")
            openings.extend(item["opening"] for item in read_json(progression)["positions"])
    elif focus in {"search_loop_escapes", "search_loop_finish_progression"}:
        name = ("cnn32-loop-finish-openings.json" if focus == "search_loop_finish_progression"
                else "cnn32-loop-escape-openings.json")
        bank = folder.parent / name
        if sha256(bank) != profile["loop_escape_bank_sha256"]:
            raise ValueError("Loop escape training opening bank changed")
        escapes = [item["opening"] for item in read_json(bank)["positions"]]
        if not escapes:
            raise ValueError("Loop escape training opening bank is empty")
        copies = max(1, len(openings) // (2 * len(escapes)))
        openings = [*openings, *(escapes * copies)]
    elif focus != "all":
        raise ValueError(f"Unknown training-only opening focus: {focus}")
    if focus in {"red_finish", "red_midgame"}:
        bank = folder / "red-finish-openings.json"
        if sha256(bank) != profile["opening_bank_sha256"]:
            raise ValueError("Red finish training opening bank changed")
        seconds = read_json(bank)["openings"]
        openings = [*openings, *seconds]
    if focus == "red_midgame":
        bank = folder / "red-midgame-openings.json"
        if sha256(bank) != profile["midgame_bank_sha256"]:
            raise ValueError("Red midgame training opening bank changed")
        openings.extend(item["opening"] for item in read_json(bank)["positions"])
    if not openings:
        raise ValueError("Training-only opening focus has no positions")
    previous = state.get("curriculum_focus", "all")
    state["curriculum_focus"] = focus
    tuning = None if previous == focus else {
        "parameter": "curriculum_openings", "before": previous, "after": focus,
        "reason": profile["opening_focus_reason"], "from_update": state["update"] + 1,
        "positions": len({tuple(opening) for opening in openings}),
    }
    return openings, tuning


def recover_policy(folder, saved):
    request = read_json(folder / "policy-recovery.json")
    if not request:
        return saved, None
    state = saved["state"]
    revision = request["checkpoint_sha256"]
    if revision == state.get("policy_recovery_sha256"):
        return saved, None
    update = request["checkpoint_update"]
    if state["qualified"] or not 0 < update < state["update"] or not request["reason"]:
        raise ValueError("Policy recovery requires an earlier checkpoint from an active run and a recorded reason")
    source = folder / f"checkpoint-{update:05d}.pt"
    candidate = request.get("local_ppo_candidate")
    if not candidate and sha256(source) != revision:
        raise ValueError("Policy recovery checkpoint changed")
    recovered = checkpoint(source)
    if recovered["config"] != saved["config"] or recovered["state"]["update"] != update:
        raise ValueError("Policy recovery must use the same frozen run")
    recovery_hyper = state["hyper"]
    reference = request.get("reference_kl_coef")
    if candidate:
        source_hash = sha256(source)
        if (len(revision) != 64 or any(c not in "0123456789abcdef" for c in revision) or
            candidate != f"local-ppo-recovery-{revision}.pt" or sha256(folder / candidate) != revision):
            raise ValueError("Local PPO continuation checkpoint changed")
        recovered = checkpoint(folder / candidate)
        evidence = recovered["state"]
        if reference is not None:
            lr = request["learning_rate"]
            if (saved["config"].get("phase1_level") not in (3, 4) or not np.isfinite(reference) or reference <= 0. or
                not np.isfinite(lr) or lr <= 0.):
                raise ValueError("The verified PPO continuation needs a supported level, reference coefficient, and rate")
            recovery_hyper = {**recovery_hyper, "lr": lr, "reference_kl_coef": reference}
        if (recovered["config"]["source_sha256"] != source_hash or
            recovered["config"]["hyperparameters"] != recovery_hyper or
            evidence["phase"] != "ppo" or evidence["update"] <= 0 or evidence["completed"] < GAMES_PER_UNIT):
            raise ValueError("Local continuation must come from the same run and complete a PPO game unit")
    if any(group["lr"] != recovery_hyper["lr"] for group in recovered["optimizer"]["param_groups"]):
        raise ValueError("Recovery optimizer and PPO learning rates disagree")
    state["hyper"] = recovery_hyper
    state["policy_recovery_sha256"] = revision
    state["streak"] = 0
    change = {"parameter": "policy_checkpoint", "before": state["update"], "after": update,
              "reason": request["reason"], "from_update": state["update"] + 1}
    if candidate:
        change.update(parameter="local_ppo_checkpoint", after="本地纠错后接续 PPO",
                      continuation_games=evidence["completed"], continuation_updates=evidence["update"],
                      source_update=update, checkpoint_sha256=revision)
        if reference is not None:
            state["policy_reference"] = {"update": update, "sha256": source_hash}
            change.update(after="带参考策略约束的 PPO", reference_kl_coef=reference, learning_rate=recovery_hyper["lr"])
    return {**saved, "model": recovered["model"], "optimizer": recovered["optimizer"]}, change


def train_stage(root, architecture):
    plan = read_json(root / "level.json")
    level = plan["level"]
    local = local_stage_paths(plan, root.parent)
    arena_path = Path(local["arena"])
    source = plan["sources"][architecture]
    source_path = Path(local["sources"][architecture]["checkpoint"])
    if sha256(arena_path) != plan["arena_sha256"] or sha256(source_path) != source["sha256"]:
        raise ValueError("Frozen Arena or preceding checkpoint changed")
    arena = read_json(arena_path)
    if arena["search"] != {name: asdict(preset) for name, preset in DEEP_SEARCH.items()}:
        raise ValueError("The frozen search configuration does not match the course")
    saved = checkpoint(source_path)
    if saved["config"]["architecture"] != architecture or source["level"] != level - 1:
        raise ValueError("A level must inherit the same architecture from its preceding level")
    seed, device, lanes, threads = (saved["config"][key] for key in ("seed", "device", "lanes", "threads"))
    torch.set_num_threads(threads)
    counts = plan["opponent_weights"]
    pool = tuple(name for name, count in counts.items() for _ in range(count))
    config = {**saved["config"], "phase1_level": level, "stage_plan_sha256": sha256(root / "level.json"),
              "arena_sha256": plan["arena_sha256"], "stage_source": source, "start_level": f"level{level}_pool",
              "opponent_pool": {name: count / len(pool) for name, count in counts.items()}}
    config.pop("comparison", None)
    folder = root / f"{architecture}-seed{seed}"
    folder.mkdir(exist_ok=True)
    latest = folder / "latest.pt"
    metrics = folder / "metrics.jsonl"
    search_workers = max(1, min(3, (os.cpu_count() or 2) // 3))
    with phase1_progress(folder, 0, None) as progress, ProcessPoolExecutor(
        max_workers=search_workers, mp_context=multiprocessing.get_context("spawn")
    ) as search_pool:
        resolve = Opponents(local["historical"], search_pool)
        state = {"update": 0, "steps": 0, "hyper": dict(saved["state"]["hyper"]), "qualified": False,
                 "streak": 0, "best_rank": None, "best_update": 0, "rollout": None, "event": None,
                 "game_units": {"size": GAMES_PER_UNIT, "started_at_update": 0, "completed": 0,
                                "next_evaluation": GAMES_PER_UNIT}, "evaluation_count": 0}
        if level != 3:
            state["hyper"].pop("level3_practice", None)
            state["hyper"].pop("level3_retention", None)
            state["hyper"].pop("repeat_temperature", None)
            state["hyper"].pop("mirror_rollouts", None)
            state["hyper"].pop("reference_kl_coef", None)
        if latest.is_file():
            saved = checkpoint(latest)
            if saved["config"] != config:
                raise ValueError("Resume requires the same frozen Level configuration")
            state = saved["state"]
            last = state["event"]
            if last and last.get("formal_eval"):
                archive(latest, folder / f"checkpoint-{state['update']:05d}.pt")
                if state["best_update"] == state["update"]:
                    archive(latest, folder / "best.pt")
            history = read_rows(metrics)
            if last and (not history or history[-1]["update"] < last["update"]):
                record(metrics, last)
        saved, recovery = recover_policy(folder, {**saved, "state": state})
        model = ActorCritic(architecture).to(device)
        model.load_state_dict(saved["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=state["hyper"]["lr"])
        optimizer.load_state_dict(saved["optimizer"])
        opts = state["hyper"]
        reference = None
        if anchor := state.get("policy_reference"):
            path = folder / f"checkpoint-{anchor['update']:05d}.pt"
            if sha256(path) != anchor["sha256"]:
                raise ValueError("Frozen PPO reference checkpoint changed")
            reference = ActorCritic(architecture).to(device).eval()
            reference.load_state_dict(checkpoint(path)["model"])
        curriculum, opening_change = training_openings(folder, arena["curriculum_openings"], state)
        experience = Experience(arena["training_openings"], lanes, seed + level * 10_000, state["rollout"],
                                curriculum, pool, resolve)
        units = state["game_units"]
        progress.update(game_units=dict(units), required_passes=3, pbrs_scale=opts.get("pbrs_scale", 0.),
                        qualified=state["qualified"],
                        gate_adjustment_reason=(state.get("event") or {}).get("gate_adjustment_reason"))
        atomic_json(folder / "config.json", config)
        if not latest.exists():
            save(latest, model=model.state_dict(), optimizer=optimizer.state_dict(), config=config, state=state)
        atomic_json(folder / PROGRESS, {**progress, "status": "training", "activity": "collecting"})
        while not state["qualified"]:
            number = state["update"] + 1
            adjustment = recovery or opening_change or training_adjustment(folder, opts, number)
            if recovery:
                recovery = None
            else:
                opening_change = None
            retention = level == 3 and opts.get("level3_retention", False)
            practice = level == 3 and opts.get("level3_practice", False) and not retention
            experience.pool_replacements = {"random": LEVEL3_PRACTICE} if practice else {}
            if retention:
                experience.pool_replacements = {"random": "bfs"}
            experience.alternate_pool_replacements = {"random": "level1"} if retention else {}
            experience.standard_only_opponents = {"bfs", "level1"} if retention else set()
            torch.manual_seed(seed + 2_000_000 + level * 100_000 + number)
            began = time.perf_counter()
            batch, outcomes, by_opponent, starts, partial = experience.collect(model, opts, device)
            sampled = time.perf_counter()
            diagnostics = update_policy(model, optimizer, batch, opts, device, reference=reference)
            ended = time.perf_counter()
            state["update"] = number
            state["steps"] += opts["rollout_actions"]
            units["completed"] += sum(outcomes.values())
            event = {"phase": "ppo", "update": number, "steps": state["steps"], "opponent": config["start_level"],
                     "device": device, "rollout_results": outcomes, "by_opponent": by_opponent, "game_starts": starts,
                     "learner_illegal_actions": 0, "active_games_bootstrapped": partial, "hyperparameters": dict(opts),
                     "seconds": round(ended - began, 4), "rollout_seconds": round(sampled - began, 4),
                     "optimizer_seconds": round(ended - sampled, 4),
                     "actions_per_second": round(opts["rollout_actions"] / (ended - began), 2), **diagnostics}
            if adjustment:
                event["tuning"] = adjustment
            if practice:
                practice_share = config["opponent_pool"]["random"] / 2
                event["training_opponent_pool"] = {**config["opponent_pool"], "random": practice_share,
                                                   LEVEL3_PRACTICE: practice_share}
            elif retention:
                other = config["opponent_pool"]["random"] / 2
                event["training_opponent_pool"] = {"level3": config["opponent_pool"]["level3"], "bfs": other, "level1": other}
            selected = False
            if units["completed"] >= units["next_evaluation"]:
                atomic_json(folder / PROGRESS, {**progress, "status": "training", "activity": "evaluating",
                                               "eval_update": number, "game_units": dict(units)})
                index = state["evaluation_count"]
                evaluation = evaluation_openings(arena, index)
                original, gate, reason = gate_policy(
                    folder, plan["gates"], original_only=plan.get("original_gates_only", False)
                )
                names = tuple(dict.fromkeys((*REGRESSION_OPPONENTS, *gate["wins"])))
                eval_start = time.perf_counter()
                scores, games = evaluate(model, evaluation, device, opts.get("action_family_balance", 0.), names, resolve)
                standard_passed = passed(scores, original["wins"], original["truncation_limit"])
                active_passed = passed(scores, gate["wins"], gate["truncation_limit"],
                                       truncation_limits=gate.get("truncation_limits"))
                if state.get("last_gate") != gate:
                    state["streak"] = 0
                state["last_gate"] = gate
                state["streak"] = state["streak"] + 1 if active_passed else 0
                state["evaluation_count"] += 1
                units["next_evaluation"] = (units["completed"] // GAMES_PER_UNIT + 1) * GAMES_PER_UNIT
                elo = model_elo({name: scores[name] for name in ANCHORS}, arena["anchor_ratings"])
                state["qualified"] = state["streak"] >= 3
                score_rank = (standard_passed, active_passed,
                              min(scores[name]["win_rate"] for name in gate["wins"]),
                              -max(scores[name]["truncation_rate"] for name in gate["wins"]), elo)
                if state["best_rank"] is None or score_rank > tuple(state["best_rank"]):
                    state["best_rank"], state["best_update"] = score_rank, number
                    selected = True
                event.update(formal_eval=scores, development_eval=scores["level1"], shortest_eval=scores["bfs"],
                             gate_passed=active_passed, original_gate_passed=standard_passed, gate_policy=gate,
                             gate_adjustment_reason=reason, qualifying_streak=state["streak"],
                             required_passes=3, qualified=state["qualified"], elo=elo, opening_bank=index % 3 + 1,
                             eval_seconds=round(time.perf_counter() - eval_start, 3))
                write_games(folder / f"ppo_{number:04d}_games.jsonl", games)
            event["game_units"] = dict(units)
            state["rollout"], state["event"] = experience.snapshot(), event
            save(latest, model=model.state_dict(), optimizer=optimizer.state_dict(), config=config, state=state)
            if "formal_eval" in event:
                archive(latest, folder / f"checkpoint-{number:05d}.pt")
                if selected:
                    archive(latest, folder / "best.pt")
            record(metrics, event)
            progress.update(activity="collecting", game_units=dict(units), qualified=state["qualified"])
            if "formal_eval" in event:
                progress["gate_adjustment_reason"] = event["gate_adjustment_reason"]
                atomic_json(folder / PROGRESS, {**progress, "status": "training"})
        archive(latest, folder / "qualified.pt")
        progress.update(ppo_total=state["update"], qualified_update=state["update"])
        atomic_json(folder / "summary.json", {"level": level, "architecture": architecture, "qualified": True,
                    "updates": state["update"], "qualified_update": state["update"], "best_update": state["best_update"],
                    "game_units": units, "gate_policy": state["event"]["gate_policy"],
                    "gate_adjustment_reason": state["event"]["gate_adjustment_reason"],
                    "finished_at": datetime.now(timezone.utc).isoformat()})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--architecture", choices=("mlp", "cnn32"), required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    train_stage(args.root, args.architecture)


if __name__ == "__main__":
    main()
