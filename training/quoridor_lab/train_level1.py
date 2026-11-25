"""Continue a verified local Phase 1 pilot under the frozen Level 1 protocol."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.distributions import Categorical

from .dataset_import import reflection_group
from .fast_observation import BOTTOM, TOP
from .level1_arena import (ANCHORS, MAX_PLIES, POLICIES, POOL, THRESHOLDS, Game, bot_action,
                           balance_action_families, bit_distance, edges_for, estimate_elo, evaluation_games,
                           model_elo, observations, passed, pawn_bit, play_evaluation, score_games, step_batch)
from .phase1_monitor import PROGRESS, phase1_progress, read_json, read_rows
from .rules import Action, State
from .search_demonstrations import opening_pool
from .train_phase1 import ActorCritic, record, save
from .training_monitor import atomic_json

PROTOCOL = "phase1-level1-v1"
SCHEDULE = {"initial": 100, "limit": 500, "evaluate_every": 25, "eval_games": 200}
HYPER = {"rollout_actions": 2048, "minibatch": 256, "update_epochs": 5,
         "lr": 3e-4, "gamma": .99, "gae_lambda": .95, "clip": .2,
         "entropy_coef": .01, "value_coef": .5, "max_grad_norm": .5}
REWARD = {"wins": 1., "losses": -1., "truncated": 0., None: 0.}
UNIT_UPDATES = 500
RANDOM_TRAINING_POOL = ("level1",) * 14 + ("bfs",) * 4 + ("randomized_search",) * 2 + ("random",) * 5
HORIZONTAL_ACTIONS = torch.tensor([
    Action(move.kind, move.row, (8 if move.kind == "move" else 7) - move.col).id
    for move in map(Action.from_id, range(209))
])


def next_unit_target(update):
    return (update // UNIT_UPDATES + 1) * UNIT_UPDATES


def bfs_goal_potential(state, player, scale, *, distance_margin=False):
    goal, opponent_goal = (TOP, BOTTOM) if player == 1 else (BOTTOM, TOP)
    edges = edges_for(state.walls)
    own = bit_distance(pawn_bit(state.pawns[player - 1]), goal, edges)
    opponent = bit_distance(pawn_bit(state.pawns[2 - player]), opponent_goal, edges) if distance_margin else 0
    return scale * (opponent - own)


def repeats_pawn_moves(history):
    """Recognize a repeated even-length sequence without any wall placements."""
    for length in (4, 6, 8):
        if (len(history) >= 2 * length and history[-2 * length:-length] == history[-length:]
                and all(len(move) == 2 for move in history[-length:])):
            return True
    return False


def rollout_observations(games, mirror_rollouts):
    boards, masks = observations(games)
    if not mirror_rollouts:
        return boards, masks, None
    mirrored = torch.tensor([bool(random.Random(5_430_000 + game.seed).getrandbits(1)) for game in games])
    source = boards[mirrored]
    view = source.flip(-1)
    # Plane 3 encodes the left anchor of a vertical wall on an eight-column grid.
    view[:, 3, :, :8] = source[:, 3, :, :8].flip(-1)
    view[:, 3, :, 8] = source[:, 3, :, 8]
    boards[mirrored] = view
    masks[mirrored] = masks[mirrored][:, HORIZONTAL_ACTIONS]
    return boards, masks, mirrored


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pilot_checkpoint(pilot, architecture):
    folder = pilot / f"{architecture}-seed0"
    checkpoint_path = folder / "latest.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    config = checkpoint["config"]
    if config["architecture"] != architecture or config["seed"] != 0:
        raise ValueError(f"Wrong pilot architecture or seed: {checkpoint_path}")
    dataset = pilot / "demonstrations.jsonl"
    if sha256(dataset) != config["demonstrations_sha256"]:
        raise ValueError("Pilot checkpoint does not match the local search demonstration file")
    history = read_rows(folder / "metrics.jsonl")
    tail = next(row for row in reversed(history) if row["phase"] == "ppo")
    if (tail["update"], tail["steps"]) != (checkpoint["update"], checkpoint["steps"]):
        raise ValueError("Pilot checkpoint and PPO log disagree")
    if any(not torch.isfinite(tensor).all().item() for tensor in checkpoint["model"].values()):
        raise ValueError("Pilot contains non-finite model weights")
    if not checkpoint["optimizer"]["state"]:
        raise ValueError("Pilot Adam state is empty")
    for state in checkpoint["optimizer"]["state"].values():
        if any(torch.is_tensor(value) and not torch.isfinite(value).all().item() for value in state.values()):
            raise ValueError("Pilot contains non-finite optimizer state")
    inherited = {"checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": sha256(checkpoint_path),
                 "pilot_updates": checkpoint["update"], "pilot_actions": checkpoint["steps"],
                 "demonstrations_sha256": config["demonstrations_sha256"]}
    return checkpoint, inherited


def prepare(root, pilot):
    root.mkdir(parents=True, exist_ok=True)
    arena_file = root / "arena.json"
    if arena_file.exists():
        prepare_curriculum(root, json.loads(arena_file.read_text()))
        return
    excluded = {reflection_group(row["opening"]) for row in read_rows(pilot / "demonstrations.jsonl")}
    for architecture in ("mlp", "cnn32"):
        for path in (pilot / f"{architecture}-seed0").glob("*games.jsonl"):
            excluded.update(reflection_group(row["opening"]) for row in read_rows(path))
    evaluation = opening_pool(910_000, SCHEDULE["eval_games"] // 2, 6, excluded)
    excluded.update(reflection_group(opening) for opening in evaluation)
    training = opening_pool(920_000, 180, 6, excluded)
    matches = {}
    for left, right in (("random", "bfs"), ("random", "level1"), ("level1", "bfs")):
        games = play_evaluation(evaluation_games(evaluation, right, 930_000), "cpu", bot=left)
        matches[left, right] = score_games(games)
        print("Arena calibration", left, right, matches[left, right], flush=True)
    atomic_json(arena_file, {"protocol": PROTOCOL, "opponent_policies": POLICIES,
                             "eval_seed": 940_000, "eval_openings": evaluation, "training_openings": training,
                             "anchor_ratings": estimate_elo(matches),
                             "calibration": [{"model": left, "opponent": right, **score}
                                             for (left, right), score in matches.items()]})
    prepare_curriculum(root, json.loads(arena_file.read_text()))


def prepare_curriculum(root, arena):
    """Generate paired near-goal starts from the training-only opening bank."""
    target = root / "curriculum.json"
    if target.is_file():
        return
    starts = {}
    holdout = {reflection_group(opening) for opening in arena["eval_openings"]}
    for index, opening in enumerate(arena["training_openings"]):
        state = State.from_moves(opening)
        taken = 0
        while state.winner is None and len(state.history) < 44 and taken < 4:
            goal = 0 if state.turn == 1 else 8
            immediate = any(row == goal for row, _ in state.legal_destinations())
            if not immediate and min(state.distance(1), state.distance(2)) <= 3:
                key = reflection_group(state.history)
                if key not in holdout and key not in starts:
                    starts[key] = state.history
                    taken += 1
            kind = "randomized_search" if index % 2 else "bfs"
            state = state.apply(bot_action(state, kind, 980_000 + index * 13 + state.turn))
    if len(starts) < 100:
        raise RuntimeError("Insufficient independent training-only near-goal prefixes")
    atomic_json(target, {"generator": "training-only-near-goal-v1", "opening_count": len(starts),
                         "arena_sha256": sha256(root / "arena.json"), "openings": list(starts.values())})
    print("Generated local near-goal training prefixes:", len(starts), flush=True)


class Experience:
    def __init__(self, openings, lanes, seed, snapshot=None, curriculum=(), pool=POOL, resolve=bot_action):
        self.openings, self.seed = openings, seed
        self.standard_openings = {tuple(opening) for opening in openings}
        self.curriculum = curriculum
        self.pool = pool
        self.pool_replacements = {}
        self.alternate_pool_replacements = {}
        self.standard_only_opponents = set()
        self.resolve = resolve
        self.resolve_batch = getattr(resolve, "batch", None)
        self.started = snapshot["started"] if snapshot else 0
        self.paired_opening = snapshot.get("paired_opening") if snapshot else None
        self.paired_opponent = snapshot.get("paired_opponent") if snapshot else None
        self.games = []
        self.block = -1
        for slot in snapshot["slots"] if snapshot else [None] * lanes:
            if slot:
                self.games.append(Game(State.from_moves(slot["moves"]), slot["player"], slot["opponent"],
                                       slot["seed"], tuple(slot["opening"]), slot["index"]))
            else:
                self.games.append(None)
        if len(self.games) != lanes:
            raise ValueError("Resume must use the original number of rollout lanes")

    def snapshot(self):
        return {"started": self.started, "paired_opening": self.paired_opening,
                "paired_opponent": self.paired_opponent, "slots": [
            {"player": game.player, "opponent": game.opponent, "seed": game.seed,
             "opening": game.opening, "moves": game.state.history, "index": game.index}
            if game else None for game in self.games]}

    def next_game(self, curriculum_fraction=0.):
        index = self.started
        pair, side = divmod(index, 2)
        block, position = divmod(pair, len(self.pool))
        if self.block != block:
            self.order = list(self.pool)
            random.Random(950_000 + self.seed + block).shuffle(self.order)
            self.block = block
        opponent = self.order[position]
        if side and self.paired_opponent is not None:
            opponent = self.paired_opponent
        elif not side:
            replacements = self.pool_replacements if block % 2 == 0 else self.alternate_pool_replacements
            opponent = replacements.get(opponent, opponent)
        opening = self.openings[pair % len(self.openings)]
        if side and self.paired_opening:
            opening = self.paired_opening
        elif not side and self.curriculum and curriculum_fraction and opponent not in self.standard_only_opponents:
            choice = random.Random(990_000 + self.seed + pair)
            if choice.random() < curriculum_fraction:
                opening = choice.choice(self.curriculum)
        self.paired_opening = opening if side == 0 else None
        self.paired_opponent = opponent if side == 0 else None
        self.started += 1
        return Game.start(opening, side + 1, opponent, 960_000 + self.seed + pair, index, self.resolve)

    def collect(self, model, hyper, device):
        data, rewards, values, ends = [], [], [], []
        outcomes = Counter(wins=0, losses=0, truncated=0)
        opponents = set(self.pool) | set(self.pool_replacements.values()) | set(self.alternate_pool_replacements.values())
        opponents.update(game.opponent for game in self.games if game)
        if self.paired_opponent is not None:
            opponents.add(self.paired_opponent)
        by_opponent = {opponent: Counter() for opponent in opponents}
        starts = {opponent: Counter() for opponent in opponents}
        lanes = len(self.games)
        pbrs_scale = hyper.get("pbrs_scale", 0.)
        distance_margin = hyper.get("pbrs_potential") == "distance_margin"
        repeat_temperature = hyper.get("repeat_temperature", 1.)
        temperature_rows = []
        mirror_rollouts = hyper.get("mirror_rollouts", False)
        gamma = hyper["gamma"]
        model.eval()
        with torch.no_grad():
            for _ in range(hyper["rollout_actions"] // lanes):
                for offset, game in enumerate(self.games):
                    if game is None:
                        game = self.games[offset] = self.next_game(hyper.get("curriculum_fraction", 0.))
                        starts[game.opponent]["red" if game.player == 1 else "blue"] += 1
                boards, masks, mirrored = rollout_observations(self.games, mirror_rollouts)
                device_mask = masks.to(device)
                logits, value = model(boards.to(device), device_mask)
                logits = balance_action_families(logits, device_mask, hyper.get("action_family_balance", 0.))
                if repeat_temperature > 1.:
                    temperatures = [repeat_temperature if repeats_pawn_moves(game.state.history) else 1.
                                    for game in self.games]
                    row = torch.tensor(temperatures, dtype=logits.dtype)
                    temperature_rows.append(row)
                    logits = logits / row.to(device).unsqueeze(1)
                policy = Categorical(logits=logits)
                if device == "mps":
                    # MPS multinomial can select a zero-weight entry in a concentrated policy.
                    # Sample the same masked probabilities on CPU and keep PPO math on MPS.
                    weights = policy.probs.cpu().masked_fill(~masks, 0.)
                    actions = torch.multinomial(weights, 1).squeeze(1)
                    device_actions = actions.to(device)
                else:
                    device_actions = policy.sample()
                    actions = device_actions.cpu()
                data.append((boards, masks, actions, policy.log_prob(device_actions).cpu()))
                values.append(value.cpu().numpy())
                tick_rewards, tick_ends = [], []
                if not masks.gather(1, actions.unsqueeze(1)).all().item():
                    raise RuntimeError("Masked PPO selected an illegal learner action")
                moves = actions
                if mirrored is not None:
                    moves = actions.clone()
                    moves[mirrored] = HORIZONTAL_ACTIONS[actions[mirrored]]
                ids = moves.tolist()
                previous_states = [game.state for game in self.games]
                results = (step_batch(self.games, ids, self.resolve_batch) if self.resolve_batch else
                           [game.step(action, self.resolve) for game, action in zip(self.games, ids)])
                for offset, (game, previous, result) in enumerate(zip(self.games, previous_states, results)):
                    by_opponent[game.opponent]["actions"] += 1
                    if repeat_temperature > 1. and temperatures[offset] > 1.:
                        by_opponent[game.opponent]["repeat_exploration_actions"] += 1
                    if mirrored is not None and bool(mirrored[offset]):
                        by_opponent[game.opponent]["mirrored_actions"] += 1
                    reward = REWARD[result]
                    if pbrs_scale:
                        # One learner decision includes the opponent reply. A capped game
                        # is a finite-horizon ending here, so both end types have zero potential.
                        before = bfs_goal_potential(previous, game.player, pbrs_scale, distance_margin=distance_margin)
                        after = (bfs_goal_potential(game.state, game.player, pbrs_scale, distance_margin=distance_margin)
                                 if result is None else 0.)
                        shaping = gamma * after - before
                        reward += shaping
                        by_opponent[game.opponent]["pbrs_reward"] += shaping
                        by_opponent[game.opponent]["pbrs_nonzero"] += int(shaping != 0.)
                    tick_rewards.append(reward)
                    tick_ends.append(result is not None)
                    if result:
                        outcomes[result] += 1
                        by_opponent[game.opponent][result] += 1
                        by_opponent[game.opponent][f"{result}_{'red' if game.player == 1 else 'blue'}"] += 1
                        opening_group = "standard" if tuple(game.opening) in self.standard_openings else "curriculum"
                        by_opponent[game.opponent][f"{result}_{opening_group}"] += 1
                        self.games[offset] = None
                rewards.append(tick_rewards)
                ends.append(tick_ends)
            active = [index for index, game in enumerate(self.games) if game]
            bootstrap = np.zeros(lanes, dtype=np.float32)
            if active:
                boards, masks, _ = rollout_observations([self.games[index] for index in active], mirror_rollouts)
                bootstrap[active] = model(boards.to(device), masks.to(device))[1].cpu().numpy()
        adv = gae(rewards, values, ends, bootstrap, gamma, hyper["gae_lambda"])
        targets = adv + np.asarray(values)
        batch = (*[torch.cat(col) for col in zip(*data)], torch.from_numpy(adv.reshape(-1)),
                 torch.from_numpy(targets.astype(np.float32).reshape(-1)))
        if temperature_rows:
            batch = (*batch, torch.cat(temperature_rows))
        return batch, dict(outcomes), {key: dict(value) for key, value in by_opponent.items()}, starts, len(active)


def gae(rewards, values, ends, bootstrap, gamma, weight):
    result = np.empty_like(np.asarray(values), dtype=np.float32)
    future = np.zeros_like(bootstrap)
    next_value = bootstrap
    for t in range(len(values) - 1, -1, -1):
        live = ~np.asarray(ends[t])
        delta = np.asarray(rewards[t]) + gamma * live * next_value - values[t]
        future = delta + gamma * weight * live * future
        result[t] = future
        next_value = values[t]
    return result


def update_policy(model, optimizer, data, hyper, device, *, reference=None):
    boards, masks, actions, old_probs, advantage, returns, *temperature = [item.to(device) for item in data]
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    if reference is not None:
        with torch.no_grad():
            reference_logits, _ = reference(boards, masks)
            reference_logits = balance_action_families(reference_logits, masks, hyper.get("action_family_balance", 0.))
            if temperature:
                reference_logits = reference_logits / temperature[0].unsqueeze(1)
            reference_log_prob = reference_logits.log_softmax(dim=-1)
            reference_prob = reference_log_prob.exp()
            reference_log_prob = reference_log_prob.masked_fill(~masks, 0.)
    totals = Counter()
    model.train()
    for _ in range(hyper["update_epochs"]):
        for batch in torch.randperm(len(actions), device=device).split(hyper["minibatch"]):
            logits, values = model(boards[batch], masks[batch])
            logits = balance_action_families(logits, masks[batch], hyper.get("action_family_balance", 0.))
            if temperature:
                logits = logits / temperature[0][batch].unsqueeze(1)
            policy = Categorical(logits=logits)
            log_ratio = policy.log_prob(actions[batch]) - old_probs[batch]
            ratio = log_ratio.exp()
            clip = hyper["clip"]
            gain = torch.minimum(advantage[batch] * ratio, advantage[batch] * ratio.clamp(1 - clip, 1 + clip))
            policy_loss = -gain.mean()
            value_loss = (values - returns[batch]).square().mean()
            entropy = policy.entropy().mean()
            total_loss = policy_loss + hyper["value_coef"] * value_loss - hyper["entropy_coef"] * entropy
            if reference is not None:
                current_log_prob = policy.logits.masked_fill(~masks[batch], 0.)
                reference_kl = (reference_prob[batch] * (reference_log_prob[batch] - current_log_prob)).sum(-1).mean()
                total_loss = total_loss + hyper["reference_kl_coef"] * reference_kl
            if not torch.isfinite(total_loss).item():
                raise RuntimeError("Non-finite Masked PPO objective")
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), hyper["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            values_to_log = {"loss": total_loss, "policy_loss": policy_loss, "value_loss": value_loss,
                             "entropy": entropy, "grad_norm": grad,
                             "approx_kl": (ratio - 1 - log_ratio).mean(),
                             "clip_fraction": ((ratio - 1).abs() > clip).float().mean()}
            if reference is not None:
                values_to_log["reference_kl"] = reference_kl
            for key, value in values_to_log.items():
                totals[key] += value.item() * len(batch)
    return {key: value / (len(actions) * hyper["update_epochs"]) for key, value in totals.items()}


def evaluate(model, arena, device, family_balance, opponents=ANCHORS, resolve=bot_action):
    scores, games = {}, []
    for index, opponent in enumerate(opponents):
        trials = evaluation_games([tuple(opening) for opening in arena["eval_openings"]],
                                  opponent, arena["eval_seed"] + 1000 * index, resolve)
        records = play_evaluation(trials, device, model=model, family_balance=family_balance, resolve=resolve)
        scores[opponent] = score_games(records)
        games.extend(records)
    return scores, games


def ranking(scores, elo):
    return (passed(scores), min(scores[name]["win_rate"] / threshold for name, threshold in THRESHOLDS.items()),
            -max(row["truncation_rate"] for row in scores.values()), elo)


def tuning(history, event, hyper):
    """Change at most one parameter after the requested 100-update baseline."""
    if event["update"] < SCHEDULE["initial"] or event["gate_passed"]:
        return None
    evaluations = [row for row in history if "formal_eval" in row]
    recent = [*history[-24:], event]
    recent_kl = np.mean([row["approx_kl"] for row in recent])
    rollout_wins = sum(row["rollout_results"]["wins"] for row in recent)
    rollout_finishes = sum(sum(row["rollout_results"].values()) for row in recent)
    curriculum = hyper.get("curriculum_fraction", 0.)
    family_balance = hyper.get("action_family_balance", 0.)
    wall_rates = [row["wall_action_rate"] for row in recent if "wall_action_rate" in row]
    wall_rate = sum(wall_rates) / len(wall_rates) if wall_rates else 0.
    scores = event["formal_eval"]
    progress = ranking(scores, event["elo"])[1]
    prior = ranking(evaluations[-1]["formal_eval"], evaluations[-1]["elo"])[1] if evaluations else 0.
    if curriculum > 0 and progress < .15 and wall_rate > .35 and family_balance < 1.:
        param, value, why = "action_family_balance", min(1., family_balance + .5), "近期墙动作超过 35%，且正式评估仍停滞。校正棋子与墙的可用动作数量差异"
    elif rollout_finishes >= 200 and rollout_wins / rollout_finishes < .005 and curriculum < .65:
        param, value, why = "curriculum_fraction", .45 if curriculum == 0 else .65, "过去 25 轮的训练胜率低于 0.5%，增加本地生成的近终局开局以提供终局信号"
    elif curriculum > .20 and progress - prior >= .15 and rollout_wins / max(1, rollout_finishes) >= .10:
        param, value, why = "curriculum_fraction", max(.20, curriculum - .15), "训练胜率与正式评估同步提升，增加普通开局的比例"
    elif recent_kl > .045 and hyper["lr"] > 8e-5:
        param, value, why = "lr", max(8e-5, .7 * hyper["lr"]), "Mean KL above 0.045 in the latest 25 updates"
    elif (progress - prior < .03 or max(row["truncation_rate"] for row in scores.values()) >= .05) and hyper["entropy_coef"] > .001:
        param, value, why = "entropy_coef", max(.001, .5 * hyper["entropy_coef"]), "Gate progress plateaued or truncation reached 5%"
    elif progress - prior < .03 and recent_kl < .007 and hyper["lr"] < 4.5e-4:
        param, value, why = "lr", min(4.5e-4, 1.2 * hyper["lr"]), "Plateau with mean KL below 0.007"
    else:
        return None
    return {"parameter": param, "before": hyper.get(param, 0.), "after": value, "reason": why,
            "from_update": event["update"] + 1}


def archive(source, target):
    temporary = target.with_suffix(".link")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    temporary.replace(target)


def write_games(path, games):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for game in games:
            stream.write(json.dumps(game) + "\n")
    temporary.replace(path)


def save_report(folder, config, state, history, finished):
    rows = [event for event in history if "formal_eval" in event]
    limit = state.get("limit", SCHEDULE["limit"])
    result = "已达标" if state["qualified"] else f"{limit} 轮内未达标" if finished else "训练中"
    comparison = config.get("comparison")
    inherited = (f"继承：Level 1 第 {comparison['source_update']} 次更新的权重和 Adam。"
                 if comparison else
                 f"继承：本地搜索试跑第 {config['inheritance']['pilot_updates']} 轮的模型和 Adam。")
    header = ["# Quoridor Phase 1 — Level 1", "",
              f"模型：{config['architecture'].upper()} · seed {config['seed']} · {config['device']}",
              f"正式训练：{state['update']} 轮 / {state['steps']:,} 有效动作。",
              f"{inherited}外部历史棋谱使用量为 0。",
              f"状态：{result}。"]
    if comparison:
        label = "Random 专项" if comparison["arm"] == "random" else "原对手池对照"
        pool = " · ".join(f"{name} {ratio:.0%}" for name, ratio in config["opponent_pool"].items())
        baseline = comparison["baseline"]
        header.extend([f"组别：{label}。对手配比：{pool}。",
                       f"起点对 Random 胜率 {baseline['random']['win_rate']:.1%}，"
                       f"截断率 {baseline['random']['truncation_rate']:.1%}。"])
        if game_units := state.get("game_units"):
            header.extend([f"从第 {game_units['started_at_update']} 次更新后按每 {game_units['size']} 盘训练对局评估，首次达标即停止。",
                           f"本次续训已结算 {game_units['completed']} 盘训练对局。"])
        else:
            header.append(f"当前累计预算为 {limit} 次 PPO 更新，训练参数保持为起点配置。")
    header.extend(["Elo 为冻结本机 Arena 内部估计，BFS=1500。胜率分母为每对手 200 局，截断与非法单列。",
                   "", "| 轮次 | Elo | Random 胜率 | Random 截断率 | BFS 胜率 | Level 1 胜率 | 最高截断率 | 非法 | 连续达标 | 动作/秒 | KL | Entropy | Loss |",
                   "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for event in rows:
        scores = event["formal_eval"]
        rates = " | ".join(f"{rate:.1%}" for rate in (
            scores["random"]["win_rate"], scores["random"]["truncation_rate"],
            scores["bfs"]["win_rate"], scores["level1"]["win_rate"]))
        header.append(f"| {event['update']} | {event['elo']:.0f} | {rates} | "
                      f"{max(row['truncation_rate'] for row in scores.values()):.1%} | "
                      f"{sum(row['illegal'] for row in scores.values())} | {event['qualifying_streak']} | "
                      f"{event['actions_per_second']:.0f} | {event['approx_kl']:.4f} | "
                      f"{event['entropy']:.4f} | {event['loss']:.4f} |")
    if state["best_update"]:
        header.extend(["", f"最佳检查点：best.pt（第 {state['best_update']} 轮）。"])
    if state.get("until_first_pass"):
        header.extend(["", f"续跑规则：每 {UNIT_UPDATES} 次 PPO 更新为一个单元，任意一次正式评估达标即结束。"])
    if state.get("qualified_update"):
        header.append(f"达标检查点：qualified.pt（第 {state['qualified_update']} 轮）。")
    if state.get("device_history"):
        header.extend(["", "## 设备切换", ""])
        header.extend(f"- 第 {entry['after_update']} 轮后从 {entry['from'].upper()} 切换到 {entry['to'].upper()}。"
                      for entry in state["device_history"])
    if state.get("pbrs_history"):
        header.extend(["", "## BFS 势函数奖励", ""])
        header.extend(f"- 第 {entry['after_update'] + 1} 轮起，BFS 距离系数由 {entry['from']:g} 调为 {entry['to']:g}。"
                      for entry in state["pbrs_history"])
    changes = [event["tuning"] for event in rows if event.get("tuning")]
    if changes:
        header.extend(["", "## 调参记录", ""])
        header.extend(f"- 第 {row['from_update']} 轮起，{row['parameter']} 从 {row['before']:.5g} 调整为 {row['after']:.5g}。{row['reason']}。"
                      for row in changes)
    tmp = folder / "REPORT.tmp"
    tmp.write_text("\n".join(header) + "\n", encoding="utf-8")
    tmp.replace(folder / "REPORT.md")


def train(args):
    torch.set_num_threads(args.threads)
    architecture = args.architecture
    pilot, inheritance = pilot_checkpoint(args.pilot, architecture)
    arena_path = args.root / "arena.json"
    arena = json.loads(arena_path.read_text())
    if arena["protocol"] != PROTOCOL or len(arena["eval_openings"]) * 2 != SCHEDULE["eval_games"]:
        raise ValueError("Arena does not match the Level 1 protocol")
    curriculum_path = args.root / "curriculum.json"
    curriculum = json.loads(curriculum_path.read_text())
    if curriculum["arena_sha256"] != sha256(arena_path):
        raise ValueError("Curriculum opening bank does not match the frozen Arena")
    curriculum_sha = sha256(curriculum_path)
    config = {"protocol": PROTOCOL, "architecture": architecture, "seed": args.seed, "device": args.device,
              "lanes": args.lanes, "threads": args.threads, "arena_sha256": sha256(arena_path),
              "demonstrations_sha256": inheritance["demonstrations_sha256"], "inheritance": inheritance,
              "schedule": SCHEDULE, "hyperparameters": HYPER, "opponent_pool": {"level1": .7, "bfs": .2, "randomized_search": .1},
              "max_plies": MAX_PLIES, "truncation": "finite-horizon zero reward", "start_level": "level1_pool",
              "torch_version": str(torch.__version__)}
    comparison = None
    origin = None
    if args.comparison_source:
        origin = torch.load(args.comparison_source, map_location="cpu", weights_only=True)
        original = origin["state"]
        if origin["config"] != config or original.get("curriculum_sha256") != curriculum_sha:
            raise ValueError("Comparison must inherit a matching Level 1 checkpoint, Arena, curriculum and device")
        evaluation = (original.get("event") or {}).get("formal_eval")
        if not evaluation or original["hyper"].get("pbrs_scale") != .02:
            raise ValueError("Comparison requires a formally evaluated checkpoint with PBRS 0.02")
        comparison = {"arm": args.comparison_arm, "source_update": original["update"],
                      "source_checkpoint": str(args.comparison_source.resolve()),
                      "source_sha256": sha256(args.comparison_source), "baseline": evaluation}
    folder = args.root / f"{architecture}-seed{args.seed}"
    if not args.resume:
        folder.mkdir(parents=True, exist_ok=False)
    latest, metrics = folder / "latest.pt", folder / "metrics.jsonl"
    with phase1_progress(folder, 0, SCHEDULE["initial"]) as progress:
        torch.manual_seed(args.seed)
        model = ActorCritic(architecture).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=HYPER["lr"])
        previous_device = None
        if args.resume:
            # Adam uses CPU step counters for the non-capturable optimizer.
            # Loading on CPU lets load_state_dict move each state tensor correctly.
            saved = torch.load(latest, map_location="cpu", weights_only=True)
            saved_config = saved["config"]
            comparison = saved_config.get("comparison")
        if comparison:
            if args.until_first_pass or args.migrate_device or args.pbrs_scale is not None:
                raise ValueError("Comparison requires an explicit update budget and retains its device and PPO settings")
            pool = RANDOM_TRAINING_POOL if comparison["arm"] == "random" else POOL
            config.update(comparison=comparison, opponent_pool={
                name: count / len(pool) for name, count in Counter(pool).items()
            })
        else:
            pool = POOL
        if args.resume:
            if saved_config != config and args.migrate_device and {**saved_config, "device": args.device} == config:
                previous_device = saved_config["device"]
            elif saved_config != config:
                raise ValueError("Resume must use the original Level 1 configuration and source")
            state = saved["state"]
        else:
            atomic_json(folder / "config.json", config)
            saved = origin if comparison else pilot
            state = {"update": 0, "steps": 0, "target": SCHEDULE["initial"], "streak": 0, "qualified": False,
                     "best_rank": None, "best_update": 0, "hyper": dict(HYPER), "rollout": None, "event": None}
            if comparison:
                state.update(target=UNIT_UPDATES, limit=UNIT_UPDATES, hyper=dict(origin["state"]["hyper"]))
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if state["event"] and "formal_eval" in state["event"]:
            milestone = folder / f"checkpoint-{state['update']:04d}.pt"
            if not milestone.is_file():
                archive(latest, milestone)
            if state["best_update"] == state["update"]:
                archive(latest, folder / "best.pt")
        history = [row for row in read_rows(metrics) if row.get("phase") == "ppo"]
        if state["event"] and (not history or history[-1]["update"] < state["update"]):
            record(metrics, state["event"])
            history.append(state["event"])

        if state.get("curriculum_sha256", curriculum_sha) != curriculum_sha:
            raise ValueError("Resume requires the original curriculum opening bank")
        state["curriculum_sha256"] = curriculum_sha
        experience = Experience(arena["training_openings"], args.lanes, args.seed, state["rollout"], curriculum["openings"], pool)
        game_units = state.get("game_units")
        game_units_started = args.until_first_pass_games is not None and game_units is None
        if game_units_started:
            if not comparison or comparison["arm"] != "random":
                raise ValueError("Game units require a resumed Random training run")
            game_units = {"size": args.until_first_pass_games, "started_at_update": state["update"],
                          "completed": 0, "next_evaluation": args.until_first_pass_games}
            state["game_units"] = game_units
        if game_units:
            if args.extend_to is not None or args.until_first_pass:
                raise ValueError("This run already has its continuation budget in training games")
            if args.until_first_pass_games is not None and args.until_first_pass_games != game_units["size"]:
                raise ValueError("Resume must keep the saved training game unit size")
        auto_units = bool(args.until_first_pass or state.get("until_first_pass"))
        required_passes = 1 if auto_units or comparison else 3
        first_pass = next((row for row in history if row.get("gate_passed")), None) if auto_units else None
        if first_pass:
            state.update(qualified=True, qualified_update=first_pass["update"], streak=max(1, state["streak"]))
        limit = state.get("limit", SCHEDULE["limit"])
        if args.extend_to is not None and args.extend_to < limit:
            raise ValueError("--extend-to cannot reduce the saved run limit")
        requested_limit = next_unit_target(state["update"]) if auto_units and not state["qualified"] else args.extend_to
        extended = requested_limit is not None and requested_limit > limit
        if extended:
            limit = requested_limit
            state.update(limit=limit, target=limit)
        if extended or game_units_started:
            for name in ("summary.json", "REPORT.md"):
                previous = folder / name
                archived = previous.with_stem(f"{previous.stem}-{state['update']:04d}")
                if previous.is_file() and not archived.exists():
                    archive(previous, archived)
        if auto_units and not state["qualified"]:
            limit = requested_limit
            state.update(limit=limit, target=limit)
        state["until_first_pass"] = auto_units
        previous_scale = state["hyper"].get("pbrs_scale", 0.)
        pbrs_changed = args.pbrs_scale is not None and args.pbrs_scale != previous_scale
        if pbrs_changed:
            state["hyper"]["pbrs_scale"] = args.pbrs_scale
            state.setdefault("pbrs_history", []).append({"after_update": state["update"],
                                                        "from": previous_scale, "to": args.pbrs_scale})
        if previous_device is not None:
            state.setdefault("device_history", []).append({"from": previous_device, "to": args.device,
                                                            "after_update": state["update"]})
        if not args.resume or extended or game_units_started or args.until_first_pass or previous_device is not None or pbrs_changed:
            save(latest, model=model.state_dict(), optimizer=optimizer.state_dict(), config=config, state=state)
        if args.resume and read_json(folder / "config.json") != config:
            atomic_json(folder / "config.json", config)
        progress.update(ppo_total=None if game_units else state["target"], required_passes=required_passes,
                        qualified=state["qualified"], pbrs_scale=state["hyper"].get("pbrs_scale", 0.))
        if game_units:
            progress["game_units"] = dict(game_units)
        atomic_json(folder / PROGRESS, {**progress, "status": "training", "activity": "collecting"})
        if extended or pbrs_changed or game_units_started or (comparison and not args.resume):
            save_report(folder, config, state, history, False)
        while (game_units or state["update"] < state["target"]) and (
            not state["qualified"] or (comparison and not game_units)
        ):
            number = state["update"] + 1
            hyper = state["hyper"]
            for group in optimizer.param_groups:
                group["lr"] = hyper["lr"]
            source_update = comparison["source_update"] if comparison else 0
            torch.manual_seed(args.seed + 1_000_000 + source_update + number)
            started = time.perf_counter()
            batch, scores, by_opponent, starts, partial = experience.collect(model, hyper, args.device)
            sampled = time.perf_counter()
            diagnostics = update_policy(model, optimizer, batch, hyper, args.device)
            ended = time.perf_counter()
            state["update"] = number
            state["steps"] += hyper["rollout_actions"]
            event = {"phase": "ppo", "update": number, "steps": state["steps"], "opponent": "level1_pool", "device": args.device,
                     "rollout_results": scores, "by_opponent": by_opponent, "game_starts": starts,
                     "learner_illegal_actions": 0, "active_games_bootstrapped": partial,
                     "seconds": round(ended - started, 4), "rollout_seconds": round(sampled - started, 4),
                     "optimizer_seconds": round(ended - sampled, 4),
                     "wall_action_rate": batch[2].ge(81).float().mean().item(),
                     "actions_per_second": round(hyper["rollout_actions"] / (ended - started), 2),
                     "hyperparameters": dict(hyper), **diagnostics}
            selected = False
            if game_units:
                game_units["completed"] += sum(scores.values())
                evaluation_due = game_units["completed"] >= game_units["next_evaluation"]
            else:
                evaluation_due = number % SCHEDULE["evaluate_every"] == 0
            if evaluation_due:
                if game_units:
                    game_units["next_evaluation"] = (game_units["completed"] // game_units["size"] + 1) * game_units["size"]
                atomic_json(folder / PROGRESS, {**progress, "status": "training", "activity": "evaluating", "eval_update": number})
                eval_start = time.perf_counter()
                scores, games = evaluate(model, arena, args.device, hyper.get("action_family_balance", 0.))
                elo = model_elo(scores, arena["anchor_ratings"])
                gate = passed(scores)
                state["streak"] = state["streak"] + 1 if gate else 0
                if state["streak"] >= required_passes and (comparison or auto_units or number >= SCHEDULE["initial"]):
                    state["qualified"] = True
                    state.setdefault("qualified_update", number)
                elif not comparison:
                    state["qualified"] = False
                event.update(formal_eval=scores, development_eval=scores["level1"], shortest_eval=scores["bfs"],
                             gate_passed=gate, qualifying_streak=state["streak"], qualified=state["qualified"],
                             required_passes=required_passes,
                             elo=elo, eval_seconds=round(time.perf_counter() - eval_start, 3))
                rank = ranking(scores, elo)
                if state["best_rank"] is None or rank > tuple(state["best_rank"]):
                    state["best_rank"], state["best_update"] = rank, number
                    selected = True
                if auto_units and number >= state["target"]:
                    event["completed_unit"] = number // UNIT_UPDATES
                    if not state["qualified"]:
                        limit = next_unit_target(number)
                        state.update(limit=limit, target=limit)
                        event["next_ppo_target"] = limit
                elif not state["qualified"] and number >= state["target"] and number < limit:
                    state["target"] = min(limit, number + SCHEDULE["evaluate_every"])
                change = tuning(history, event, hyper) if not comparison and not state["qualified"] and number < limit else None
                if change:
                    event["tuning"] = change
                    state["hyper"] = {**hyper, change["parameter"]: change["after"]}
                write_games(folder / f"ppo_{number:04d}_games.jsonl", games)
            if game_units:
                event["game_units"] = dict(game_units)
                progress["game_units"] = dict(game_units)
            state["rollout"], state["event"] = experience.snapshot(), event
            save(latest, model=model.state_dict(), optimizer=optimizer.state_dict(), config=config, state=state)
            if "formal_eval" in event:
                archive(latest, folder / f"checkpoint-{number:04d}.pt")
                if selected:
                    archive(latest, folder / "best.pt")
            record(metrics, event)
            history.append(event)
            progress.update(ppo_total=None if game_units else state["target"], qualified=state["qualified"])
            if "formal_eval" in event:
                atomic_json(folder / PROGRESS, {**progress, "status": "training", "activity": "collecting"})
                save_report(folder, config, state, history, False)

        if state["qualified"]:
            update = state.get("qualified_update", state["update"])
            qualification = folder / f"checkpoint-{update:04d}.pt"
            archive(qualification, folder / "qualified.pt")
            progress.update(qualified=True, qualified_update=update)
        if game_units:
            progress["ppo_total"] = state["update"]
        save_report(folder, config, state, history, True)
        atomic_json(folder / "summary.json", {"protocol": PROTOCOL, "architecture": architecture,
                    "qualified": state["qualified"], "updates": state["update"], "limit": None if game_units else limit,
                    "steps": state["steps"], **({"game_units": game_units} if game_units else {}),
                    "streak": state["streak"], "required_passes": required_passes,
                    "pbrs_scale": state["hyper"].get("pbrs_scale", 0.),
                    "qualified_update": state.get("qualified_update"), "best_update": state["best_update"],
                    "best_checkpoint": "best.pt", "report": "REPORT.md", "finished_at": datetime.now(timezone.utc).isoformat()})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--architecture", choices=("mlp", "cnn32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--migrate-device", action="store_true", help="Allow only the training device to change on resume")
    parser.add_argument("--pbrs-scale", type=float,
                        help="BFS goal-distance coefficient, 0 disables, omitted preserves a resumed run's coefficient")
    parser.add_argument("--comparison-source", type=Path, help="Start a fixed-budget Random comparison from a Level 1 checkpoint")
    parser.add_argument("--comparison-arm", choices=("control", "random"), help="Select the training opponent mix")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--extend-to", type=int, help="Explicit cumulative update budget for a resumed run")
    continuation.add_argument("--until-first-pass", action="store_true",
                              help="Continue in 500-update units until the first formal evaluation passes")
    continuation.add_argument("--until-first-pass-games", type=int, metavar="GAMES",
                              help="Evaluate after each unit of completed training games until formally qualified")
    args = parser.parse_args()
    if bool(args.comparison_source) != bool(args.comparison_arm) or (args.comparison_source and
            (not args.architecture or args.resume or args.pbrs_scale is not None)):
        parser.error("A new comparison requires an architecture, source checkpoint and arm and inherits PPO settings")
    if args.threads < 1 or args.lanes < 1 or HYPER["rollout_actions"] % args.lanes:
        parser.error("Positive threads and a lane count that divides 2048 are required")
    if args.extend_to is not None and (not args.resume or not args.architecture or args.extend_to < SCHEDULE["limit"]
                                       or args.extend_to % SCHEDULE["evaluate_every"]):
        parser.error("--extend-to requires an architecture and --resume, and must end on an evaluation boundary")
    if args.until_first_pass and (not args.resume or not args.architecture):
        parser.error("--until-first-pass requires an architecture and --resume")
    if args.until_first_pass_games is not None and (not args.resume or not args.architecture or args.until_first_pass_games < 1):
        parser.error("--until-first-pass-games requires an architecture, --resume and a positive game count")
    if args.migrate_device and (not args.resume or not args.architecture):
        parser.error("--migrate-device requires an architecture and --resume")
    if args.pbrs_scale is not None and (not args.architecture or not math.isfinite(args.pbrs_scale) or args.pbrs_scale < 0.):
        parser.error("--pbrs-scale requires an architecture and a finite nonnegative coefficient")
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if args.architecture:
        train(args)
    else:
        prepare(args.root, args.pilot)


if __name__ == "__main__":
    main()
