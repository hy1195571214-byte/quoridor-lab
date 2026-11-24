"""Warm-start Quoridor policies from local search and train them with masked PPO."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .dataset_import import file_sha256, reflection_group
from .env import QuoridorEnv
from .fast_observation import training_observation
from .learning import network
from .opponents import LocalPolicy
from .phase1_monitor import phase1_progress
from .rules import Action, State
from .search import SearchAgent
from .search_demonstrations import SEARCH_LEVELS, opening_pool


class ActorCritic(nn.Module):
    def __init__(self, architecture: str):
        super().__init__()
        policy = network(architecture)
        self.encoder = policy[:-1]
        self.actor = policy[-1]
        self.critic = nn.Linear(self.actor.in_features, 1)

    def forward(self, boards, masks):
        features = self.encoder(boards)
        logits = self.actor(features).masked_fill(~masks, -torch.inf)
        return logits, self.critic(features).squeeze(-1)


def load_demonstrations(path: Path, max_plies: int) -> tuple[dict, set[str]]:
    splits = {"train": [], "validation": []}
    groups = {}
    openings = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        game = json.loads(line)
        split = game["split"]
        opening = tuple(game["opening"])
        moves = tuple(game["moves"])
        if moves[:len(opening)] != opening or not len(opening) < len(moves) <= max_plies:
            raise ValueError("Demonstration length or opening is inconsistent")
        openings.add(reflection_group(opening))
        group = reflection_group(moves)
        if group in groups:
            raise ValueError("Duplicate or mirrored demonstration game")
        groups[group] = split
        state = State.from_moves(opening)
        for notation in moves[len(opening):]:
            obs = training_observation(state, max_plies)
            next_state = state.apply(notation)
            action = Action.parse(notation).id
            if not obs["action_mask"][action]:
                raise RuntimeError("Legal mask excluded a teacher action")
            splits[split].append((obs["board"], obs["action_mask"], action))
            state = next_state
        if state.winner != game["winner"] or (state.winner is None) != game["truncated"]:
            raise ValueError("Demonstration result differs from rules replay")
        if state.winner is None and len(moves) != max_plies:
            raise ValueError("Unfinished demonstration ended before the configured ply limit")
    return {key: make_batch(samples) for key, samples in splits.items()}, openings


def make_batch(samples):
    boards, masks, actions = zip(*samples, strict=True)
    return (torch.from_numpy(np.stack(boards)),
            torch.from_numpy(np.stack(masks)).bool(),
            torch.tensor(actions, dtype=torch.long))


def behavior_clone(model, splits, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(args.bc_epochs):
        model.train()
        boards, masks, actions = splits["train"]
        train_loss = 0.0
        for batch in torch.randperm(len(actions)).split(args.minibatch):
            x, legal, target = (data[batch].to(device) for data in (boards, masks, actions))
            logits, _ = model(x, legal)
            loss = nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch)
        model.eval()
        val_boards, val_masks, val_actions = splits["validation"]
        correct, val_loss = 0, 0.0
        with torch.no_grad():
            for batch in torch.arange(len(val_actions)).split(args.minibatch):
                valid = val_actions[batch].to(device)
                logits, _ = model(val_boards[batch].to(device), val_masks[batch].to(device))
                correct += (logits.argmax(-1) == valid).sum().item()
                val_loss += nn.functional.cross_entropy(logits, valid, reduction="sum").item()
        yield {"epoch": epoch + 1, "train_loss": train_loss / len(actions),
               "val_loss": val_loss / len(val_actions), "val_accuracy": correct / len(val_actions),
               "train_positions": len(actions), "val_positions": len(val_actions)}


def start_game(level, seed, player, opening, max_plies):
    opponent = (SearchAgent(config=SEARCH_LEVELS[level], seed=seed)
                if level in SEARCH_LEVELS else LocalPolicy(level, seed))
    env = QuoridorEnv(opponent, learner_player=player, max_plies=max_plies)
    state = State.from_moves(opening)
    if state.turn != player:
        state = state.apply(opponent.choose(state).action)
    env.state = state
    obs = training_observation(state, max_plies)
    return env, obs


def predict(model, obs, device):
    board = torch.from_numpy(obs["board"]).unsqueeze(0).to(device)
    legal = torch.from_numpy(obs["action_mask"]).unsqueeze(0).to(device).bool()
    return model(board, legal)


def evaluate(model, openings, level, max_plies, seed, device, *, game_records=None):
    scores = Counter(wins=0, losses=0, truncated=0, wins_red=0, wins_blue=0)
    model.eval()
    with torch.no_grad():
        for index, opening in enumerate(openings):
            for player in (1, 2):
                env, obs = start_game(level, seed + index, player, opening, max_plies)
                try:
                    while True:
                        logits, _ = predict(model, obs, device)
                        obs, reward, terminated, truncated, _ = env.step(logits.argmax(-1).item())
                        if terminated or truncated:
                            break
                    label = "truncated" if truncated else "wins" if reward > 0 else "losses"
                    scores[label] += 1
                    if reward > 0:
                        scores["wins_red" if player == 1 else "wins_blue"] += 1
                    if game_records is not None:
                        game_records.append({"opponent": level, "seed": seed + index,
                                             "learner_player": player, "result": label,
                                             "opening": opening, "moves": env.state.history})
                finally:
                    env.close()
    return dict(scores)


def advantages(rewards, values, boundaries, last_value, gamma=0.99, gae_lambda=0.95):
    """GAE from the learner's view. Capped games are finite, reward-zero endings."""
    result = np.zeros(len(rewards), dtype=np.float32)
    future = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        live = 0.0 if boundaries[t] else 1.0
        delta = rewards[t] + gamma * live * next_value - values[t]
        future = delta + gamma * gae_lambda * live * future
        result[t] = future
        next_value = values[t]
    return result


def rollout(model, openings, level, args, update, device):
    samples, log_probs, rewards, values, boundaries = [], [], [], [], []
    outcomes = Counter(wins=0, losses=0, truncated=0)
    start_seed = args.seed + 100_000 + update * 10_000
    env = None
    game_number = 0
    try:
        with torch.no_grad():
            for _ in range(args.rollout):
                if env is None:
                    opening = openings[(update + game_number // 2) % len(openings)]
                    player = 1 + ((update + game_number) % 2)
                    env, obs = start_game(level, start_seed + game_number // 2,
                                          player, opening, args.max_plies)
                    game_number += 1
                logits, value = predict(model, obs, device)
                policy = Categorical(logits=logits)
                action = policy.sample()
                move = action.item()
                if not obs["action_mask"][move]:
                    raise RuntimeError("PPO sampled an illegal move")
                samples.append((obs["board"], obs["action_mask"], move))
                log_probs.append(policy.log_prob(action).item())
                values.append(value.item())
                obs, reward, terminated, truncated, _ = env.step(move)
                rewards.append(reward)
                boundaries.append(terminated or truncated)
                if terminated or truncated:
                    outcomes["truncated" if truncated else "wins" if reward > 0 else "losses"] += 1
                    env.close()
                    env = None
            last_value = predict(model, obs, device)[1].item() if env is not None else 0.0
    finally:
        if env is not None:
            env.close()
    adv = advantages(rewards, values, boundaries, last_value)
    returns = adv + np.asarray(values, dtype=np.float32)
    return (*make_batch(samples), torch.tensor(log_probs), torch.from_numpy(adv),
            torch.from_numpy(returns)), dict(outcomes), int(not boundaries[-1])


def ppo_update(model, optimizer, rollout_data, args, device):
    boards, masks, actions, old_probs, adv, returns = rollout_data
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
    totals = Counter()
    model.train()
    for _ in range(args.update_epochs):
        for batch in torch.randperm(len(actions)).split(args.minibatch):
            x, legal, moves, old, advantage, target = (data[batch].to(device) for data in
                                                       (boards, masks, actions, old_probs, adv, returns))
            logits, value = model(x, legal)
            policy = Categorical(logits=logits)
            log_ratio = policy.log_prob(moves) - old
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(advantage * ratio, advantage * ratio.clamp(0.8, 1.2)).mean()
            value_loss = nn.functional.mse_loss(value, target)
            entropy = policy.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite PPO objective")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                metrics = {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy,
                           "grad_norm": grad_norm, "approx_kl": (ratio - 1 - log_ratio).mean(),
                           "clip_fraction": ((ratio - 1).abs() > 0.2).float().mean()}
                for name, value in metrics.items():
                    totals[name] += value.item() * len(batch)
    return {name: total / (args.update_epochs * len(actions)) for name, total in totals.items()}


def record(path, item):
    entry = json.dumps(item, allow_nan=False)
    with path.open("a", encoding="utf-8") as out:
        out.write(entry + "\n")
    print(entry, flush=True)


def save(path, **checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    splits, demonstration_openings = load_demonstrations(args.demonstrations, args.max_plies)
    config = dict(vars(args))
    config["demonstrations"] = str(args.demonstrations.resolve())
    config["output"] = str(args.output.resolve())
    config["demonstrations_sha256"] = file_sha256(args.demonstrations)
    config["device"] = device
    config["torch_version"] = str(torch.__version__)
    config["opponents"] = {level: asdict(cfg) for level, cfg in SEARCH_LEVELS.items()}
    del config["resume"], config["updates"]
    model = ActorCritic(args.architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    metrics = args.output / "metrics.jsonl"
    latest = args.output / "latest.pt"
    evaluation_openings = opening_pool(500_000, args.eval_pairs, 6, demonstration_openings)
    reserved = demonstration_openings | {reflection_group(moves) for moves in evaluation_openings}
    training_openings = opening_pool(200_000 + args.seed, 80, 6, reserved)

    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    with phase1_progress(args.output, args.bc_epochs, args.updates):
        update, level, streak, steps = 0, args.start_level, 0, 0
        if args.resume:
            checkpoint = torch.load(latest, map_location=device, weights_only=True)
            if checkpoint["config"] != config:
                raise ValueError("Resume configuration or demonstration file changed")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            update, level, streak, steps = (checkpoint[key] for key in ("update", "level", "streak", "steps"))
        else:
            (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            for metric in behavior_clone(model, splits, args, device):
                record(metrics, {"phase": "imitation", **metric})
            save(args.output / "warm_start.pt", model=model.state_dict(), config=config)
            for opponent in (level, "shortest"):
                games = []
                summary = evaluate(model, evaluation_openings, opponent, args.max_plies,
                                   700_000, device, game_records=games)
                write_games(args.output / "warm_start_games.jsonl", games)
                record(metrics, {"phase": "warm_start_eval", "opponent": opponent, **summary})
            save(latest, model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                 update=update, level=level, streak=streak, steps=steps)

        while update < args.updates:
            update += 1
            torch.manual_seed(args.seed + 100_000 + update)
            start = time.monotonic()
            data, outcomes, partial = rollout(model, training_openings, level, args, update, device)
            diagnostics = ppo_update(model, optimizer, data, args, device)
            steps += args.rollout
            event = {"phase": "ppo", "update": update, "steps": steps, "opponent": level,
                     "seconds": round(time.monotonic() - start, 2), "rollout_results": outcomes,
                     "rollout_end_bootstrapped": bool(partial), **diagnostics}
            if update % args.eval_every == 0:
                games = []
                summary = evaluate(model, evaluation_openings, level, args.max_plies,
                                   700_000, device, game_records=games)
                event["development_eval"] = summary
                if update == args.updates:
                    event["shortest_eval"] = evaluate(model, evaluation_openings, "shortest",
                                                      args.max_plies, 700_000, device, game_records=games)
                write_games(args.output / f"ppo_{update:04d}_games.jsonl", games)
                qualifying_wins = math.ceil(0.70 * 2 * args.eval_pairs)
                streak = streak + 1 if summary["wins"] >= qualifying_wins else 0
                if streak >= 3 and level != "hard":
                    level = "medium" if level == "easy" else "hard"
                    streak = 0
                    event["next_opponent"] = level
            save(latest, model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                 update=update, level=level, streak=streak, steps=steps)
            record(metrics, event)


def write_games(path, games):
    with path.open("a", encoding="utf-8") as output:
        for game in games:
            output.write(json.dumps(game) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("mlp", "cnn32"), required=True)
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--rollout", type=int, default=2048)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument("--eval-pairs", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--start-level", choices=tuple(SEARCH_LEVELS), default="easy")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.bc_epochs, args.updates, args.minibatch, args.update_epochs,
           args.eval_pairs, args.eval_every) < 1 or args.rollout < 2 or args.max_plies <= 7:
        parser.error("Training sizes must be positive, with at least two rollout steps and eight plies")
    train(args)


if __name__ == "__main__":
    main()
