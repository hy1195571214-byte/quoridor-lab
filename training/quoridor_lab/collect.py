"""Collect and checkpoint games through a shared, turn-by-turn request queue."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from .rules import State
from .opponents import LocalPolicy, RemoteError, RemoteExpert


def _append(path, record):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def _atomic(path, record):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


def collect(opponent, *, output="data/collection", games=1, policy="shortest", seed=0,
            expert_side="alternate", max_plies=256, resume=False, progress=print,
            remote_sides=None, opening=(), concurrent_games=1):
    if games < 1 or max_plies < 2 or expert_side not in {"1", "2", "alternate"}:
        raise ValueError("Invalid collection configuration")
    if remote_sides is not None:
        remote_sides = list(remote_sides)
        if len(remote_sides) != 2 or any(level not in RemoteExpert.difficulties for level in remote_sides):
            raise ValueError("Provide one valid remote difficulty per side")
    if concurrent_games < 1 or (concurrent_games > 1 and remote_sides is None):
        raise ValueError("Concurrent collection requires dual remote play")
    opening_state = State.from_moves(opening)
    if opening_state.winner or len(opening_state.history) >= max_plies:
        raise ValueError("Opening must leave room for further play")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    pending_dir = root / "pending"
    pending_dir.mkdir(exist_ok=True)
    finished = []
    # A crashed run can leave either a resumable checkpoint or a final record
    # written immediately before its checkpoint was deleted.
    existing = {}
    opponent_name = getattr(opponent, "difficulty", getattr(opponent, "name", "unknown"))
    if (root / "episodes.jsonl").exists():
        for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            existing[item["episode_id"]] = item
    pending = sorted(pending_dir.glob("*.json")) if resume else []
    if remote_sides:
        for saved in list(pending):
            if saved.stem in existing:
                result_path = root / f"episode-{saved.stem}.json"
                if not result_path.exists():
                    _atomic(result_path, existing[saved.stem])
                saved.unlink()
                pending.remove(saved)
    saved_indices = {
        item["game_number"] for item in existing.values() if "game_number" in item
    }
    if remote_sides:
        for saved in pending:
            checkpoint = json.loads(saved.read_text(encoding="utf-8"))
            if "game_number" in checkpoint:
                saved_indices.add(checkpoint["game_number"])

    def play(game_index, resumed):
        if resumed:
            checkpoint = json.loads(resumed.read_text(encoding="utf-8"))
            config = checkpoint["config"]
            if config.get("remote_sides") != remote_sides or (
                remote_sides is None and config["opponent"] != opponent_name
            ):
                raise ValueError("Resume opponent differs from checkpoint")
            max_game_plies = config["max_plies"]
            episode_id, side = checkpoint["episode_id"], config["expert_side"]
            state = State.from_moves(checkpoint["moves"])
            baseline = None if remote_sides else LocalPolicy(config["policy"], config["seed"])
            if baseline:
                baseline.rng.setstate(_tuples(checkpoint["random_state"]))
            if episode_id in existing:
                resumed.unlink()
                return existing[episode_id], None
        else:
            episode_id = uuid.uuid4().hex
            # Across repeated invocations, alternate sides based on saved games.
            side = None if remote_sides else (
                1 + ((len(existing) + game_index) % 2) if expert_side == "alternate" else int(expert_side)
            )
            config = {"expert_side": side, "policy": policy, "seed": seed + game_index,
                      "max_plies": max_plies,
                      "opponent": opponent_name}
            if remote_sides:
                config.update(opponent="barricade", remote_sides=remote_sides)
            if opening_state.history:
                config["opening"] = list(opening_state.history)
            max_game_plies, state = max_plies, opening_state
            baseline = None if remote_sides else LocalPolicy(policy, seed + game_index)
            checkpoint = {"episode_id": episode_id, "config": config,
                          "started_at": datetime.now(timezone.utc).isoformat(),
                          "moves": list(state.history), "random_state": baseline.rng.getstate() if baseline else None}
        if remote_sides and "game_number" not in checkpoint:
            game_number = 1
            while game_number in saved_indices:
                game_number += 1
            saved_indices.add(game_number)
            checkpoint["game_number"] = game_number
        checkpoint_path = pending_dir / f"{episode_id}.json"
        # Truncate uncommitted trailing records from an interrupted write.
        # Keep rows for other episodes. A committed prefix is the checkpoint history.
        transitions_path = root / "transitions.jsonl"
        if resumed and transitions_path.exists():
            kept = []
            for line in transitions_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row["episode_id"] != episode_id or row["ply"] < len(state.history):
                    kept.append(line)
            temp = transitions_path.with_suffix(".tmp")
            temp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            temp.replace(transitions_path)
        _atomic(checkpoint_path, checkpoint)
        start = time.monotonic()
        progress(f"game {game_index + 1}/{games} id={episode_id} expert_side={side} starting_ply={len(state.history)}")
        yield
        try:
            while state.winner is None and len(state.history) < max_game_plies:
                teacher_turn = bool(remote_sides) or state.turn == side
                if remote_sides:
                    decision = opponent.choose(state, difficulty=remote_sides[state.turn - 1])
                else:
                    decision = (opponent if teacher_turn else baseline).choose(state)
                after = state.apply(decision.action)
                obs = state.observation(max_game_plies)
                row = {"schema_version": 1, "episode_id": episode_id, "ply": len(state.history),
                       "player": state.turn, "is_teacher": teacher_turn,
                       "history": list(state.history), "observation": obs["board"].tolist(),
                       "action_mask": obs["action_mask"].tolist(),
                       **decision.to_dict(), "after": after.to_dict()}
                _append(transitions_path, row)
                state = after
                checkpoint.update(moves=list(state.history), random_state=baseline.rng.getstate() if baseline else None)
                _atomic(checkpoint_path, checkpoint)
                if teacher_turn:
                    progress(f"  ply={len(state.history)} expert={decision.action.notation} cached={decision.cached}")
                if state.winner is None and len(state.history) < max_game_plies:
                    yield
        except (RemoteError, KeyboardInterrupt) as exc:
            error = {"kind": getattr(exc, "kind", "interrupted"), "message": str(exc)}
            _atomic(root / "last-error.json", {"episode_id": episode_id, **error})
            progress(f"paused id={episode_id} ply={len(state.history)} reason={error['kind']}; resume with --resume")
            return None, error
        result = {"schema_version": 1, "episode_id": episode_id, "config": config,
                  "started_at": checkpoint["started_at"],
                  "finished_at": datetime.now(timezone.utc).isoformat(),
                  "status": "completed" if state.winner else "truncated",
                  "winner": state.winner, "finished_reason": "goal" if state.winner else "max_plies",
                  "teacher_reward": None if remote_sides else ((1 if state.winner == side else -1) if state.winner else 0),
                  "ply_count": len(state.history), "moves": list(state.history),
                  "final": state.to_dict(), "run_seconds": time.monotonic() - start}
        if "game_number" in checkpoint:
            result["game_number"] = checkpoint["game_number"]
        _append(root / "episodes.jsonl", result)
        _atomic(root / f"episode-{episode_id}.json", result)
        checkpoint_path.unlink(missing_ok=True)
        progress(f"finished id={episode_id} status={result['status']} winner={state.winner} plies={len(state.history)}")
        return result, None

    # One worker owns the connection and disk writes. A game queues its next
    # request only after its last reply has been applied and checkpointed.
    queue = deque()
    started = 0
    error = None
    while queue or started < games:
        while len(queue) < concurrent_games and started < games:
            game = play(started, pending.pop(0) if pending else None)
            started += 1
            try:
                next(game)
            except StopIteration as done:
                finished.append(done.value[0])
            else:
                queue.append(game)
        if not queue:
            continue
        game = queue.popleft()
        try:
            next(game)
        except StopIteration as done:
            result, error = done.value
            if error:
                break
            finished.append(result)
        else:
            queue.append(game)
    return {"requested_games": games, "finished_games": len(finished), "episodes": finished,
            "error": error,
            "network_requests": getattr(opponent, "network_requests", 0),
            "cache_hits": getattr(opponent, "cache_hits", 0)}
