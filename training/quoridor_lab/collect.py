"""Sequential collection; transitions are flushed per ply and games are resumable."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from .rules import State
from .opponents import LocalPolicy, RemoteError


def _append(path, record):
    with Path(path).open("a") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def _atomic(path, record):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    temp.replace(path)


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


def collect(opponent, *, output="data/collection", games=1, policy="shortest", seed=0,
            expert_side="alternate", max_plies=256, resume=False, progress=print):
    if games < 1 or max_plies < 2 or expert_side not in {"1", "2", "alternate"}:
        raise ValueError("Invalid collection configuration")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    pending_dir = root / "pending"
    pending_dir.mkdir(exist_ok=True)
    finished = []
    # A crashed run can leave either a resumable checkpoint or a final record
    # written immediately before its checkpoint was deleted.
    existing = {}
    if (root / "episodes.jsonl").exists():
        for line in (root / "episodes.jsonl").read_text().splitlines():
            item = json.loads(line)
            existing[item["episode_id"]] = item
    pending = sorted(pending_dir.glob("*.json")) if resume else []
    for game_index in range(games):
        resumed = pending.pop(0) if pending else None
        if resumed:
            checkpoint = json.loads(resumed.read_text())
            config = checkpoint["config"]
            if config["opponent"] != getattr(opponent, "difficulty", getattr(opponent, "name", "unknown")):
                raise ValueError("Resume opponent differs from checkpoint")
            max_game_plies = config["max_plies"]
            episode_id, side = checkpoint["episode_id"], config["expert_side"]
            state = State.from_moves(checkpoint["moves"])
            baseline = LocalPolicy(config["policy"], config["seed"])
            baseline.rng.setstate(_tuples(checkpoint["random_state"]))
            if episode_id in existing:
                resumed.unlink()
                finished.append(existing[episode_id])
                continue
        else:
            episode_id = uuid.uuid4().hex
            # Across repeated invocations, alternate sides based on saved games.
            side = 1 + ((len(existing) + game_index) % 2) if expert_side == "alternate" else int(expert_side)
            config = {"expert_side": side, "policy": policy, "seed": seed + game_index,
                      "max_plies": max_plies,
                      "opponent": getattr(opponent, "difficulty", getattr(opponent, "name", "unknown"))}
            max_game_plies, state = max_plies, State()
            baseline = LocalPolicy(policy, seed + game_index)
            checkpoint = {"episode_id": episode_id, "config": config,
                          "started_at": datetime.now(timezone.utc).isoformat(),
                          "moves": [], "random_state": baseline.rng.getstate()}
        checkpoint_path = pending_dir / f"{episode_id}.json"
        # Truncate uncommitted trailing records from an interrupted write.
        # Keep rows for other episodes. A committed prefix is the checkpoint history.
        transitions_path = root / "transitions.jsonl"
        if resumed and transitions_path.exists():
            kept = []
            for line in transitions_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row["episode_id"] != episode_id or row["ply"] < len(state.history):
                    kept.append(line)
            temp = transitions_path.with_suffix(".tmp")
            temp.write_text("\n".join(kept) + ("\n" if kept else ""))
            temp.replace(transitions_path)
        _atomic(checkpoint_path, checkpoint)
        error = None
        start = time.monotonic()
        progress(f"game {game_index + 1}/{games} id={episode_id} expert_side={side} starting_ply={len(state.history)}")
        try:
            while state.winner is None and len(state.history) < max_game_plies:
                teacher_turn = state.turn == side
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
                checkpoint.update(moves=list(state.history), random_state=baseline.rng.getstate())
                _atomic(checkpoint_path, checkpoint)
                if teacher_turn:
                    progress(f"  ply={len(state.history)} expert={decision.action.notation} cached={decision.cached}")
        except (RemoteError, KeyboardInterrupt) as exc:
            error = {"kind": getattr(exc, "kind", "interrupted"), "message": str(exc)}
            _atomic(root / "last-error.json", {"episode_id": episode_id, **error})
            progress(f"paused id={episode_id} ply={len(state.history)} reason={error['kind']}; resume with --resume")
            break  # In particular: no retries or identity rotation on rate limiting.
        result = {"schema_version": 1, "episode_id": episode_id, "config": config,
                  "started_at": checkpoint["started_at"],
                  "finished_at": datetime.now(timezone.utc).isoformat(),
                  "status": "completed" if state.winner else "truncated",
                  "winner": state.winner, "finished_reason": "goal" if state.winner else "max_plies",
                  "teacher_reward": (1 if state.winner == side else -1) if state.winner else 0,
                  "ply_count": len(state.history), "moves": list(state.history),
                  "final": state.to_dict(), "run_seconds": time.monotonic() - start}
        _append(root / "episodes.jsonl", result)
        _atomic(root / f"episode-{episode_id}.json", result)
        checkpoint_path.unlink(missing_ok=True)
        finished.append(result)
        progress(f"finished id={episode_id} status={result['status']} winner={state.winner} plies={len(state.history)}")
    return {"requested_games": games, "finished_games": len(finished), "episodes": finished,
            "error": error if 'error' in locals() else None,
            "network_requests": getattr(opponent, "network_requests", 0),
            "cache_hits": getattr(opponent, "cache_hits", 0)}
