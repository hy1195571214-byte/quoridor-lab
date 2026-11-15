"""Collect and checkpoint games through a shared, turn-by-turn request queue."""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
import time
import uuid

from .rules import State
from .opponents import LocalPolicy, RemoteError, RemoteExpert


DEFERRED_RETRY_DELAY = 30


def _retry_file_access(operation, *args, **kwargs):
    for attempt in range(6):
        try:
            return operation(*args, **kwargs)
        except PermissionError:
            if attempt == 5:
                raise
            # Windows can briefly lock a file while another thread opens or replaces it.
            time.sleep(0.02 * (attempt + 1))


def _append(path, record):
    with _retry_file_access(Path(path).open, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def _atomic(path, record):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    _replace(temp, path)


def _replace(temp, path):
    _retry_file_access(temp.replace, path)


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


def collect(opponent, *, output="data/collection", games=1, policy="shortest", seed=0,
            expert_side="alternate", max_plies=256, resume=False, progress=print,
            remote_sides=None, opening=(), concurrent_games=1, deadline=None,
            request_lanes=(), games_per_connection=None, illegal_move_retries=0,
            defer_illegal_responses=False, deferred_retry_limit=0, retry_transient=False):
    if games < 1 or max_plies < 2 or expert_side not in {"1", "2", "alternate"}:
        raise ValueError("Invalid collection configuration")
    if not 0 <= illegal_move_retries <= 2:
        raise ValueError("Illegal move retry count must be between 0 and 2")
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
    saved_checkpoints = {}
    transitions_path = root / "transitions.jsonl"
    retry_log_lock = Lock()

    def choose_remote(expert, state, difficulty, options, episode_id):
        illegal_attempts = 0
        retried_transport = False
        while True:
            try:
                return expert.choose(state, difficulty=difficulty, **options)
            except RemoteError as exc:
                if retry_transient and exc.retryable and exc.kind in {"connection", "transport"} and not retried_transport:
                    retried_transport = True
                    with retry_log_lock:
                        _append(root / "transport-retries.jsonl", {
                            "episode_id": episode_id, "ply": len(state.history), "difficulty": difficulty,
                            "kind": exc.kind, "error": str(exc), "observed_at": datetime.now(timezone.utc).isoformat(),
                        })
                    continue
                if exc.kind != "illegal_response" or illegal_attempts == illegal_move_retries:
                    raise
                illegal_attempts += 1
                with retry_log_lock:
                    _append(root / "rejected-moves.jsonl", {
                        "episode_id": episode_id, "ply": len(state.history), "difficulty": difficulty,
                        "attempt": illegal_attempts, "error": str(exc),
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "diagnostics": exc.diagnostics,
                    })
    if remote_sides:
        for saved in pending:
            checkpoint = json.loads(saved.read_text(encoding="utf-8"))
            saved_checkpoints[saved] = checkpoint
            if "game_number" in checkpoint:
                saved_indices.add(checkpoint["game_number"])
        if saved_checkpoints and transitions_path.exists():
            prefixes = {checkpoint["episode_id"]: len(checkpoint["moves"])
                        for checkpoint in saved_checkpoints.values()}
            temp = transitions_path.with_suffix(".tmp")
            with transitions_path.open(encoding="utf-8") as source, temp.open("w", encoding="utf-8") as target:
                for line in source:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        if line.endswith("\n"):
                            raise
                        break
                    if row["episode_id"] not in prefixes or row["ply"] < prefixes[row["episode_id"]]:
                        target.write(line)
            _replace(temp, transitions_path)

    def play(game_index, resumed):
        if resumed:
            checkpoint = saved_checkpoints.pop(resumed, None)
            if checkpoint is None:
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
        if resumed and not remote_sides and transitions_path.exists():
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
        expired = yield
        try:
            while not expired and state.winner is None and len(state.history) < max_game_plies:
                teacher_turn = bool(remote_sides) or state.turn == side
                if remote_sides:
                    options = {"deadline": deadline} if deadline is not None else {}
                    difficulty = remote_sides[state.turn - 1]
                    if request_lanes:
                        decision = yield state, difficulty, options, episode_id
                    else:
                        decision = choose_remote(opponent, state, difficulty, options, episode_id)
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
                checkpoint.pop("deferred_error", None)
                checkpoint.update(moves=list(state.history), random_state=baseline.rng.getstate() if baseline else None)
                _atomic(checkpoint_path, checkpoint)
                if teacher_turn:
                    progress(f"  ply={len(state.history)} expert={decision.action.notation} cached={decision.cached}")
                if state.winner is None and len(state.history) < max_game_plies:
                    expired = yield
        except (RemoteError, KeyboardInterrupt) as exc:
            error = {"kind": getattr(exc, "kind", "interrupted"), "message": str(exc)}
            if error["kind"] == "time_limit":
                expired = True
            elif defer_illegal_responses and request_lanes and error["kind"] == "illegal_response":
                checkpoint["deferred_error"] = {
                    "ply": len(state.history), "difficulty": difficulty, **error,
                    "observed_at": datetime.now(timezone.utc).isoformat(), "diagnostics": exc.diagnostics,
                }
                _atomic(checkpoint_path, checkpoint)
                _append(root / "deferred-games.jsonl", {"episode_id": episode_id, **checkpoint["deferred_error"]})
                return None, {**checkpoint["deferred_error"], "episode_id": episode_id, "game_index": game_index}
            else:
                if error["kind"] != "stopped":
                    _atomic(root / "last-error.json", {"episode_id": episode_id, **error})
                progress(f"paused id={episode_id} ply={len(state.history)} reason={error['kind']}; resume with --resume")
                return None, error
        result = {"schema_version": 1, "episode_id": episode_id, "config": config,
                  "started_at": checkpoint["started_at"],
                  "finished_at": datetime.now(timezone.utc).isoformat(),
                  "status": "completed" if state.winner else "truncated",
                  "winner": state.winner, "finished_reason": "goal" if state.winner else (
                      "time_limit" if expired else "max_plies"),
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
    cutoff = False
    sampling_stopped_at = None
    if request_lanes:
        deferred = 0
        deferred_queue = deque()
        retry_counts = {}

        def request_move(expert, state, difficulty, options, episode_id):
            try:
                return choose_remote(expert, state, difficulty, options, episode_id)
            except RemoteError as failure:
                if failure.kind != "time_limit" and not (
                    defer_illegal_responses and failure.kind == "illegal_response"
                ):
                    expert.request_gate.stopped.set()
                raise

        def advance(game, value=None, failure=None):
            nonlocal cutoff, deferred, error, sampling_stopped_at
            try:
                return game.throw(failure) if failure else game.send(value)
            except StopIteration as done:
                result, failure = done.value
                if failure:
                    if defer_illegal_responses and failure["kind"] == "illegal_response":
                        episode_id = failure["episode_id"]
                        ply, count = retry_counts.get(episode_id, (-1, 0))
                        count = count + 1 if ply == failure["ply"] else 1
                        if count <= deferred_retry_limit and not cutoff:
                            retry_counts[episode_id] = (failure["ply"], count)
                            path = pending_dir / f"{episode_id}.json"
                            due = time.monotonic() + DEFERRED_RETRY_DELAY
                            deferred_queue.append((due, failure["game_index"], path))
                        else:
                            deferred += 1
                    elif error is None or error["kind"] == "stopped":
                        error = failure
                else:
                    finished.append(result)
                    retry_counts.pop(result["episode_id"], None)
                    if result["finished_reason"] == "time_limit" and not cutoff:
                        cutoff = True
                        sampling_stopped_at = time.monotonic()
                return None

        def expired():
            nonlocal cutoff, sampling_stopped_at
            if not cutoff and deadline is not None and time.monotonic() >= deadline:
                cutoff = True
                sampling_stopped_at = time.monotonic()
            return cutoff

        lanes = [deque() for _ in request_lanes]
        inflight = {}
        with ThreadPoolExecutor(max_workers=len(lanes)) as workers:
            while inflight or any(lanes) or deferred_queue or started < games:
                for lane, waiting in enumerate(lanes):
                    if lane in inflight:
                        continue
                    while not error and not expired() and len(waiting) < games_per_connection:
                        if sum(map(len, lanes)) + len(inflight) >= concurrent_games:
                            break
                        if deferred_queue and deferred_queue[0][0] <= time.monotonic():
                            _, game_index, checkpoint = deferred_queue.popleft()
                        elif started < games:
                            game_index = started
                            checkpoint = pending.pop(0) if pending else None
                            started += 1
                        else:
                            break
                        game = play(game_index, checkpoint)
                        try:
                            next(game)
                        except StopIteration as done:
                            finished.append(done.value[0])
                        else:
                            waiting.append(game)
                    if error:
                        continue
                    if expired():
                        while waiting:
                            advance(waiting.popleft(), True)
                    elif waiting:
                        game = waiting.popleft()
                        request = advance(game, False)
                        if request is not None:
                            future = workers.submit(request_move, request_lanes[lane], *request)
                            inflight[lane] = (future, game)
                if not inflight:
                    if error or cutoff:
                        break
                    if not any(lanes) and started == games:
                        if not deferred_queue:
                            break
                        time.sleep(min(1, max(0, deferred_queue[0][0] - time.monotonic())))
                    continue
                ready, _ = wait([future for future, _ in inflight.values()], return_when=FIRST_COMPLETED)
                for lane, (future, game) in list(inflight.items()):
                    if future not in ready:
                        continue
                    del inflight[lane]
                    failure = future.exception()
                    if failure:
                        advance(game, failure=failure)
                    else:
                        if advance(game, future.result()) is not None:
                            raise RuntimeError("A game requested two moves without a queue turn")
                        if game.gi_frame is not None:
                            lanes[lane].append(game)
        deferred += len(deferred_queue)
        if deferred and error is None:
            error = {"kind": "illegal_response", "message": f"{deferred} 盘收到非法应手，已保留进度，可恢复采集。"}
        return {"requested_games": games, "finished_games": len(finished), "episodes": finished,
                "error": error, "time_limit_reached": cutoff, "sampling_stopped_at": sampling_stopped_at,
                "network_requests": sum(lane.network_requests for lane in request_lanes),
                "cache_hits": sum(lane.cache_hits for lane in request_lanes)}

    while queue or started < games:
        if not cutoff and deadline is not None and time.monotonic() >= deadline:
            cutoff = True
            sampling_stopped_at = time.monotonic()
        while not cutoff and len(queue) < concurrent_games and started < games:
            if deadline is not None and time.monotonic() >= deadline:
                cutoff = True
                sampling_stopped_at = time.monotonic()
                break
            game = play(started, pending.pop(0) if pending else None)
            started += 1
            try:
                next(game)
            except StopIteration as done:
                finished.append(done.value[0])
            else:
                queue.append(game)
        if not queue:
            if cutoff:
                break
            continue
        game = queue.popleft()
        try:
            game.send(cutoff)
        except StopIteration as done:
            result, error = done.value
            if error:
                break
            finished.append(result)
            if result["finished_reason"] == "time_limit" and not cutoff:
                cutoff = True
                sampling_stopped_at = time.monotonic()
        else:
            queue.append(game)
    return {"requested_games": games, "finished_games": len(finished), "episodes": finished,
            "error": error, "time_limit_reached": cutoff, "sampling_stopped_at": sampling_stopped_at,
            "network_requests": getattr(opponent, "network_requests", 0),
            "cache_hits": getattr(opponent, "cache_hits", 0)}
