"""Collect and checkpoint games through a shared, turn-by-turn request queue."""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
import heapq
import json
from pathlib import Path
from threading import Lock
import time
import uuid

from .rules import State
from .opponents import LocalPolicy, RemoteError, RemoteExpert
from .quality import e2_quality


DEFERRED_RETRY_DELAY = 30
MAX_DEFERRED_RETRY_DELAY = 300


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
            defer_illegal_responses=False, deferred_retry_limit=0, retry_transient=False,
            drain_pending=False, abandon_after_retries=None, completed_only=False):
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
    if drain_pending and not (remote_sides and request_lanes and defer_illegal_responses):
        raise ValueError("Backlog collection requires deferred remote request lanes")
    if abandon_after_retries is not None and (
        not isinstance(abandon_after_retries, int) or abandon_after_retries < 0
        or not (remote_sides and request_lanes and defer_illegal_responses)
    ):
        raise ValueError("Abandonment requires a nonnegative retry limit and deferred remote lanes")
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
    abandoned = {}
    opponent_name = getattr(opponent, "difficulty", getattr(opponent, "name", "unknown"))
    if (root / "episodes.jsonl").exists():
        for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            existing[item["episode_id"]] = item
    if (root / "abandoned-games.jsonl").exists():
        for line in (root / "abandoned-games.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            abandoned[item["episode_id"]] = item
    pending = sorted(pending_dir.glob("*.json")) if resume else []
    if remote_sides:
        for saved in list(pending):
            if saved.stem in existing or saved.stem in abandoned:
                result_path = root / f"episode-{saved.stem}.json"
                if not result_path.exists():
                    _atomic(result_path, existing.get(saved.stem) or abandoned[saved.stem])
                _retry_file_access(saved.unlink, missing_ok=True)
                pending.remove(saved)
    saved_indices = {
        item["game_number"] for item in (*existing.values(), *abandoned.values()) if "game_number" in item
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
                if (exc.diagnostics or {}).get('quality'):
                    with retry_log_lock:
                        _append(root / 'e2-events.jsonl', {
                            'episode_id': episode_id, 'ply': len(state.history), 'kind': exc.kind,
                            'status': ('quarantined' if exc.kind == 'suspect_e2' else
                                       'rejected' if exc.kind in {'illegal_response','e2_service_unhealthy'} else 'verification_failed'),
                            'observed_at': datetime.now(timezone.utc).isoformat(),
                            'quality': exc.diagnostics['quality'], 'diagnostics': exc.diagnostics,
                        })
                if retry_transient and exc.retryable and exc.kind in {"connection", "transport"} and not retried_transport:
                    retried_transport = True
                    with retry_log_lock:
                        _append(root / "transport-retries.jsonl", {
                            "episode_id": episode_id, "ply": len(state.history), "difficulty": difficulty,
                            "kind": exc.kind, "error": str(exc), "observed_at": datetime.now(timezone.utc).isoformat(),
                        })
                    continue
                retry_limit = max(1, illegal_move_retries) if exc.kind == 'suspect_e2' else illegal_move_retries
                if exc.kind not in {"illegal_response", "suspect_e2"} or illegal_attempts >= retry_limit:
                    raise
                illegal_attempts += 1
                with retry_log_lock:
                    _append(root / "rejected-moves.jsonl", {
                        "episode_id": episode_id, "ply": len(state.history), "difficulty": difficulty,
                        "attempt": illegal_attempts, "error": str(exc),
                            "kind": exc.kind,
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "diagnostics": exc.diagnostics,
                    })
    if remote_sides:
        for saved in pending:
            checkpoint = json.loads(saved.read_text(encoding="utf-8"))
            saved_checkpoints[saved] = checkpoint
            if "game_number" in checkpoint:
                saved_indices.add(checkpoint["game_number"])
        if drain_pending:
            pending.sort(key=lambda path: (
                -len(saved_checkpoints[path]["moves"]),
                saved_checkpoints[path].get("game_number", games + 1),
                path.name,
            ))
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

    priority_scores = {}

    def remember_priority(path, state):
        if drain_pending:
            distances = (state.distance(1), state.distance(2))
            priority_scores[path] = (min(distances), distances[state.turn - 1])

    def checkpoint_priority(path):
        if path not in priority_scores:
            saved = saved_checkpoints.get(path)
            if saved is None:
                saved = json.loads(_retry_file_access(path.read_text, encoding="utf-8"))
            remember_priority(path, State.from_moves(saved["moves"]))
        return priority_scores[path]

    def play(game_index, resumed, replacement_for=None):
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
                _retry_file_access(resumed.unlink, missing_ok=True)
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
            if replacement_for is not None:
                config["replaces_episode_id"] = replacement_for
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
        remember_priority(checkpoint_path, state)
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
                    if teacher_turn and isinstance(opponent, RemoteExpert):
                        options = {"deadline": deadline} if deadline is not None else {}
                        decision = choose_remote(opponent, state, opponent.difficulty, options, episode_id)
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
                reconsidered = decision.metadata.get('e2_reconsideration')
                if reconsidered:
                    with retry_log_lock:
                        _append(root / 'e2-events.jsonl', {
                            'episode_id': episode_id, 'ply': len(state.history),
                            'status': reconsidered['outcome'],
                            'observed_at': datetime.now(timezone.utc).isoformat(),
                            'quality': reconsidered['candidate']['quality'],
                            'candidate': reconsidered['candidate'],
                            'accepted_move': decision.action.notation,
                            'accepted_request': decision.metadata.get('request_diagnostics'),
                        })
                if teacher_turn and decision.source.startswith('barricade:') and decision.action.notation == 'e2':
                    with retry_log_lock:
                        _append(root / 'e2-events.jsonl', {
                            'episode_id': episode_id, 'ply': len(state.history), 'status': 'accepted',
                            'observed_at': datetime.now(timezone.utc).isoformat(), 'cached': decision.cached,
                            'quality': decision.metadata.get('quality') or e2_quality(state, decision.action),
                        })
                state = after
                checkpoint.pop("deferred_error", None)
                checkpoint.pop("retry_state", None)
                checkpoint.update(moves=list(state.history), random_state=baseline.rng.getstate() if baseline else None)
                _atomic(checkpoint_path, checkpoint)
                remember_priority(checkpoint_path, state)
                if teacher_turn:
                    progress(f"  ply={len(state.history)} expert={decision.action.notation} cached={decision.cached}")
                if state.winner is None and len(state.history) < max_game_plies:
                    expired = yield
        except (RemoteError, KeyboardInterrupt) as exc:
            error = {"kind": getattr(exc, "kind", "interrupted"), "message": str(exc)}
            if getattr(exc, "diagnostics", None):
                error["diagnostics"] = exc.diagnostics
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
        _retry_file_access(checkpoint_path.unlink, missing_ok=True)
        progress(f"finished id={episode_id} status={result['status']} winner={state.winner} plies={len(state.history)}")
        return result, None

    # One worker owns the connection and disk writes. A game queues its next
    # request only after its last reply has been applied and checkpointed.
    queue = deque()
    started = 0
    next_game_index = 0
    error = None
    cutoff = False
    sampling_stopped_at = None
    if request_lanes:
        deferred = 0
        deferred_queue = []
        retry_counts = {}
        retry_serial = 0
        replaced_ids = {item["config"].get("replaces_episode_id")
                        for item in (*existing.values(), *abandoned.values(), *saved_checkpoints.values())}
        replacement_sources = dict(abandoned) if abandon_after_retries is not None else {}
        if completed_only:
            replacement_sources.update({episode_id: item for episode_id, item in existing.items()
                                        if item["status"] == "truncated"})
        replacement_queue = deque(episode_id for episode_id in replacement_sources
                                  if episode_id not in replaced_ids)
        prefer_replacement = True
        abandoned_this_run = []
        backlog_ids = {item["episode_id"] for item in saved_checkpoints.values()} if drain_pending else set()
        stop = getattr(getattr(request_lanes[0], "request_gate", None), "stopped", None)
        prioritize_near = True
        prefer_deferred = True

        def abandon_game(path, count, *, counted):
            nonlocal started
            saved = saved_checkpoints.pop(path, None)
            if saved is None:
                saved = json.loads(_retry_file_access(path.read_text, encoding="utf-8"))
            episode_id = saved["episode_id"]
            if episode_id not in abandoned:
                state = State.from_moves(saved["moves"])
                result = {
                    "schema_version": 1, "episode_id": episode_id, "config": saved["config"],
                    "started_at": saved["started_at"], "finished_at": datetime.now(timezone.utc).isoformat(),
                    "status": "abandoned", "winner": None, "finished_reason": "illegal_response_retry_limit",
                    "teacher_reward": None, "ply_count": len(saved["moves"]), "moves": saved["moves"],
                    "final": state.to_dict(), "retry_count": count, "retry_limit": abandon_after_retries,
                    "error": saved.get("deferred_error", {"kind": "illegal_response"}),
                }
                if "game_number" in saved:
                    result["game_number"] = saved["game_number"]
                _append(root / "abandoned-games.jsonl", result)
                abandoned[episode_id] = result
                abandoned_this_run.append(result)
                replacement_queue.append(episode_id)
                if counted:
                    started -= 1
            _atomic(root / f"episode-{episode_id}.json", abandoned[episode_id])
            _retry_file_access(path.unlink, missing_ok=True)
            priority_scores.pop(path, None)
            backlog_ids.discard(episode_id)
            retry_counts.pop(episode_id, None)

        if abandon_after_retries is not None:
            for path in list(pending):
                if (stop is not None and stop.is_set()) or (deadline is not None and time.monotonic() >= deadline):
                    break
                saved = saved_checkpoints[path]
                retry = saved.get("retry_state")
                if retry and retry["ply"] == len(saved["moves"]) and retry["count"] > abandon_after_retries:
                    abandon_game(path, retry["count"], counted=False)
                    pending.remove(path)

        def enqueue_retry(game_index, path, delay):
            nonlocal retry_serial
            retry_serial += 1
            heapq.heappush(deferred_queue, (time.monotonic() + delay, retry_serial, game_index, path))

        if drain_pending:
            # Reserve the existing target slots, but keep future retries out of
            # the ready pending queue. Their persisted cooldown is unchanged.
            for path in list(pending):
                if started >= games or (stop is not None and stop.is_set()) or (
                    deadline is not None and time.monotonic() >= deadline
                ):
                    break
                saved = saved_checkpoints[path]
                retry = saved.get("retry_state")
                if not retry or retry["ply"] != len(saved["moves"]):
                    continue
                delay = max(0, datetime.fromisoformat(retry["retry_at"]).timestamp()
                            - datetime.now(timezone.utc).timestamp())
                if delay:
                    retry_counts[saved["episode_id"]] = (retry["ply"], retry["count"])
                    enqueue_retry(next_game_index, path, delay)
                    next_game_index += 1
                    started += 1
                    pending.remove(path)

        def choose_ready_existing():
            nonlocal prioritize_near, prefer_deferred
            now = time.monotonic()
            pending_ready = bool(pending) and started < games
            due_indices = [index for index, item in enumerate(deferred_queue) if item[0] <= now]
            selected = None
            if prioritize_near:
                candidates = []
                if pending_ready:
                    for index, path in enumerate(pending):
                        score = checkpoint_priority(path)
                        if score[0] <= 2:
                            candidates.append((*score, 1, (index,), "pending", index))
                for index in due_indices:
                    due, serial, _, path = deferred_queue[index]
                    score = checkpoint_priority(path)
                    if score[0] <= 2:
                        candidates.append((*score, 0, (due, serial), "deferred", index))
                if candidates:
                    selected = min(candidates)[-2:]
            if selected is None:
                if due_indices and (prefer_deferred or not pending_ready):
                    selected = ("deferred", 0)
                elif pending_ready:
                    selected = ("pending", 0)
                else:
                    return None
                prefer_deferred = selected[0] == "pending"
            # Every other existing-game selection remains an ordinary turn,
            # regardless of how many games stay geometrically near a goal.
            prioritize_near = not prioritize_near
            return selected

        def pop_deferred(index):
            if index == 0:
                return heapq.heappop(deferred_queue)
            item = deferred_queue.pop(index)
            heapq.heapify(deferred_queue)
            return item

        def request_move(expert, state, difficulty, options, episode_id):
            try:
                return choose_remote(expert, state, difficulty, options, episode_id)
            except RemoteError as failure:
                if failure.kind != "time_limit" and not (
                    defer_illegal_responses and failure.kind == "illegal_response"
                ):
                    expert.request_gate.stopped.set()
                raise

        def finish_game(result):
            nonlocal started
            finished.append(result)
            priority_scores.pop(pending_dir / f"{result['episode_id']}.json", None)
            backlog_ids.discard(result["episode_id"])
            retry_counts.pop(result["episode_id"], None)
            if completed_only and result["status"] != "completed":
                # Preserve the saved replay, but reserve this target slot for a
                # game that actually reaches a winner. Pause/deadline guards in
                # the scheduler still take precedence over opening replacements.
                started -= 1
                replacement_queue.append(result["episode_id"])

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
                        if (abandon_after_retries is not None and count > abandon_after_retries
                            and not cutoff and error is None and (stop is None or not stop.is_set())):
                            abandon_game(pending_dir / f"{episode_id}.json", count, counted=True)
                        elif (drain_pending or abandon_after_retries is not None or count <= deferred_retry_limit) and not cutoff:
                            retry_counts[episode_id] = (failure["ply"], count)
                            path = pending_dir / f"{episode_id}.json"
                            delay = DEFERRED_RETRY_DELAY
                            if drain_pending or abandon_after_retries is not None:
                                if drain_pending:
                                    backlog_ids.add(episode_id)
                                delay = min(DEFERRED_RETRY_DELAY * 2 ** min(count - 1, 4),
                                            MAX_DEFERRED_RETRY_DELAY)
                                retry_state = {
                                    "ply": failure["ply"], "count": count,
                                    "retry_at": (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat(),
                                    "delay_seconds": delay,
                                }
                                checkpoint = json.loads(_retry_file_access(path.read_text, encoding="utf-8"))
                                checkpoint["retry_state"] = retry_state
                                _atomic(path, checkpoint)
                                _append(root / "retry-schedule.jsonl", {"episode_id": episode_id, **retry_state})
                            enqueue_retry(failure["game_index"], path, delay)
                        else:
                            deferred += 1
                    elif error is None or error["kind"] == "stopped":
                        error = failure
                else:
                    finish_game(result)
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
                if stop is not None and stop.is_set() and error is None:
                    error = {"kind": "stopped", "message": "Collection stopped; checkpoints preserved."}
                for lane, waiting in enumerate(lanes):
                    if lane in inflight:
                        continue
                    while not error and not expired() and len(waiting) < games_per_connection:
                        if sum(map(len, lanes)) + len(inflight) >= concurrent_games:
                            break
                        old_ready = (bool(pending) and started < games) or bool(
                            deferred_queue and deferred_queue[0][0] <= time.monotonic()
                        )
                        replacement_for = None
                        if replacement_queue and started + len(pending) < games and (prefer_replacement or not old_ready):
                            game_index = next_game_index
                            next_game_index += 1
                            checkpoint = None
                            started += 1
                            replacement_for = replacement_queue.popleft()
                            prefer_replacement = False
                        elif drain_pending and old_ready:
                            source, index = choose_ready_existing()
                            if source == "pending":
                                game_index = next_game_index
                                next_game_index += 1
                                checkpoint = pending.pop(index)
                                started += 1
                                saved = saved_checkpoints[checkpoint]
                                retry = saved.get("retry_state")
                                if retry and retry["ply"] == len(saved["moves"]):
                                    retry_counts[saved["episode_id"]] = (retry["ply"], retry["count"])
                            else:
                                _, _, game_index, checkpoint = pop_deferred(index)
                            prefer_replacement = True
                        elif abandon_after_retries is not None and pending and started < games:
                            game_index = next_game_index
                            next_game_index += 1
                            checkpoint = pending.pop(0)
                            started += 1
                            prefer_replacement = True
                            saved = saved_checkpoints[checkpoint]
                            retry_state = saved.get("retry_state")
                            if retry_state and retry_state["ply"] == len(saved["moves"]):
                                retry_counts[saved["episode_id"]] = (retry_state["ply"], retry_state["count"])
                                delay = max(0, datetime.fromisoformat(retry_state["retry_at"]).timestamp()
                                            - datetime.now(timezone.utc).timestamp())
                                if delay:
                                    enqueue_retry(game_index, checkpoint, delay)
                                    continue
                        elif deferred_queue and deferred_queue[0][0] <= time.monotonic():
                            _, _, game_index, checkpoint = heapq.heappop(deferred_queue)
                            prefer_replacement = True
                        elif started < games and not (drain_pending and backlog_ids):
                            game_index = next_game_index
                            next_game_index += 1
                            checkpoint = pending.pop(0) if pending else None
                            started += 1
                        else:
                            break
                        game = play(game_index, checkpoint, replacement_for)
                        try:
                            next(game)
                        except StopIteration as done:
                            result = done.value[0]
                            finish_game(result)
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
                    if not any(lanes):
                        if deferred_queue:
                            delay = min(1, max(0, deferred_queue[0][0] - time.monotonic()))
                            if stop is not None:
                                stop.wait(delay)
                            else:
                                time.sleep(delay)
                        elif started == games:
                            break
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
                "completed_games": sum(item["status"] == "completed" for item in finished),
                "target_games_finished": sum(not completed_only or item["status"] == "completed" for item in finished),
                "abandoned_games": len(abandoned_this_run),
                "error": error, "time_limit_reached": cutoff, "sampling_stopped_at": sampling_stopped_at,
                "network_requests": sum(lane.network_requests for lane in request_lanes),
                "cache_hits": sum(lane.cache_hits for lane in request_lanes)}

    stop = getattr(getattr(opponent, "request_gate", None), "stopped", None)
    while queue or started < games:
        if stop is not None and stop.is_set():
            error = {"kind": "stopped", "message": "Collection stopped; checkpoints preserved."}
            break
        if not cutoff and deadline is not None and time.monotonic() >= deadline:
            cutoff = True
            sampling_stopped_at = time.monotonic()
        while not cutoff and len(queue) < concurrent_games and started < games:
            if deadline is not None and time.monotonic() >= deadline:
                cutoff = True
                sampling_stopped_at = time.monotonic()
                break
            game = play(next_game_index, pending.pop(0) if pending else None)
            next_game_index += 1
            started += 1
            try:
                next(game)
            except StopIteration as done:
                result = done.value[0]
                finished.append(result)
                if completed_only and result["status"] != "completed":
                    started -= 1
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
            if completed_only and result["status"] != "completed":
                started -= 1
            if result["finished_reason"] == "time_limit" and not cutoff:
                cutoff = True
                sampling_stopped_at = time.monotonic()
        else:
            queue.append(game)
    return {"requested_games": games, "finished_games": len(finished), "episodes": finished,
            "completed_games": sum(item["status"] == "completed" for item in finished),
            "target_games_finished": sum(not completed_only or item["status"] == "completed" for item in finished),
            "error": error, "time_limit_reached": cutoff, "sampling_stopped_at": sampling_stopped_at,
            "network_requests": getattr(opponent, "network_requests", 0),
            "cache_hits": getattr(opponent, "cache_hits", 0)}
