"""Offline regression tests for finishing checkpointed games before new ones."""
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from quoridor_lab.collect import collect
from quoridor_lab.opponents import Decision, LocalPolicy, RemoteError
from quoridor_lab.rules import State


collect_module = importlib.import_module("quoridor_lab.collect")
NEAR_WIN = ["e2", "d9", "e3", "c9", "e4", "b9", "e5", "a9",
            "e6", "b9", "e7", "a9", "e8", "b9"]


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class VirtualClock:
    def __init__(self):
        self.elapsed = 0.0
        self.epoch = datetime(2026, 11, 15, tzinfo=timezone.utc)
        self.sleeps = []
        self.on_sleep = None
        self.monotonic_calls = 0

    def monotonic(self):
        self.monotonic_calls += 1
        if self.monotonic_calls > 50000:
            raise AssertionError("Collector spun without finishing or waiting")
        return self.elapsed

    def sleep(self, seconds):
        if seconds <= 0:
            raise AssertionError("Cooldown should wait for a positive interval")
        self.sleeps.append(seconds)
        self.elapsed += seconds
        if self.on_sleep:
            self.on_sleep()

    def now(self):
        return self.epoch + timedelta(seconds=self.elapsed)


class VirtualStop:
    def __init__(self, clock):
        self.clock = clock
        self.value = False

    def is_set(self):
        return self.value

    def set(self):
        self.value = True

    def wait(self, timeout=None):
        if not self.value:
            if timeout is None:
                raise AssertionError("Collector must use bounded cooldown waits")
            self.clock.sleep(timeout)
        return self.value


class ImmediateExecutor:
    """Finish fake network calls synchronously so queue ordering is deterministic."""
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


class ScriptedExpert:
    cache_hits = 0

    def __init__(self, clock, failures=None, before_choose=None):
        self.clock = clock
        self.failures = dict(failures or {})
        self.before_choose = before_choose
        self.calls = []
        self.policy = LocalPolicy(seed=0)
        self.request_gate = SimpleNamespace(stopped=VirtualStop(clock))

    @property
    def network_requests(self):
        return len(self.calls)

    def choose(self, state, *, difficulty):
        self.calls.append((self.clock.elapsed, state.history))
        if self.before_choose:
            self.before_choose(state)
        ply = len(state.history)
        if self.failures.get(ply, 0):
            self.failures[ply] -= 1
            raise RemoteError("Illegal expert response", kind="illegal_response")
        action = self.policy.choose(state).action
        return Decision(action, f"barricade:{difficulty}",
                        metadata={"difficulty": difficulty})


@contextmanager
def virtual_scheduler(clock):
    class ClockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = clock.now()
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    with ExitStack() as stack:
        stack.enter_context(patch.object(collect_module, "time", SimpleNamespace(
            monotonic=clock.monotonic, sleep=clock.sleep,
            time=lambda: clock.now().timestamp())))
        stack.enter_context(patch.object(collect_module, "datetime", ClockDatetime))
        stack.enter_context(patch.object(collect_module, "ThreadPoolExecutor", ImmediateExecutor))
        yield


def checkpoint(root, clock, episode_id, moves=NEAR_WIN, *, number=1, retry=None):
    pending = root / "pending"
    pending.mkdir(exist_ok=True)
    config = {"expert_side": None, "policy": "shortest", "seed": 0,
              "max_plies": 32, "opponent": "barricade",
              "remote_sides": ["master", "master"]}
    saved = {"episode_id": episode_id, "game_number": number, "config": config,
             "started_at": clock.now().isoformat(), "moves": list(moves), "random_state": None}
    if retry:
        saved["retry_state"] = retry
        saved["deferred_error"] = {"kind": "illegal_response", "ply": len(moves),
                                   "message": "Illegal expert response"}
    path = pending / f"{episode_id}.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    state = State()
    with (root / "transitions.jsonl").open("a", encoding="utf-8") as handle:
        for ply, move in enumerate(moves):
            after = state.apply(move)
            handle.write(json.dumps({"episode_id": episode_id, "ply": ply,
                "history": list(state.history), "notation": move, "after": after.to_dict()}) + "\n")
            state = after
    return path


def run_drain(root, expert, **options):
    settings = dict(output=root, games=1, max_plies=32, resume=True,
                    remote_sides=("master", "master"), concurrent_games=1,
                    request_lanes=[expert], games_per_connection=1,
                    illegal_move_retries=2, defer_illegal_responses=True,
                    deferred_retry_limit=5, retry_transient=True,
                    drain_pending=True, progress=lambda _: None)
    settings.update(options)
    return collect(expert, **settings)


class BacklogCollectionTests(unittest.TestCase):
    def test_real_workers_finish_both_checkpoints_before_starting_a_new_game(self):
        clock = VirtualClock()
        barrier = threading.Barrier(2, timeout=3)
        stopped = threading.Event()
        worker_ids = set()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint(root, clock, "old-a", number=1)
            checkpoint(root, clock, "old-b", number=2)

            def before_choose(state):
                if state.history == tuple(NEAR_WIN):
                    self.assertEqual(read_rows(root / "episodes.jsonl"), [])
                    worker_ids.add(threading.get_ident())
                    barrier.wait()
                elif not state.history:
                    self.assertEqual({r["episode_id"] for r in read_rows(root / "episodes.jsonl")},
                                     {"old-a", "old-b"})

            lanes = [ScriptedExpert(clock, before_choose=before_choose) for _ in range(2)]
            for lane in lanes:
                lane.request_gate = SimpleNamespace(stopped=stopped)
            result = run_drain(root, lanes[0], games=3, concurrent_games=2, request_lanes=lanes)
            self.assertIsNone(result["error"])
            self.assertEqual(len(worker_ids), 2)
            self.assertEqual(result["finished_games"], 3)
            self.assertEqual({r["episode_id"] for r in result["episodes"][:2]}, {"old-a", "old-b"})
            trace = read_rows(root / "transitions.jsonl")
            keys = [(r["episode_id"], r["ply"]) for r in trace]
            self.assertEqual(len(keys), len(set(keys)))
            for game in result["episodes"]:
                steps = [r for r in trace if r["episode_id"] == game["episode_id"]]
                self.assertEqual(game["status"], "completed")
                self.assertEqual([r["ply"] for r in steps], list(range(game["ply_count"])))
                self.assertEqual([r["notation"] for r in steps], game["moves"])

    def test_empty_backlog_still_dispatches_multiple_games_concurrently(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            dispatched = []

            def before_choose(state):
                # Both slots should dispatch before either one-move game is committed.
                self.assertEqual(state.history, tuple(NEAR_WIN))
                self.assertEqual(read_rows(root / "episodes.jsonl"), [])
                dispatched.append(state.history)

            lanes = [ScriptedExpert(clock, before_choose=before_choose) for _ in range(2)]
            result = run_drain(root, lanes[0], games=2, resume=False, opening=NEAR_WIN,
                               concurrent_games=2, request_lanes=lanes)
            self.assertIsNone(result["error"])
            self.assertEqual(len(dispatched), 2)
            self.assertEqual(result["finished_games"], 2)
            self.assertTrue(all(r["status"] == "completed" for r in result["episodes"]))

    def test_exhausting_old_retry_budget_still_finishes_without_duplicate_steps(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            path = checkpoint(root, clock, "old-game")
            schedules = []

            def observe_retry():
                retry = json.loads(path.read_text(encoding="utf-8")).get("retry_state")
                if retry and (not schedules or schedules[-1] != retry):
                    schedules.append(retry)

            clock.on_sleep = observe_retry
            # Six whole rounds fail; the seventh finally returns a legal winning move.
            expert = ScriptedExpert(clock, failures={len(NEAR_WIN): 19})
            result = run_drain(root, expert)
            self.assertIsNone(result["error"])
            [game] = result["episodes"]
            self.assertEqual(game["episode_id"], "old-game")
            self.assertEqual(game["status"], "completed")
            self.assertEqual(game["moves"], NEAR_WIN + ["e9"])
            self.assertEqual(len(expert.calls), 20)
            self.assertEqual([r["delay_seconds"] for r in schedules], [30, 60, 120, 240, 300, 300])
            trace = read_rows(root / "transitions.jsonl")
            self.assertEqual([r["ply"] for r in trace], list(range(15)))
            self.assertEqual([r["notation"] for r in trace], game["moves"])
            self.assertFalse(path.exists())

    def test_finishes_cooling_checkpoint_before_opening_any_new_game(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            checkpoint(root, clock, "existing-game", retry={"ply": 14, "count": 2,
                "retry_at": (clock.now() + timedelta(seconds=60)).isoformat(), "delay_seconds": 60})

            def check_before_new_game(state):
                if not state.history:
                    finished = read_rows(root / "episodes.jsonl")
                    self.assertEqual([r["episode_id"] for r in finished], ["existing-game"])

            expert = ScriptedExpert(clock, before_choose=check_before_new_game)
            result = run_drain(root, expert, games=2)
            self.assertIsNone(result["error"])
            self.assertEqual(result["finished_games"], 2)
            self.assertEqual(expert.calls[0], (60, tuple(NEAR_WIN)))
            self.assertEqual(result["episodes"][0]["episode_id"], "existing-game")
            self.assertTrue(all(r["status"] == "completed" for r in result["episodes"]))
            trace = read_rows(root / "transitions.jsonl")
            keys = [(r["episode_id"], r["ply"]) for r in trace]
            self.assertEqual(len(keys), len(set(keys)))

    def test_pause_during_cooldown_returns_and_preserves_saved_history(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            retry = {"ply": 14, "count": 6,
                     "retry_at": (clock.now() + timedelta(seconds=300)).isoformat(), "delay_seconds": 300}
            path = checkpoint(root, clock, "paused-game", retry=retry)
            expert = ScriptedExpert(clock)
            clock.on_sleep = expert.request_gate.stopped.set
            result = run_drain(root, expert, games=2)
            self.assertEqual(result["error"]["kind"], "stopped")
            self.assertEqual(expert.calls, [])
            self.assertEqual(read_rows(root / "episodes.jsonl"), [])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["moves"], NEAR_WIN)
            self.assertEqual(saved["retry_state"], retry)
            self.assertLess(clock.elapsed, 300)
            self.assertEqual(list((root / "pending").glob("*.json")), [path])

    def test_rate_limit_stops_drain_instead_of_becoming_an_illegal_move_retry(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            path = checkpoint(root, clock, "limited-game")

            def limit(state):
                raise RemoteError("Rate limited", kind="rate_limited", retryable=True)

            expert = ScriptedExpert(clock, before_choose=limit)
            result = run_drain(root, expert)
            self.assertEqual(result["error"]["kind"], "rate_limited")
            self.assertEqual(len(expert.calls), 1)
            self.assertTrue(expert.request_gate.stopped.is_set())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["moves"], NEAR_WIN)
            self.assertEqual(read_rows(root / "episodes.jsonl"), [])
            self.assertEqual(read_rows(root / "deferred-games.jsonl"), [])

    def test_earlier_retry_is_not_blocked_by_a_longer_cooldown_checkpoint(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            for episode_id, number, delay in (("a-later", 1, 300), ("b-ready-first", 2, 30)):
                checkpoint(root, clock, episode_id, number=number, retry={"ply": 14, "count": 1,
                    "retry_at": (clock.now() + timedelta(seconds=delay)).isoformat(), "delay_seconds": delay})
            expert = ScriptedExpert(clock)
            result = run_drain(root, expert, games=2)
            self.assertIsNone(result["error"])
            self.assertEqual([r["episode_id"] for r in result["episodes"]], ["b-ready-first", "a-later"])
            self.assertEqual([when for when, _ in expert.calls], [30, 300])
            self.assertEqual(list((root / "pending").glob("*.json")), [])

    def test_backoff_count_resets_after_a_successful_move(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            path = checkpoint(root, clock, "progressing-game", moves=NEAR_WIN[:12])
            schedules = []

            def observe_retry():
                retry = json.loads(path.read_text(encoding="utf-8")).get("retry_state")
                if retry and (not schedules or schedules[-1] != retry):
                    schedules.append(retry)

            clock.on_sleep = observe_retry
            expert = ScriptedExpert(clock, failures={12: 6, 13: 3})
            result = run_drain(root, expert)
            self.assertIsNone(result["error"])
            self.assertEqual(result["episodes"][0]["status"], "completed")
            self.assertEqual([(r["ply"], r["count"], r["delay_seconds"]) for r in schedules],
                             [(12, 1, 30), (12, 2, 60), (13, 1, 30)])
            self.assertEqual([r["ply"] for r in read_rows(root / "transitions.jsonl")], list(range(15)))


if __name__ == "__main__":
    unittest.main()
