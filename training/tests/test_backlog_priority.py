"""Offline behavior checks for fair, endgame-aware checkpoint scheduling."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from quoridor_lab.opponents import RemoteError
from quoridor_lab.rules import State
from tests.test_abandoned_collection import retry_state
from tests.test_backlog_collection import (
    NEAR_WIN, ScriptedExpert, VirtualClock, checkpoint, read_rows,
    run_drain, virtual_scheduler,
)


FAR_LOOP = ["e2", "e8", "e1", "e9"]


def saved_game(root, clock, episode_id, moves, *, number, retry=None):
    """Allow long, legal detours without truncating the test's saved game."""
    path = checkpoint(root, clock, episode_id, moves, number=number, retry=retry)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["config"]["max_plies"] = 128
    path.write_text(json.dumps(saved), encoding="utf-8")
    return path


class BacklogPriorityTests(unittest.TestCase):
    def assert_saved_once(self, root, result, expected_ids):
        self.assertIsNone(result["error"])
        self.assertEqual(result["completed_games"], len(expected_ids))
        episodes = read_rows(root / "episodes.jsonl")
        self.assertEqual(len(episodes), len(expected_ids))
        self.assertEqual({game["episode_id"] for game in episodes}, set(expected_ids))
        self.assertTrue(all(game["status"] == "completed" for game in episodes))
        transitions = read_rows(root / "transitions.jsonl")
        keys = [(row["episode_id"], row["ply"]) for row in transitions]
        self.assertEqual(len(keys), len(set(keys)))
        for game in episodes:
            steps = [row for row in transitions if row["episode_id"] == game["episode_id"]]
            self.assertEqual([row["ply"] for row in steps], list(range(game["ply_count"])))
            self.assertEqual([row["notation"] for row in steps], game["moves"])
        self.assertEqual(list((root / "pending").glob("*.json")), [])

    def test_short_near_goal_history_precedes_long_far_history_for_either_player(self):
        # In the second case blue is to move and is still eight squares away:
        # red being two squares away must also qualify the position as near.
        for near_moves in (NEAR_WIN, NEAR_WIN + ["e7"]):
            with self.subTest(moves=near_moves):
                clock = VirtualClock()
                with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
                    root = Path(tmp)
                    far_moves = FAR_LOOP * 10
                    self.assertGreater(min(State.from_moves(far_moves).distance(p) for p in (1, 2)), 2)
                    self.assertLessEqual(min(State.from_moves(near_moves).distance(p) for p in (1, 2)), 2)
                    self.assertLess(len(near_moves), len(far_moves))
                    saved_game(root, clock, "far-long", far_moves, number=1)
                    saved_game(root, clock, "near-short", near_moves, number=2)
                    expert = ScriptedExpert(clock)
                    result = run_drain(root, expert, games=2, max_plies=128)
                    self.assertEqual(expert.calls[0][1], tuple(near_moves))
                    self.assert_saved_once(root, result, {"far-long", "near-short"})

    def test_far_ready_game_gets_a_turn_while_many_near_games_remain(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            far_moves = FAR_LOOP * 10
            saved_game(root, clock, "far", far_moves, number=1)
            near_ids = []
            for number in range(6):
                episode_id = f"near-{number}"
                near_ids.append(episode_id)
                saved_game(root, clock, episode_id, FAR_LOOP * number + NEAR_WIN,
                           number=number + 2)
            expert = ScriptedExpert(clock)
            result = run_drain(root, expert, games=7, max_plies=128)
            # Every near game would win with its first request. Reserving the
            # intervening ordinary turn gives the far game a bounded wait.
            far_first = next(i for i, (_, history) in enumerate(expert.calls)
                             if history == tuple(far_moves))
            self.assertEqual(far_first, 1)
            self.assert_saved_once(root, result, {"far", *near_ids})

    def test_near_goal_priority_never_bypasses_a_future_retry_time(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            saved_game(root, clock, "cooling-near", NEAR_WIN, number=1,
                       retry=retry_state(clock, ply=len(NEAR_WIN), count=1, delay=60))
            far_moves = FAR_LOOP * 5
            saved_game(root, clock, "ready-far", far_moves, number=2)
            expert = ScriptedExpert(clock)
            result = run_drain(root, expert, games=2, max_plies=128)
            near_calls = [(when, history) for when, history in expert.calls
                          if history == tuple(NEAR_WIN)]
            self.assertEqual(expert.calls[0], (0, tuple(far_moves)))
            self.assertEqual(near_calls, [(60, tuple(NEAR_WIN))])
            self.assertTrue(clock.sleeps)
            self.assert_saved_once(root, result, {"cooling-near", "ready-far"})

    def test_due_retry_and_unstarted_checkpoints_both_progress_before_backlog_drains(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            histories = {}
            for number, loops in enumerate(range(12, 5, -1), 1):
                episode_id = "retry-far" if number == 1 else f"pending-{number}"
                moves = FAR_LOOP * loops
                histories[tuple(moves)] = episode_id
                saved_game(root, clock, episode_id, moves, number=number)
            selected = []
            failed = False

            def track_first_requests(state):
                nonlocal failed
                episode_id = histories.get(state.history)
                if episode_id is None:
                    return
                selected.append((episode_id, clock.elapsed))
                if episode_id == "retry-far" and not failed:
                    failed = True
                    raise RemoteError("One invalid response", kind="illegal_response")
                if episode_id != "retry-far" and clock.elapsed < 30:
                    # Make the deferred request eligible while six original
                    # checkpoints still compete for the single connection.
                    clock.elapsed = 31

            expert = ScriptedExpert(clock, before_choose=track_first_requests)
            result = run_drain(root, expert, games=7, max_plies=128,
                               illegal_move_retries=0)
            ids = [episode_id for episode_id, _ in selected]
            retry_positions = [i for i, episode_id in enumerate(ids) if episode_id == "retry-far"]
            self.assertEqual(len(retry_positions), 2)
            retry_index = retry_positions[1]
            self.assertLessEqual(retry_index, 3,
                                 "An eligible retry must not wait for every initial checkpoint")
            self.assertTrue(any(name != "retry-far" for name in ids[:retry_index]))
            self.assertTrue(any(name != "retry-far" for name in ids[retry_index + 1:]))
            self.assertGreaterEqual(selected[retry_index][1], 30)
            self.assert_saved_once(root, result, set(histories.values()))

    def test_legal_progress_refreshes_near_goal_priority_before_a_deferred_retry(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            progressing = FAR_LOOP * 10 + NEAR_WIN[:10]
            far_moves = FAR_LOOP * 8
            other_near = NEAR_WIN + ["e7"]
            self.assertEqual(min(State.from_moves(progressing).distance(p) for p in (1, 2)), 3)
            saved_game(root, clock, "progressing", progressing, number=1)
            saved_game(root, clock, "far", far_moves, number=2)
            saved_game(root, clock, "other-near", other_near, number=3,
                       retry=retry_state(clock, ply=len(other_near), count=1, delay=30))
            failed_history = None
            events = []

            def track_progress(state):
                nonlocal failed_history
                if state.history == tuple(far_moves):
                    clock.elapsed = 31
                if state.history == tuple(other_near):
                    events.append("other-near")
                if failed_history is not None and state.history == failed_history:
                    events.append("progressed-retry")
                elif (failed_history is None
                      and state.history[:len(progressing)] == tuple(progressing)
                      and len(state.history) == len(progressing) + 2):
                    # Red has advanced from e6 to e7 and blue has replied.
                    # This position is now near the goal although its initial
                    # checkpoint was not; red is the closer player to move.
                    self.assertEqual(state.distance(1), 2)
                    failed_history = state.history
                    events.append("progressed-defer")
                    raise RemoteError("Invalid response after legal progress", kind="illegal_response")

            expert = ScriptedExpert(clock, before_choose=track_progress)
            result = run_drain(root, expert, games=3, max_plies=128,
                               illegal_move_retries=0)
            self.assertIsNotNone(failed_history)
            self.assertLess(events.index("progressed-retry"), events.index("other-near"))
            self.assert_saved_once(root, result, {"progressing", "far", "other-near"})

    def test_two_real_workers_never_dispatch_the_same_checkpoint_twice(self):
        clock = VirtualClock()
        barrier = threading.Barrier(2, timeout=5)
        stopped = threading.Event()
        lock = threading.Lock()
        active = set()
        dispatched = []
        worker_ids = set()
        histories = {tuple(FAR_LOOP * i + NEAR_WIN): f"near-{i}" for i in range(4)}
        testcase = self

        class ConcurrentExpert(ScriptedExpert):
            def choose(self, state, *, difficulty):
                with lock:
                    testcase.assertIn(state.history, histories)
                    episode_id = histories[state.history]
                    testcase.assertNotIn(episode_id, active)
                    active.add(episode_id)
                    dispatched.append(episode_id)
                    worker_ids.add(threading.get_ident())
                try:
                    barrier.wait()
                    return super().choose(state, difficulty=difficulty)
                finally:
                    with lock:
                        active.remove(episode_id)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for number, (moves, episode_id) in enumerate(histories.items(), 1):
                saved_game(root, clock, episode_id, moves, number=number)
            lanes = [ConcurrentExpert(clock) for _ in range(2)]
            for lane in lanes:
                lane.request_gate = SimpleNamespace(stopped=stopped)
            result = run_drain(root, lanes[0], games=4, max_plies=128,
                               concurrent_games=2, request_lanes=lanes)
            self.assertEqual(len(worker_ids), 2)
            self.assertEqual(len(dispatched), 4)
            self.assertEqual(set(dispatched), set(histories.values()))
            self.assertEqual(active, set())
            self.assert_saved_once(root, result, set(histories.values()))


if __name__ == "__main__":
    unittest.main()
