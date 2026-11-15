"""Offline regressions for archiving repeatedly invalid games and replacing them."""
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

from quoridor_lab.opponents import RemoteError
from quoridor_lab.rules import State
from tests.test_backlog_collection import (
    NEAR_WIN, ScriptedExpert, VirtualClock, checkpoint, read_rows,
    run_drain, virtual_scheduler,
)


def run_with_replacements(root, expert, **options):
    return run_drain(root, expert, abandon_after_retries=4, **options)


def retry_state(clock, *, ply=14, count=4, delay=0):
    return {"ply": ply, "count": count,
            "retry_at": (clock.now() + timedelta(seconds=delay)).isoformat(),
            "delay_seconds": delay}


def archived_record(saved, clock):
    state = State.from_moves(saved["moves"])
    return {"schema_version": 1, "episode_id": saved["episode_id"],
            "game_number": saved["game_number"], "config": saved["config"],
            "started_at": saved["started_at"], "finished_at": clock.now().isoformat(),
            "status": "abandoned", "winner": None,
            "finished_reason": "illegal_response_retry_limit", "teacher_reward": None,
            "ply_count": len(saved["moves"]), "moves": saved["moves"], "final": state.to_dict(),
            "run_seconds": 0, "retry_count": 5, "retry_limit": 4,
            "error": {"kind": "illegal_response", "message": "Repeated invalid response"}}


class AbandonedCollectionTests(unittest.TestCase):
    def assert_no_duplicate_steps(self, root):
        trace = read_rows(root / "transitions.jsonl")
        keys = [(row["episode_id"], row["ply"]) for row in trace]
        self.assertEqual(len(keys), len(set(keys)))
        return trace

    def test_four_failed_rounds_can_still_finish_the_original_game(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            checkpoint(root, clock, "four-rounds")
            expert = ScriptedExpert(clock, failures={14: 12})
            result = run_with_replacements(root, expert)
            self.assertIsNone(result["error"])
            self.assertEqual(result["finished_games"], 1)
            self.assertEqual(result["episodes"][0]["episode_id"], "four-rounds")
            self.assertEqual(result["episodes"][0]["status"], "completed")
            self.assertEqual(len(expert.calls), 13)
            self.assertEqual(read_rows(root / "abandoned-games.jsonl"), [])
            self.assert_no_duplicate_steps(root)

    def test_fifth_failed_round_archives_history_and_replacement_fills_target(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            path = checkpoint(root, clock, "abandoned-old", number=7)
            old_trace = read_rows(root / "transitions.jsonl")
            calls_when_archive_exists = []

            def inspect_boundary(state):
                if state.history == tuple(NEAR_WIN):
                    calls_when_archive_exists.append(bool(read_rows(root / "abandoned-games.jsonl")))

            expert = ScriptedExpert(clock, failures={14: 15}, before_choose=inspect_boundary)
            result = run_with_replacements(root, expert)
            self.assertIsNone(result["error"])
            self.assertEqual(result["finished_games"], 1)
            [finished] = result["episodes"]
            self.assertEqual(finished["status"], "completed")
            self.assertNotEqual(finished["episode_id"], "abandoned-old")
            self.assertNotEqual(finished["game_number"], 7)
            self.assertEqual(calls_when_archive_exists, [False] * 15)
            [abandoned] = read_rows(root / "abandoned-games.jsonl")
            self.assertEqual(abandoned["episode_id"], "abandoned-old")
            self.assertEqual(abandoned["status"], "abandoned")
            self.assertEqual(abandoned["finished_reason"], "illegal_response_retry_limit")
            self.assertEqual((abandoned["retry_count"], abandoned["retry_limit"]), (5, 4))
            self.assertEqual(abandoned["error"]["kind"], "illegal_response")
            self.assertEqual(abandoned["moves"], NEAR_WIN)
            self.assertEqual(abandoned["final"], State.from_moves(NEAR_WIN).to_dict())
            self.assertIsNone(abandoned["winner"])
            self.assertFalse(path.exists())
            saved_archive = json.loads((root / "episode-abandoned-old.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_archive, abandoned)
            self.assertEqual(read_rows(root / "episodes.jsonl"), [finished])
            trace = self.assert_no_duplicate_steps(root)
            self.assertEqual([r for r in trace if r["episode_id"] == "abandoned-old"], old_trace)
            self.assertEqual([r["notation"] for r in trace if r["episode_id"] == finished["episode_id"]],
                             finished["moves"])

    def test_advancing_to_another_position_resets_the_abandonment_count(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            checkpoint(root, clock, "advancing", moves=NEAR_WIN[:12])
            expert = ScriptedExpert(clock, failures={12: 12, 13: 12})
            result = run_with_replacements(root, expert)
            self.assertIsNone(result["error"])
            [finished] = result["episodes"]
            self.assertEqual(finished["episode_id"], "advancing")
            self.assertEqual(finished["status"], "completed")
            self.assertEqual(read_rows(root / "abandoned-games.jsonl"), [])
            self.assertEqual([r["ply"] for r in self.assert_no_duplicate_steps(root)], list(range(15)))

    def test_multiple_abandoned_replacements_do_not_consume_the_requested_target(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            checkpoint(root, clock, "first-abandoned")
            # The old game fails, then the first replacement also fails at its opening.
            expert = ScriptedExpert(clock, failures={14: 15, 0: 15})
            result = run_with_replacements(root, expert)
            self.assertIsNone(result["error"])
            self.assertEqual(result["finished_games"], 1)
            self.assertEqual(len(read_rows(root / "episodes.jsonl")), 1)
            abandoned = read_rows(root / "abandoned-games.jsonl")
            self.assertEqual(len(abandoned), 2)
            self.assertEqual([r["ply_count"] for r in abandoned], [14, 0])
            self.assertEqual(result["episodes"][0]["status"], "completed")
            all_games = abandoned + result["episodes"]
            self.assertEqual(len({r["episode_id"] for r in all_games}), 3)
            self.assertEqual(len({r["game_number"] for r in all_games}), 3)
            self.assertEqual(list((root / "pending").glob("*.json")), [])
            self.assert_no_duplicate_steps(root)

    def test_replacement_starts_while_another_old_game_is_still_cooling(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            checkpoint(root, clock, "ready-to-abandon", number=1, retry=retry_state(clock))
            cooling = checkpoint(root, clock, "still-cooling", number=2,
                                 retry=retry_state(clock, count=1, delay=90))
            replacement_started = []

            def before_choose(state):
                if not state.history:
                    replacement_started.append(clock.elapsed)
                    self.assertTrue(cooling.exists())
                    self.assertLess(clock.elapsed, 90)
                    self.assertEqual(read_rows(root / "episodes.jsonl"), [])

            expert = ScriptedExpert(clock, failures={14: 3}, before_choose=before_choose)
            result = run_with_replacements(root, expert, games=2)
            self.assertIsNone(result["error"])
            self.assertEqual(len(replacement_started), 1)
            self.assertEqual(result["finished_games"], 2)
            self.assertEqual(len(read_rows(root / "abandoned-games.jsonl")), 1)
            self.assertNotIn(result["episodes"][0]["episode_id"], {"ready-to-abandon", "still-cooling"})
            self.assertEqual(result["episodes"][1]["episode_id"], "still-cooling")
            self.assertTrue(all(g["status"] == "completed" for g in result["episodes"]))
            self.assertEqual(len({g["game_number"] for g in result["episodes"]} | {1}), 3)
            self.assert_no_duplicate_steps(root)

    def test_rate_limit_still_pauses_without_abandoning_a_four_failure_position(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            path = checkpoint(root, clock, "limited", retry=retry_state(clock))

            def rate_limit(state):
                raise RemoteError("Rate limited", kind="rate_limited", retryable=True)

            expert = ScriptedExpert(clock, before_choose=rate_limit)
            result = run_with_replacements(root, expert)
            self.assertEqual(result["error"]["kind"], "rate_limited")
            self.assertEqual(len(expert.calls), 1)
            self.assertTrue(expert.request_gate.stopped.is_set())
            self.assertEqual(read_rows(root / "abandoned-games.jsonl"), [])
            self.assertEqual(read_rows(root / "episodes.jsonl"), [])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["moves"], NEAR_WIN)

    def test_pause_on_fifth_failure_preserves_the_checkpoint_until_next_resume(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            path = checkpoint(root, clock, "paused-at-limit", retry=retry_state(clock))

            def pause_then_fail(state):
                expert.request_gate.stopped.set()
                raise RemoteError("Illegal response during pause", kind="illegal_response")

            expert = ScriptedExpert(clock, before_choose=pause_then_fail)
            result = run_with_replacements(root, expert, illegal_move_retries=0)
            self.assertEqual(result["error"]["kind"], "stopped")
            self.assertEqual(len(expert.calls), 1)
            self.assertEqual(read_rows(root / "abandoned-games.jsonl"), [])
            self.assertEqual(read_rows(root / "episodes.jsonl"), [])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["moves"], NEAR_WIN)
            self.assertEqual(saved["retry_state"]["count"], 5)

            resumed_expert = ScriptedExpert(clock)
            resumed = run_with_replacements(root, resumed_expert)
            self.assertIsNone(resumed["error"])
            self.assertEqual(resumed["finished_games"], 1)
            self.assertNotEqual(resumed["episodes"][0]["episode_id"], "paused-at-limit")
            [abandoned] = read_rows(root / "abandoned-games.jsonl")
            self.assertEqual(abandoned["episode_id"], "paused-at-limit")
            self.assertEqual(abandoned["retry_count"], 5)
            self.assertFalse(path.exists())
            self.assert_no_duplicate_steps(root)

    def test_resuming_an_over_limit_checkpoint_archives_it_before_any_request(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            path = checkpoint(root, clock, "already-exhausted", number=4,
                              retry=retry_state(clock, count=5, delay=300))
            expert = ScriptedExpert(clock)
            result = run_with_replacements(root, expert)
            self.assertIsNone(result["error"])
            self.assertEqual(result["finished_games"], 1)
            self.assertEqual(expert.calls[0], (0, ()))
            self.assertTrue(all(history != tuple(NEAR_WIN) for _, history in expert.calls))
            [abandoned] = read_rows(root / "abandoned-games.jsonl")
            self.assertEqual(abandoned["episode_id"], "already-exhausted")
            self.assertEqual(abandoned["retry_count"], 5)
            self.assertFalse(path.exists())
            self.assertNotEqual(result["episodes"][0]["game_number"], 4)
            self.assert_no_duplicate_steps(root)

    def test_resume_cleans_an_archived_checkpoint_without_rearchiving_or_replaying_it(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            path = checkpoint(root, clock, "crash-after-archive", number=1,
                              retry=retry_state(clock, count=5))
            saved = json.loads(path.read_text(encoding="utf-8"))
            archived = archived_record(saved, clock)
            # Crash after appending the archive but before the JSON file and unlink.
            archive_path = root / "abandoned-games.jsonl"
            archive_path.write_text(json.dumps(archived) + "\n", encoding="utf-8")
            expert = ScriptedExpert(clock)
            result = run_with_replacements(root, expert)
            self.assertIsNone(result["error"])
            self.assertEqual(result["finished_games"], 1)
            self.assertEqual(expert.calls[0][1], ())
            self.assertTrue(all(history != tuple(NEAR_WIN) for _, history in expert.calls))
            self.assertEqual(read_rows(archive_path), [archived])
            self.assertFalse(path.exists())
            self.assertEqual(json.loads((root / "episode-crash-after-archive.json").read_text(encoding="utf-8")),
                             archived)
            self.assertNotEqual(result["episodes"][0]["game_number"], 1)
            trace = self.assert_no_duplicate_steps(root)
            self.assertEqual([r["notation"] for r in trace if r["episode_id"] == "crash-after-archive"],
                             NEAR_WIN)

    def test_crash_before_starting_replacement_restores_it_ahead_of_cooling_backlog(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            old_path = checkpoint(root, clock, "archived-before-crash", number=1)
            archived = archived_record(json.loads(old_path.read_text(encoding="utf-8")), clock)
            (root / "abandoned-games.jsonl").write_text(json.dumps(archived) + "\n", encoding="utf-8")
            (root / "episode-archived-before-crash.json").write_text(json.dumps(archived), encoding="utf-8")
            old_path.unlink()
            cooling = checkpoint(root, clock, "cooling", number=2,
                                 retry=retry_state(clock, count=1, delay=90))
            new_calls = []

            def before_choose(state):
                if not state.history:
                    self.assertTrue(cooling.exists())
                    self.assertEqual(clock.elapsed, 0)
                    new_calls.append(state.history)

            expert = ScriptedExpert(clock, before_choose=before_choose)
            result = run_with_replacements(root, expert, games=2)
            self.assertIsNone(result["error"])
            self.assertEqual(len(new_calls), 1)
            self.assertEqual(result["finished_games"], 2)
            self.assertEqual(result["episodes"][0]["config"]["replaces_episode_id"], "archived-before-crash")
            self.assertEqual(result["episodes"][1]["episode_id"], "cooling")
            self.assertEqual(read_rows(root / "abandoned-games.jsonl"), [archived])
            self.assert_no_duplicate_steps(root)

    def test_existing_replacement_checkpoint_prevents_a_second_replacement_after_resume(self):
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            old_path = checkpoint(root, clock, "already-replaced", number=1)
            archived = archived_record(json.loads(old_path.read_text(encoding="utf-8")), clock)
            (root / "abandoned-games.jsonl").write_text(json.dumps(archived) + "\n", encoding="utf-8")
            old_path.unlink()
            checkpoint(root, clock, "cooling", number=2, retry=retry_state(clock, count=1, delay=90))
            replacement = checkpoint(root, clock, "saved-replacement", number=3)
            saved = json.loads(replacement.read_text(encoding="utf-8"))
            saved["config"]["replaces_episode_id"] = "already-replaced"
            replacement.write_text(json.dumps(saved), encoding="utf-8")

            def before_choose(state):
                self.assertTrue(state.history, "An already-replaced archive must not open another game")

            expert = ScriptedExpert(clock, before_choose=before_choose)
            result = run_with_replacements(root, expert, games=2)
            self.assertIsNone(result["error"])
            self.assertEqual(result["finished_games"], 2)
            self.assertEqual({r["episode_id"] for r in result["episodes"]}, {"cooling", "saved-replacement"})
            self.assertEqual(read_rows(root / "abandoned-games.jsonl"), [archived])
            self.assert_no_duplicate_steps(root)


if __name__ == "__main__":
    unittest.main()
