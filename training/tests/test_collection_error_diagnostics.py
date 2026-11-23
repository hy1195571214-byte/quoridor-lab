"""Provider cooldown details must survive the collector's failure boundary."""
import json
from pathlib import Path
import tempfile
import unittest

from quoridor_lab.collect import collect
from quoridor_lab.opponents import RemoteError
from tests.test_backlog_collection import (
    NEAR_WIN, ScriptedExpert, VirtualClock, checkpoint, read_rows, run_drain, virtual_scheduler,
)


class CollectionErrorDiagnosticsTests(unittest.TestCase):
    def check_rate_limit_preservation(self, *, use_lanes):
        diagnostics = {
            "provider_response": {"error": "rate_limited", "retryAfterMs": 300000},
            "observed_at": "2026-11-15T08:49:15.125000+00:00",
            "latency_seconds": 0.125,
        }
        expected_error = {
            "kind": "rate_limited",
            "message": "Provider requested a cooldown",
            "diagnostics": diagnostics,
        }
        clock = VirtualClock()
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            path = checkpoint(root, clock, "rate-limited-game")
            original_trace = read_rows(root / "transitions.jsonl")

            def rate_limit(state):
                raise RemoteError(expected_error["message"], kind="rate_limited",
                                  retryable=True, diagnostics=diagnostics)

            expert = ScriptedExpert(clock, before_choose=rate_limit)
            settings = dict(output=root, games=2, max_plies=32, resume=True,
                            remote_sides=("master", "master"), concurrent_games=1,
                            illegal_move_retries=2, retry_transient=True,
                            completed_only=True, progress=lambda _: None)
            if use_lanes:
                settings.update(request_lanes=[expert], games_per_connection=1,
                                defer_illegal_responses=True, deferred_retry_limit=5,
                                drain_pending=True, abandon_after_retries=4)
            result = collect(expert, **settings)

            self.assertEqual(result["error"], expected_error)
            self.assertEqual(json.loads((root / "last-error.json").read_text(encoding="utf-8")),
                             {"episode_id": "rate-limited-game", **expected_error})
            self.assertEqual(result["network_requests"], 1)
            self.assertEqual(expert.calls, [(0, tuple(NEAR_WIN))])
            self.assertEqual(result["finished_games"], 0)
            self.assertEqual(result["completed_games"], 0)
            self.assertEqual(result.get("abandoned_games", 0), 0)
            self.assertEqual(read_rows(root / "abandoned-games.jsonl"), [])
            self.assertEqual(read_rows(root / "episodes.jsonl"), [])
            self.assertEqual(read_rows(root / "deferred-games.jsonl"), [])
            self.assertEqual(read_rows(root / "transport-retries.jsonl"), [])
            self.assertEqual(read_rows(root / "transitions.jsonl"), original_trace)
            self.assertEqual(list((root / "pending").glob("*.json")), [path])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["moves"], NEAR_WIN)
            if use_lanes:
                self.assertTrue(expert.request_gate.stopped.is_set())

    def test_rate_limit_diagnostics_survive_lane_collection_without_retry_or_abandonment(self):
        self.check_rate_limit_preservation(use_lanes=True)

    def test_rate_limit_diagnostics_survive_sequential_collection_without_another_request(self):
        self.check_rate_limit_preservation(use_lanes=False)

    def test_illegal_reply_context_survives_immediate_retries_deferral_and_abandonment(self):
        clock = VirtualClock()
        observed = []
        with tempfile.TemporaryDirectory() as tmp, virtual_scheduler(clock):
            root = Path(tmp)
            path = checkpoint(root, clock, "invalid-reply-game")
            original_trace = read_rows(root / "transitions.jsonl")

            def illegal_reply(state):
                if state.history != tuple(NEAR_WIN):
                    return
                correlation = f"failed-request-{len(observed) + 1}"
                diagnostics = {
                    "provider_response": {"ok": True, "correlationId": correlation, "move": "e2"},
                    "request": {"moves": ",".join(NEAR_WIN), "difficulty": "master", "correlationId": correlation},
                    "observed_at": clock.now().isoformat(),
                    "latency_seconds": 0.5,
                    "move": "e2",
                    "rule_error": "illegal-move",
                    "correlation_verified": True,
                }
                observed.append(diagnostics)
                raise RemoteError("Illegal expert response: e2", kind="illegal_response", diagnostics=diagnostics)

            expert = ScriptedExpert(clock, before_choose=illegal_reply)
            # Stop in the first cooldown so all three failures can be inspected in place.
            clock.on_sleep = expert.request_gate.stopped.set
            result = run_drain(root, expert, abandon_after_retries=4, completed_only=True)
            self.assertEqual(result["error"]["kind"], "stopped")
            self.assertEqual(len(expert.calls), 3)
            self.assertEqual(result["completed_games"], 0)
            self.assertEqual(result["abandoned_games"], 0)
            rejected = read_rows(root / "rejected-moves.jsonl")
            self.assertEqual([row["attempt"] for row in rejected], [1, 2])
            self.assertEqual([row["diagnostics"] for row in rejected], observed[:2])
            [deferred] = read_rows(root / "deferred-games.jsonl")
            self.assertEqual(deferred["episode_id"], "invalid-reply-game")
            self.assertEqual(deferred["ply"], len(NEAR_WIN))
            self.assertEqual(deferred["diagnostics"], observed[2])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["moves"], NEAR_WIN)
            self.assertEqual(saved["deferred_error"]["diagnostics"], observed[2])
            self.assertEqual(read_rows(root / "transitions.jsonl"), original_trace)
            self.assertEqual(read_rows(root / "episodes.jsonl"), [])

            # Resume the same position until round five archives it, then finish its replacement.
            clock.on_sleep = None
            resumed_expert = ScriptedExpert(clock, before_choose=illegal_reply)
            resumed = run_drain(root, resumed_expert, abandon_after_retries=4, completed_only=True)
            self.assertIsNone(resumed["error"])
            self.assertEqual(resumed["completed_games"], 1)
            self.assertEqual(resumed["abandoned_games"], 1)
            self.assertEqual(len(observed), 15)
            [archived] = read_rows(root / "abandoned-games.jsonl")
            self.assertEqual(archived["episode_id"], "invalid-reply-game")
            self.assertEqual(archived["retry_count"], 5)
            self.assertEqual(archived["error"]["diagnostics"], observed[-1])
            self.assertEqual(archived["moves"], NEAR_WIN)
            self.assertFalse(path.exists())
            self.assertEqual([row["diagnostics"] for row in read_rows(root / "rejected-moves.jsonl")],
                             [diag for index, diag in enumerate(observed) if index % 3 != 2])
            self.assertEqual([row["diagnostics"] for row in read_rows(root / "deferred-games.jsonl")],
                             observed[2::3])
            self.assertEqual([row for row in read_rows(root / "transitions.jsonl")
                              if row["episode_id"] == "invalid-reply-game"], original_trace)


if __name__ == "__main__":
    unittest.main()
