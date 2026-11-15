from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quoridor_lab.collect import collect
from quoridor_lab.opponents import RemoteError, RemoteExpert, RequestGate
from quoridor_lab.rules import State


class FakeSocket:
    connected = True

    def __init__(self, response):
        self.response = response
        self.calls = []

    def call(self, event, payload, **kwargs):
        self.calls.append((event, payload, kwargs))
        self.returned_response = self.response(payload) if callable(self.response) else self.response
        return self.returned_response

    def disconnect(self):
        self.connected = False


class RemoteErrorDiagnosticsTests(unittest.TestCase):
    def test_rejection_keeps_raw_response_and_existing_error_semantics(self):
        cases = [
            ({"ok": False, "error": "rate_limited", "retryAfterMs": 120000,
              "details": {"scope": "unspecified"}}, "rate_limited"),
            ({"ok": False, "retryAfterMs": "120000"}, "rejected"),
            ({"error": "access_denied", "retryAfter": "unknown"}, "access_denied"),
        ]
        for response, kind in cases:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as root, patch(
                "quoridor_lab.opponents.socketio.Client",
                side_effect=AssertionError("Tests must not create a network client"),
            ):
                expert = RemoteExpert(data_dir=root, interval=0.05)
                fake = FakeSocket(response)
                expert.client = fake
                device_id = expert.device_id
                before = datetime.now(timezone.utc)
                try:
                    with self.assertRaises(RemoteError) as raised:
                        expert.choose(State.from_moves(["e2"]))
                    after = datetime.now(timezone.utc)
                    error = raised.exception
                    self.assertEqual(error.kind, kind)
                    self.assertFalse(error.retryable)
                    self.assertEqual(error.diagnostics["provider_response"], response)
                    self.assertEqual(error.diagnostics["request"], {
                        field: fake.calls[0][1][field]
                        for field in ("moves", "difficulty", "correlationId")
                    })
                    self.assertNotIn("deviceId", error.diagnostics["request"])
                    observed = datetime.fromisoformat(error.diagnostics["observed_at"])
                    self.assertEqual(observed.utcoffset(), timedelta(0))
                    self.assertLessEqual(before, observed)
                    self.assertLessEqual(observed, after)
                    self.assertGreaterEqual(error.diagnostics["latency_seconds"], 0)
                    self.assertEqual(len(fake.calls), 1)
                    self.assertEqual(expert.network_requests, 1)
                    self.assertEqual(expert.device_id, device_id)
                    self.assertEqual(fake.calls[0][1]["deviceId"], device_id)
                    self.assertFalse(expert.cache)
                    self.assertFalse(expert.cache_path.exists())
                finally:
                    expert.close()

    def test_invalid_replies_keep_raw_diagnostics_and_actual_request_context(self):
        cases = [
            ("illegal_move", lambda payload: {
                "ok": True, "move": "e2", "correlationId": payload["correlationId"],
                "providerDebug": {"opaque": [1, "unknown"], "retryAfterMs": "undocumented"},
            }, "illegal_response"),
            ("invalid_notation", lambda payload: {
                "ok": True, "move": "invalid-notation", "correlationId": payload["correlationId"],
                "providerDebug": {"version": 7, "unknownFlag": False},
            }, "protocol"),
            ("non_object", ["unexpected", {"providerDebug": {"retryAfterMs": 123}}], "protocol"),
            ("correlation_mismatch", {
                "ok": True, "move": "e3", "correlationId": "different-response-id",
                "providerDebug": {"requestStatus": "unknown"},
            }, "protocol"),
        ]
        for label, response, kind in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as root, patch(
                "quoridor_lab.opponents.socketio.Client",
                side_effect=AssertionError("Tests must not create a network client"),
            ):
                expert = RemoteExpert(data_dir=root, difficulty="easy", interval=0.05)
                fake = FakeSocket(response)
                expert.client = fake
                before = datetime.now(timezone.utc)
                try:
                    with self.assertRaises(RemoteError) as raised:
                        expert.choose(State.from_moves(["e2", "e8"]), difficulty="master")
                    after = datetime.now(timezone.utc)
                    error = raised.exception
                    self.assertEqual(error.kind, kind)
                    self.assertFalse(error.retryable)
                    diagnostics = error.diagnostics
                    self.assertEqual(diagnostics["provider_response"], fake.returned_response)
                    event, payload, options = fake.calls[0]
                    self.assertEqual(event, "ai:get_move")
                    self.assertEqual(options["namespace"], "/ai")
                    self.assertEqual(diagnostics["request"], {
                        "moves": payload["moves"], "difficulty": payload["difficulty"],
                        "correlationId": payload["correlationId"],
                    })
                    self.assertEqual(diagnostics["request"]["moves"], "e2,e8")
                    self.assertEqual(diagnostics["request"]["difficulty"], "master")
                    self.assertNotIn("deviceId", diagnostics["request"])
                    self.assertNotIn("deviceId", diagnostics)
                    observed = datetime.fromisoformat(diagnostics["observed_at"])
                    self.assertEqual(observed.utcoffset(), timedelta(0))
                    self.assertLessEqual(before, observed)
                    self.assertLessEqual(observed, after)
                    self.assertGreaterEqual(diagnostics["latency_seconds"], 0)
                    if label == "illegal_move":
                        self.assertEqual(diagnostics["move"], "e2")
                        self.assertEqual(diagnostics["rule_error"], "illegal-move")
                        self.assertTrue(diagnostics["correlation_verified"])
                    self.assertEqual(len(fake.calls), 1)
                    self.assertEqual(expert.network_requests, 1)
                    self.assertFalse(expert.cache)
                    self.assertFalse(expert.cache_path.exists())
                finally:
                    expert.close()

    def test_rate_limit_still_stops_collection_without_replacement(self):
        with tempfile.TemporaryDirectory() as root, patch(
            "quoridor_lab.opponents.socketio.Client",
            side_effect=AssertionError("Tests must not create a network client"),
        ):
            gate = RequestGate(interval=0.05)
            expert = RemoteExpert(data_dir=Path(root) / "expert", interval=0.05,
                                  request_gate=gate, cache=False)
            fake = FakeSocket({"ok": False, "error": "rate_limited", "retryAfterMs": 300000})
            expert.client = fake
            output = Path(root) / "collection"
            try:
                result = collect(
                    expert, output=output, games=2, remote_sides=("master", "master"),
                    concurrent_games=1, request_lanes=(expert,), games_per_connection=1,
                    illegal_move_retries=2, defer_illegal_responses=True,
                    retry_transient=True, drain_pending=True, abandon_after_retries=4,
                    completed_only=True, progress=lambda _: None,
                )
                self.assertEqual(result["error"]["kind"], "rate_limited")
                self.assertTrue(gate.stopped.is_set())
                self.assertEqual(result["finished_games"], 0)
                self.assertEqual(result["abandoned_games"], 0)
                self.assertEqual(len(fake.calls), 1)
                self.assertEqual(len(list((output / "pending").glob("*.json"))), 1)
                self.assertFalse((output / "episodes.jsonl").exists())
                self.assertFalse((output / "abandoned-games.jsonl").exists())
                self.assertFalse((output / "transitions.jsonl").exists())
            finally:
                expert.close()


if __name__ == "__main__":
    unittest.main()
