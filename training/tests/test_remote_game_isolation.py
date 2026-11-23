"""Exercise real request serialization and collection with a fake wire client."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from quoridor_lab.collect import collect
from quoridor_lab.opponents import RemoteExpert, RequestGate
from quoridor_lab.rules import State


GAMES = [
    # Walls persist across restoration; the last move is a diagonal goal move
    # around the blue pawn against the board edge.
    "e2,ha1,e3,ha2,e4,ha3,e5,ha4,e6,ha5,e7,ha6,e8,ha7,d9".split(","),
    "d1,e8,d2,e7,d3,e6,d4,e5,d5,e4,d6,e3,d7,e2,d8,f2,d9".split(","),
]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def save_checkpoint(root, index, moves):
    pending = root / "pending"
    pending.mkdir(exist_ok=True)
    record = {
        "episode_id": f"game-{index}", "game_number": index + 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": {"expert_side": None, "policy": "shortest", "seed": index,
                   "max_plies": 32, "opponent": "barricade",
                   "remote_sides": ["master", "master"]},
        "moves": moves, "random_state": None,
    }
    (pending / f"game-{index}.json").write_text(json.dumps(record))
    state = State()
    with (root / "transitions.jsonl").open("a") as file:
        for ply, move in enumerate(moves):
            after = state.apply(move)
            file.write(json.dumps({"episode_id": record["episode_id"], "ply": ply,
                                   "history": list(state.history), "player": state.turn,
                                   "notation": move, "after": after.to_dict()}) + "\n")
            state = after


class WireClient:
    connected = True

    def __init__(self, test, calls, barrier=None, fail=None):
        self.test, self.calls, self.barrier, self.fail = test, calls, barrier, fail
        self.first = True

    def call(self, event, payload, **options):
        self.test.assertEqual(event, "ai:get_move")
        self.test.assertEqual(options["namespace"], "/ai")
        self.test.assertEqual(payload["difficulty"], "master")
        history = payload["moves"].split(",") if payload["moves"] else []
        matches = [game for game in GAMES if game[:len(history)] == history]
        self.test.assertTrue(matches, f"Unexpected or incomplete wire history: {history}")
        self.calls.append(dict(payload))
        if self.first and self.barrier:
            self.first = False
            self.barrier.wait(timeout=5)
        if self.fail:
            move = self.fail(history)
        else:
            move = None
        return {"ok": True, "move": move or matches[0][len(history)],
                "correlationId": payload["correlationId"], "served_moves": payload["moves"]}

    def disconnect(self):
        self.connected = False

    def connect(self, *args, **kwargs):
        self.connected = True

    def transport(self):
        return 'websocket'


class RemoteGameIsolationTests(unittest.TestCase):
    def run_games(self, root, clients, *, resume=True, games=2):
        return collect(clients[0], output=root, games=games, resume=resume,
                       remote_sides=("master", "master"), max_plies=32,
                       concurrent_games=games, request_lanes=clients,
                       games_per_connection=2 if len(clients) == 1 else 1,
                       illegal_move_retries=2, progress=lambda _: None)

    def experts(self, root, count, calls, stack, barrier=None, fail=None):
        gate = RequestGate(interval=0)
        experts = []
        for _ in range(count):
            expert = RemoteExpert(data_dir=root / "identity", cache=False,
                                  request_gate=gate, interval=0.05)
            expert.client = WireClient(self, calls, barrier, fail)
            stack.callback(expert.close)
            experts.append(expert)
        return experts

    def assert_trace(self, root, result):
        self.assertIsNone(result["error"])
        self.assertEqual(result["completed_games"], 2)
        trace = rows(root / "transitions.jsonl")
        self.assertEqual(len(trace), len({(r["episode_id"], r["ply"]) for r in trace}))
        for index, expected in enumerate(GAMES):
            steps = [r for r in trace if r["episode_id"] == f"game-{index}"]
            self.assertEqual([r["notation"] for r in steps], expected)
            for step in steps:
                history = expected[:step["ply"]]
                self.assertEqual(step["history"], history)
                before = State.from_moves(history)
                self.assertEqual(step["player"], before.turn)
                self.assertEqual(step["after"], before.apply(step["notation"]).to_dict())
                if "metadata" in step:
                    self.assertEqual(step["metadata"]["provider_response"]["served_moves"], ",".join(history))
            game = next(g for g in result["episodes"] if g["episode_id"] == f"game-{index}")
            self.assertEqual(game["moves"], expected)
            self.assertEqual(game["winner"], State.from_moves(expected).winner)

    def test_distinct_resumed_histories_with_shared_lane_and_parallel_lanes(self):
        for count in (1, 2):
            for prefix in (1, 6, 10):
                with self.subTest(lanes=count, prefix=prefix), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                    root = Path(tmp)
                    for index, game in enumerate(GAMES):
                        save_checkpoint(root, index, game[:prefix])
                    calls = []
                    barrier = threading.Barrier(2) if count == 2 else None
                    experts = self.experts(root, count, calls, stack, barrier)
                    with patch("quoridor_lab.opponents.socketio.Client", side_effect=AssertionError("No network")):
                        result = self.run_games(root, experts)
                    self.assert_trace(root, result)
                    self.assertEqual(len({p["correlationId"] for p in calls}), len(calls))
                    self.assertEqual({p["moves"] for p in calls}, {
                        ",".join(game[:ply]) for game in GAMES for ply in range(prefix, len(game))})

    def test_illegal_blue_e2_retries_same_history_without_committing(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            for index, game in enumerate(GAMES):
                save_checkpoint(root, index, game[:1])
            calls, failures = [], []
            def fail(history):
                if history == ["e2"] and len(failures) < 2:
                    failures.append(history)
                    return "e2"
            experts = self.experts(root, 2, calls, stack, fail=fail)
            with patch('quoridor_lab.opponents.socketio.Client',
                       side_effect=lambda **kwargs: WireClient(self,calls,fail=fail)):
                self.assert_trace(root, self.run_games(root, experts))
            self.assertEqual(len(rows(root / "rejected-moves.jsonl")), 2)
            self.assertEqual(sum(p["moves"] == "e2" for p in calls), 3)

    def test_normal_red_first_e2_is_committed_and_transport_resume_keeps_prefix(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root, calls = Path(tmp), []
            def fail(history):
                if history == ["e2"]:
                    raise TimeoutError("Simulated interruption after committed first move")
            experts = self.experts(root, 1, calls, stack, fail=fail)
            first = self.run_games(root, experts, resume=False, games=1)
            self.assertEqual(first["error"]["kind"], "transport")
            [checkpoint] = list((root / "pending").glob("*.json"))
            self.assertEqual(json.loads(checkpoint.read_text())["moves"], ["e2"])
            self.assertEqual([r["notation"] for r in rows(root / "transitions.jsonl")], ["e2"])
            calls.clear()
            resumed = self.experts(root, 1, calls, stack)
            result = self.run_games(root, resumed, games=1)
            self.assertIsNone(result["error"])
            self.assertEqual(calls[0]["moves"], "e2")
            self.assertEqual(result["episodes"][0]["moves"], GAMES[0])
            self.assertEqual([r["notation"] for r in rows(root / "transitions.jsonl")], GAMES[0])


if __name__ == "__main__":
    unittest.main()
