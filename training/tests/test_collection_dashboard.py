from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse

from quoridor_lab.collect import collect
from quoridor_lab.dashboard import replay_link, rows, sample_stats
from quoridor_lab.opponents import Decision, LocalPolicy, RemoteError, RemoteExpert
from quoridor_lab.rules import State


class FakeExpert:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.policy = LocalPolicy(seed=0)

    def choose(self, state, *, difficulty):
        self.calls.append((state.turn, difficulty))
        if self.fail and len(self.calls) == 2:
            raise RemoteError("busy", kind="rate_limited")
        return Decision(self.policy.choose(state).action, f"barricade:{difficulty}",
                        metadata={"difficulty": difficulty})


class DashboardCollectionTests(unittest.TestCase):
    def test_concurrent_games_take_turns_and_resume_after_shared_error(self):
        expert = FakeExpert()
        expert.fail = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"output": root, "remote_sides": ("expert", "easy"),
                      "games": 3, "concurrent_games": 3, "max_plies": 4,
                      "progress": lambda _: None}
            paused = collect(expert, **config)
            self.assertEqual(paused["error"]["kind"], "rate_limited")
            self.assertEqual(len(list((root / "pending").glob("*.json"))), 3)
            self.assertEqual(expert.calls, [(1, "expert"), (1, "expert")])
            expert.fail = False
            result = collect(expert, resume=True, **config)
            self.assertIsNone(result["error"])
            self.assertEqual({game["game_number"] for game in result["episodes"]}, {1, 2, 3})
            samples = list(rows(root / "transitions.jsonl"))
            self.assertEqual(len({row["episode_id"] for row in samples if row["ply"] == 0}), 3)
            for game in result["episodes"]:
                trace = [row for row in samples if row["episode_id"] == game["episode_id"]]
                self.assertEqual([row["ply"] for row in trace], [0, 1, 2, 3])
                for row in trace:
                    self.assertEqual(row["history"], game["moves"][:row["ply"]])
                    self.assertEqual(row["notation"], game["moves"][row["ply"]])
                self.assertEqual(State.from_moves(game["moves"]).to_dict(), game["final"])
            self.assertFalse(list((root / "pending").glob("*.json")))

    def test_dual_remote_opening_resume_and_replay(self):
        expert = FakeExpert()
        expert.fail = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"output": root, "remote_sides": ("master", "easy"), "max_plies": 4,
                      "opening": ("e2",), "progress": lambda _: None}
            run = collect(expert, **config)
            self.assertEqual(run["error"]["kind"], "rate_limited")
            self.assertEqual(expert.calls, [(2, "easy"), (1, "master")])
            expert.fail = False
            run = collect(expert, resume=True, **config)
            [game] = run["episodes"]
            samples = list(rows(root / "transitions.jsonl"))
            self.assertEqual([row["ply"] for row in samples], [1, 2, 3])
            self.assertTrue(all(row["is_teacher"] for row in samples))
            self.assertEqual([row["source"] for row in samples],
                             ["barricade:easy", "barricade:master", "barricade:easy"])
            self.assertEqual(State.from_moves(game["moves"]).to_dict(), game["final"])
            self.assertEqual(game["status"], "truncated")
            query = parse_qs(urlparse(replay_link("http://127.0.0.1:4173", game)).query)
            self.assertEqual(query["moves"], [",".join(game["moves"])])
            self.assertEqual(query["ai"], ["master,easy"])

    def test_duplicate_demonstrations_use_position_difficulty_and_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = collect(FakeExpert(), output=root, games=2, max_plies=2,
                          remote_sides=("expert", "expert"), concurrent_games=2,
                          progress=lambda _: None)
            trace = list(rows(root / "transitions.jsonl"))
            self.assertEqual([row["ply"] for row in trace], [0, 0, 1, 1])
            self.assertEqual([row["episode_id"] for row in trace[:2]],
                             [row["episode_id"] for row in trace[2:]])
            stats = sample_stats(root / "transitions.jsonl", run["episodes"])
            self.assertEqual(stats["samples"], 4)
            self.assertEqual(stats["repeated_positions"], 2)
            self.assertEqual(stats["repeated_demonstrations"], 2)
            self.assertEqual(stats["repeated_request_histories"], 2)
            self.assertEqual(stats["changed_request_histories"], 0)

    def test_remote_cache_distinguishes_difficulty_on_one_connection(self):
        class Socket:
            connected = True

            def call(self, event, payload, **kwargs):
                return {"ok": True, "move": "e8"}

        with tempfile.TemporaryDirectory() as tmp:
            expert = RemoteExpert(data_dir=tmp, interval=.25)
            expert.client = Socket()
            state = State.from_moves(["e2"])
            self.assertFalse(expert.choose(state, difficulty="easy").cached)
            self.assertFalse(expert.choose(state, difficulty="master").cached)
            self.assertEqual(expert.choose(state, difficulty="easy").source, "barricade:easy")
            self.assertEqual(expert.network_requests, 2)
            self.assertEqual(expert.cache_hits, 1)
