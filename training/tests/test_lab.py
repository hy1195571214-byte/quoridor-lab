import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from gymnasium.utils.passive_env_checker import env_reset_passive_checker, env_step_passive_checker
from fastapi.testclient import TestClient

from barricade_lab.rules import Action, State
from barricade_lab.opponents import RemoteExpert, RemoteError, LocalPolicy
from barricade_lab.env import BarricadeEnv
from barricade_lab.collect import collect
from barricade_lab.api import create_app

REAL_GAME = "e2,e8,e3,e7,e4,e6,e5,e4,hd3,hd5,f5,hf5,ve3,hh5,vc4,vb6,f4,vf4,f3,hf2,g3,vg2,g4,vb8,g5,e5,h5,f5,h4,f4,h3,f3,h2,g3,h1,g4,hg1,h4,vh1,i4,g1,i3,f1,i2,e1,i1".split(",")


class RuleTests(unittest.TestCase):
    def test_action_id_and_notation_bijection(self):
        for action_id in range(209):
            a = Action.from_id(action_id)
            self.assertEqual(Action.parse(a.notation).id, action_id)
        self.assertEqual(Action.parse("h4v"), Action.parse("vh4"))
        for bad in ["j2", "e0", "hi1", "e22", "", "E2"]:
            with self.assertRaises(ValueError):
                Action.parse(bad)
        for bad in [-1, 209, True, 2.3]:
            with self.assertRaises(ValueError):
                Action.from_id(bad)

    def test_opening_and_immutability(self):
        s = State()
        self.assertEqual(s.action_mask().sum(), 131)
        self.assertEqual(s.legal_destinations(), {(7, 4), (8, 3), (8, 5)})
        with self.assertRaises(ValueError):
            s.apply("e3")
        self.assertEqual(s.history, ())
        self.assertEqual(s.apply("e2").pawns[0], (7, 4))

    def test_jump_and_blocked_diagonal(self):
        s = State(pawns=((4, 4), (3, 4)))
        self.assertIn((2, 4), s.legal_destinations())
        self.assertNotIn((3, 3), s.legal_destinations())
        s = State(pawns=s.pawns, walls=(("h", 2, 4),), remaining=(9, 10))
        self.assertIn((3, 3), s.legal_destinations())
        self.assertIn((3, 5), s.legal_destinations())

    def test_crossing_overlap_and_seam(self):
        s = State(walls=(("v", 1, 2), ("v", 3, 2)), remaining=(8, 10))
        self.assertIsNone(s.error(Action("h", 2, 2)))
        self.assertEqual(s.error(Action("h", 1, 2)), "crossing")
        self.assertEqual(s.error(Action("v", 2, 2)), "overlap")

    def test_last_route_and_zero_walls(self):
        s = State(pawns=((8, 0), (0, 4)), walls=(("v", 7, 1),), remaining=(9, 10))
        self.assertEqual(s.error(Action("h", 7, 0)), "blocks-path")
        s = State(remaining=(0, 10))
        self.assertEqual(len(s.legal_actions()), 3)

    def test_recorded_expert_game(self):
        s = State.from_moves(REAL_GAME)
        self.assertEqual(s.winner, 2)
        self.assertEqual(s.pawns, ((8, 4), (8, 8)))
        self.assertEqual(s.remaining, (5, 2))
        self.assertFalse(s.action_mask().any())
        with self.assertRaises(ValueError):
            s.apply("i2")
        self.assertEqual(s.history, tuple(REAL_GAME))

    def test_environment_contract(self):
        env = BarricadeEnv(LocalPolicy("shortest"), max_plies=32)
        # The generic checker samples unmasked illegal actions; this environment
        # intentionally rejects those instead of silently substituting a move.
        obs, _ = env_reset_passive_checker(env, seed=4)
        env_step_passive_checker(env, int(np.flatnonzero(obs["action_mask"])[0]))
        for side in (1, 2):
            env.learner_player = side
            obs, _ = env.reset(seed=4)
            self.assertEqual(env.state.turn, side)
            self.assertTrue(env.observation_space.contains(obs))
            while True:
                action = LocalPolicy().choose(env.state).action.id
                obs, reward, terminated, truncated, _ = env.step(action)
                self.assertTrue(env.observation_space.contains(obs))
                if terminated or truncated:
                    self.assertFalse(obs["action_mask"].any())
                    self.assertIn(reward, (-1, 0, 1))
                    break
        env.close()


class FakeSocket:
    connected = True

    def __init__(self, response=None):
        self.response = response or {"ok": True, "move": "e8"}
        self.calls = []

    def call(self, event, payload, **kwargs):
        self.calls.append((event, payload, kwargs))
        return self.response

    def disconnect(self):
        self.connected = False


class AdapterTests(unittest.TestCase):
    def test_request_and_cache(self):
        with tempfile.TemporaryDirectory() as root:
            expert = RemoteExpert(data_dir=root, interval=.25)
            fake = FakeSocket()
            expert.client = fake
            state = State.from_moves(["e2"])
            first, second = expert.choose(state), expert.choose(state)
            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(fake.calls[0][0], "ai:get_move")
            self.assertEqual(fake.calls[0][1]["moves"], "e2")
            self.assertEqual(fake.calls[0][1]["difficulty"], "expert")
            self.assertEqual(RemoteExpert(data_dir=root).device_id, expert.device_id)
            expert.close()

    def test_rate_limit_and_illegal_response_do_not_substitute(self):
        for response, kind in [({"error": "rate_limited"}, "rate_limited"),
                               ({"ok": True, "move": "e1"}, "illegal_response")]:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as root:
                expert = RemoteExpert(data_dir=root)
                expert.client = FakeSocket(response)
                with self.assertRaises(RemoteError) as raised:
                    expert.choose(State.from_moves(["e2"]))
                self.assertEqual(raised.exception.kind, kind)
                self.assertFalse(expert.cache)
                expert.close()

    def test_local_api_is_usable_and_validates_actions(self):
        with tempfile.TemporaryDirectory() as root:
            with TestClient(create_app(RemoteExpert(data_dir=root))) as api:
                self.assertEqual(api.get("/health").json()["action_count"], 209)
                self.assertEqual(api.post("/state", json={"moves": ["e3"]}).status_code, 400)
                new = api.post("/games", json={"opponent": "shortest"}).json()
                game_id = new["game_id"]
                self.assertEqual(api.post(f"/games/{game_id}/step", json={"notation": "e3"}).status_code, 400)
                result = api.post(f"/games/{game_id}/step", json={"notation": "e2"}).json()
                self.assertEqual(len(result["state"]["moves"]), 2)
                self.assertEqual(result["state"]["turn"], 1)

    def test_collection_checkpoint_resume(self):
        class Interrupting:
            difficulty = "expert"
            def __init__(self):
                self.calls = 0
                self.fail = True
                self.local = LocalPolicy()
            def choose(self, state):
                self.calls += 1
                if self.fail and self.calls == 2:
                    raise RemoteError("busy", kind="rate_limited")
                return self.local.choose(state)
        teacher = Interrupting()
        with tempfile.TemporaryDirectory() as root:
            result = collect(teacher, output=root, games=1, expert_side="2", progress=lambda _: None)
            self.assertEqual(result["error"]["kind"], "rate_limited")
            self.assertEqual(result["finished_games"], 0)
            teacher.fail = False
            result = collect(teacher, output=root, games=1, resume=True, progress=lambda _: None)
            self.assertEqual(result["finished_games"], 1)
            self.assertEqual(result["episodes"][0]["status"], "completed")
            rows = [json.loads(line) for line in (Path(root) / "transitions.jsonl").read_text().splitlines()]
            self.assertEqual([row["ply"] for row in rows], list(range(len(rows))))
            self.assertFalse(list((Path(root) / "pending").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
