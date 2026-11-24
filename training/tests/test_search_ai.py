import math
import time
import unittest

from quoridor_lab.rules import Action, State
import tempfile

from fastapi.testclient import TestClient

from quoridor_lab.api import create_app
from quoridor_lab.opponents import RemoteExpert
from quoridor_lab.search import DIFFICULTY_PRESETS, SearchAgent


class SearchAgentTests(unittest.TestCase):
    def test_choose_only_legal(self):
        state = State()
        for level in ("easy", "medium", "hard"):
            agent = SearchAgent(level, seed=1)
            for _ in range(5):
                decision = agent.choose(state)
                self.assertIsNone(state.error(decision.action))
                state = state.apply(decision.action)
                if state.winner is not None:
                    break

    def test_all_difficulties_constructible(self):
        for name in ("easy", "medium", "hard"):
            agent = SearchAgent(name)
            self.assertEqual(agent.difficulty, name)
            self.assertIn(name, DIFFICULTY_PRESETS)

    def test_hard_beats_random_policy_short_horizon(self):
        """Smoke test: hard search should finish a self-play line without illegal moves."""
        state = State()
        agents = {1: SearchAgent("hard", seed=0), 2: SearchAgent("medium", seed=1)}
        illegal = 0
        for _ in range(40):
            if state.winner is not None:
                break
            d = agents[state.turn].choose(state)
            if state.error(d.action):
                illegal += 1
                break
            state = state.apply(d.action)
        self.assertEqual(illegal, 0)

    def test_distance_999_is_finite(self):
        from quoridor_lab.search import _distance_margin

        state = State()
        m = _distance_margin(state, 1, 2)
        self.assertFalse(math.isnan(m))
        self.assertFalse(math.isinf(m))

    def test_decision_latency_recorded(self):
        agent = SearchAgent("easy", seed=0)
        started = time.monotonic()
        decision = agent.choose(State())
        self.assertGreaterEqual(decision.latency_seconds, 0.0)
        self.assertLess(decision.latency_seconds, time.monotonic() - started + 0.5)


    def test_search_http_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            agents = {
                "search_medium": SearchAgent("medium"),
                "search_hard": SearchAgent("hard"),
            }
            with TestClient(create_app(RemoteExpert(data_dir=folder), agents=agents)) as client:
                for agent_id in ("search_medium", "search_hard"):
                    response = client.post(f"/agents/{agent_id}/move", json={"moves": ["e2"]})
                    self.assertEqual(response.status_code, 200, agent_id)
                    body = response.json()
                    st = State.from_moves(["e2"])
                    self.assertIsNone(st.error(Action.from_id(body["action_id"])))
                    self.assertIn("notation", body)
                    self.assertTrue(body["source"].startswith("search:"))
                    self.assertGreaterEqual(body["latency_seconds"], 0.0)

    def test_easy_is_greedy_not_minimax(self):
        decision = SearchAgent("easy", seed=0).choose(State())
        self.assertEqual(decision.metadata.get("mode"), "greedy")


if __name__ == "__main__":
    unittest.main()