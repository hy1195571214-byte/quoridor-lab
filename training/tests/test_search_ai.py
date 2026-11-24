import math
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from quoridor_lab.rules import Action, State
import tempfile

from fastapi.testclient import TestClient

from quoridor_lab.api import create_app
from quoridor_lab.opponents import RemoteExpert
from quoridor_lab.search import (
    DIFFICULTY_PRESETS, WIN_SCORE, SearchAgent, SearchStats, TimeBudgetExceeded,
    _choose_minimax, _search, _search_root,
)


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

    def test_self_play_short_horizon_only_legal(self):
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

        for distances in ((999, 8), (8, 999), (999, 999)):
            with self.subTest(distances=distances), patch.object(
                State, "distance", lambda state, player: distances[player - 1]
            ):
                m = _distance_margin(State(), 1, 2)
                self.assertTrue(math.isfinite(m))
                self.assertEqual(m > 0, distances == (8, 999))
                self.assertEqual(m < 0, distances == (999, 8))

    def test_leaf_score_keeps_root_perspective_for_either_turn(self):
        for turn in (1, 2):
            state = State(pawns=((6, 4), (1, 4)), remaining=(0, 0), turn=turn)
            for root in (1, 2):
                with self.subTest(turn=turn, root=root):
                    score = _search(state, 0, -math.inf, math.inf, root,
                                    DIFFICULTY_PRESETS["medium"], SearchStats(), None)
                    self.assertEqual(score, state.distance(3 - root) - state.distance(root))

    def test_depth_one_advances_toward_goal_for_either_side(self):
        for player in (1, 2):
            state = State(remaining=(0, 0), turn=player)
            action, _, _ = _search_root(state, 1, DIFFICULTY_PRESETS["medium"], None)
            self.assertLess(state.apply(action).distance(player), state.distance(player))

    def test_alpha_beta_matches_exhaustive_search_at_odd_and_even_depths(self):
        def exhaustive(state, depth, root):
            if state.winner is not None:
                return WIN_SCORE if state.winner == root else -WIN_SCORE
            if depth == 0:
                return state.distance(3 - root) - state.distance(root)
            values = [exhaustive(state.apply(a), depth - 1, root) for a in state.legal_actions()]
            return (max if state.turn == root else min)(values)

        for player in (1, 2):
            state = State(pawns=((5, 4), (3, 5)), remaining=(0, 0), turn=player)
            for depth in (1, 2, 3):
                with self.subTest(player=player, depth=depth):
                    action, _, score = _search_root(state, depth, DIFFICULTY_PRESETS["hard"], None)
                    self.assertEqual(score, exhaustive(state, depth, player))
                    self.assertEqual(score, exhaustive(state.apply(action), depth - 1, player))

    def test_both_sides_take_an_immediate_win(self):
        for player in (1, 2):
            state = State(pawns=((1, 4), (7, 2)), remaining=(0, 0), turn=player)
            decision = SearchAgent("medium").choose(state)
            self.assertEqual(state.apply(decision.action).winner, player)

    def test_timeout_retains_last_completed_iteration(self):
        state = State()
        config = replace(DIFFICULTY_PRESETS["hard"], min_depth=1, max_depth=2)
        action = Action.parse("e2")
        with patch("quoridor_lab.search._search_root", side_effect=[
            (action, SearchStats(completed_depth=1), 1.0), TimeBudgetExceeded()
        ]):
            chosen, stats, score = _choose_minimax(state, config)
        self.assertEqual(chosen, action)
        self.assertEqual(stats.completed_depth, 1)
        self.assertEqual(score, 1.0)

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
