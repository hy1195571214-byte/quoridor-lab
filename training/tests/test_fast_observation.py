"""Differential checks against the strict reference, including last-path walls."""
import json
from pathlib import Path
import random
import unittest

import numpy as np

from quoridor_lab.fast_observation import training_observation
from quoridor_lab.rules import State


class FastObservationTests(unittest.TestCase):
    def assert_same(self, state):
        expected, actual = state.observation(), training_observation(state)
        np.testing.assert_array_equal(actual['board'], expected['board'])
        np.testing.assert_array_equal(actual['action_mask'], expected['action_mask'])

    def test_path_blocking_crossing_jumps_edges_and_terminal(self):
        states = [State(), State(remaining=(0, 10)), State(remaining=(10, 0), turn=2),
                  State(pawns=((4, 4), (3, 4))),
                  State(pawns=((4, 4), (3, 4)), walls=(("h", 2, 4),)),
                  State(walls=(("v", 1, 2), ("v", 3, 2)), remaining=(8, 10)),
                  State(pawns=((8, 0), (0, 4)), walls=(("v", 7, 1),), remaining=(9, 10)),
                  State(pawns=((0, 0), (8, 8)), walls=(("h", 0, 0), ("v", 0, 1), ("h", 1, 0))),
                  State(pawns=((0, 4), (1, 4)), winner=1)]
        for state in states:
            with self.subTest(state=state): self.assert_same(state)

    def test_random_reachable_states(self):
        rng = random.Random(7031)
        for game in range(12):
            state = State()
            for ply in range(60):
                with self.subTest(game=game, ply=ply): self.assert_same(state)
                actions = state.legal_actions()
                if not actions:
                    break
                moves = [action for action in actions if action.kind == 'move']
                walls = [action for action in actions if action.kind != 'move']
                state = state.apply(rng.choice(walls if walls and rng.random() < .55 else moves))

    def test_recorded_training_games(self):
        corpus = Path(__file__).resolve().parents[1] / 'data/datasets/round1-2500/episodes.jsonl'
        checked = 0
        with corpus.open() as source:
            for line in source:
                wrapper = json.loads(line)
                if wrapper['split'] != 'train': continue
                state = State()
                for move in wrapper['episode']['moves']:
                    self.assert_same(state)
                    state = state.apply(move)
                checked += 1
                if checked == 12: break
        self.assertEqual(checked, 12)


if __name__ == '__main__':
    unittest.main()
