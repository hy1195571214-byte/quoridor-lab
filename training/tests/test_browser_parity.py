"""Cross-language regression for the browser rules; no network or training."""
import json
import random
import subprocess
from pathlib import Path
import unittest

from barricade_lab.rules import Action, State

ROOT = Path(__file__).resolve().parents[2]


def browser_state(s):
    return {"pawns": s.pawns, "walls": s.walls, "remaining": s.remaining,
            "turn": s.turn, "winner": s.winner, "history": s.history}


class BrowserParity(unittest.TestCase):
    def test_reachable_states(self):
        rng = random.Random(8472)
        states = [State(), State(pawns=((4, 4), (3, 4))),
                  State(pawns=((4, 4), (3, 4)), walls=(("h", 2, 4),)),
                  State(pawns=((8, 0), (0, 4)), walls=(("v", 7, 1),), remaining=(9, 10)),
                  State(pawns=((0, 0), (8, 8)), walls=(("h", 0, 0), ("v", 0, 1), ("h", 1, 0)))]
        for _ in range(5):
            state = State()
            for ply in range(36):
                actions = state.legal_actions()
                if not actions:
                    break
                # Encourage walls as well as pawn movement.
                moves = [a for a in actions if a.kind == "move"]
                walls = [a for a in actions if a.kind != "move"]
                state = state.apply(rng.choice(walls if walls and rng.random() < .5 else moves))
                if ply % 5 == 0 or state.winner:
                    states.append(state)
                if state.winner:
                    break
        process = subprocess.run(["node", "--experimental-default-type=module", str(ROOT / "frontend/tests/rules-parity.mjs")],
                                 input=json.dumps([browser_state(s) for s in states]), text=True,
                                 capture_output=True, check=True)
        results = json.loads(process.stdout)
        for state, result in zip(states, results):
            with self.subTest(history=state.history):
                self.assertEqual(result['dest'], [list(p) for p in sorted(state.legal_destinations())])
                self.assertEqual(result['distances'], [state.distance(1), state.distance(2)])
                self.assertEqual(result['errors'], [state.error(Action.from_id(i)) for i in range(209)])
                self.assertEqual(result['mask'], state.action_mask().tolist())
                self.assertEqual(result['obs']['action_mask'], state.action_mask().tolist())
                # float32 planes can differ from JS doubles by a few ulps.
                import numpy as np
                np.testing.assert_allclose(result['obs']['board'], state.observation()['board'], atol=1e-7)
                self.assertEqual(result['next'], json.loads(json.dumps([browser_state(state.apply(a)) for a in state.legal_actions()[:4]])))
