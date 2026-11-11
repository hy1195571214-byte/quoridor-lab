"""Local agent API contracts; no external requests are made."""
import tempfile
import unittest
from fastapi.testclient import TestClient
from barricade_lab.api import create_app
from barricade_lab.opponents import Decision, RemoteExpert
from barricade_lab.rules import Action


class FixedPolicy:
    def __init__(self, result):
        self.result = result
        self.history = None

    def choose(self, state):
        self.history = state.history
        return self.result


class ContradictoryDecision:
    action = Action.parse('e2')

    def to_dict(self):
        return {'action_id': 0, 'notation': 'a9', 'source': 'custom'}


class AgentApiTests(unittest.TestCase):
    def test_registry_and_validation(self):
        policy = FixedPolicy(Decision(Action.parse('e8'), 'test'))
        illegal = FixedPolicy(Decision(Action.parse('e1'), 'test'))
        malformed = FixedPolicy({'action_id': 13})
        with tempfile.TemporaryDirectory() as folder:
            app = create_app(RemoteExpert(data_dir=folder), agents={'custom': policy, 'bad': illegal, 'malformed': malformed})
            with TestClient(app) as client:
                entries = {entry['id']: entry for entry in client.get('/agents').json()['agents']}
                self.assertTrue(entries['custom']['available'])
                self.assertEqual(entries['shortest']['kind'], 'baseline')
                self.assertFalse(entries['dl']['available'])
                self.assertEqual(client.post('/agents/dl/move', json={'moves': []}).status_code, 503)
                self.assertEqual(client.post('/agents/missing/move', json={'moves': []}).status_code, 404)
                response = client.post('/agents/custom/move', json={'moves': ['e2']})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['notation'], 'e8')
                self.assertEqual(policy.history, ('e2',))
                self.assertEqual(client.post('/agents/custom/move', json={'moves': ['e3']}).status_code, 400)
                self.assertEqual(client.post('/agents/bad/move', json={'moves': []}).status_code, 502)
                self.assertEqual(client.post('/agents/malformed/move', json={'moves': []}).status_code, 502)
                obs = client.post('/state', json={'moves': ['e2']}).json()
                self.assertEqual((len(obs['observation']), len(obs['observation'][0]), len(obs['observation'][0][0])), (8, 9, 9))
                self.assertEqual(len(obs['action_mask']), 209)
                self.assertEqual(obs['state']['moves'], ['e2'])

    def test_overridden_baseline_is_custom(self):
        with tempfile.TemporaryDirectory() as folder:
            policy = FixedPolicy(Decision(Action.parse('e2'), 'test'))
            with TestClient(create_app(RemoteExpert(data_dir=folder), agents={'shortest': policy})) as client:
                entries = {entry['id']: entry for entry in client.get('/agents').json()['agents']}
                self.assertEqual(entries['shortest']['kind'], 'custom')
                self.assertEqual(client.post('/agents/shortest/move', json={'moves': []}).json()['source'], 'test')

    def test_response_uses_validated_action(self):
        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(RemoteExpert(data_dir=folder), agents={'custom': FixedPolicy(ContradictoryDecision())})) as client:
                result = client.post('/agents/custom/move', json={'moves': []}).json()
                self.assertEqual(result['action_id'], Action.parse('e2').id)
                self.assertEqual(result['notation'], 'e2')
