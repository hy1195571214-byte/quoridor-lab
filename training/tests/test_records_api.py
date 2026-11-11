"""Persistent local records and offline browser access; no network requests."""
import tempfile
import unittest
from fastapi.testclient import TestClient
from barricade_lab.api import create_app
from barricade_lab.opponents import RemoteExpert


class RecordApiTests(unittest.TestCase):
    def test_crud_validation_and_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            def app():
                return create_app(RemoteExpert(data_dir=folder))
            with TestClient(app()) as client:
                record = {'moves': ['e2', 'e8'], 'variations': [{'anchor': 0, 'moves': ['d9']}], 'result': None}
                response = client.post('/records', json=record)
                self.assertEqual(response.status_code, 201, response.text)
                key = response.json()['id']
                self.assertEqual(client.get('/records/' + key).json()['record']['moves'], ['e2', 'e8'])
                for invalid in [
                    {'moves': ['e3']},
                    {'moves': ['e2'], 'variations': [{'anchor': 0, 'moves': ['e3']}]},
                    {'moves': [], 'result': {'winner': 1, 'reason': 'goal'}},
                    {'moves': [], 'result': {'reason': 'resignation'}},
                    {'moves': [], 'variations': [{'anchor': True, 'moves': ['e2']}]},
                    {'schema_version': True},
                    {'initial': {'pawns': ['e1', 'e9'], 'walls': [10, 10], 'turn': True}},
                ]:
                    self.assertEqual(client.post('/records', json=invalid).status_code, 400)
                self.assertEqual(client.post('/records', content=b'{"result":{"winner":null,"reason":"\\ud800"}}', headers={'Content-Type': 'application/json'}).status_code, 400)
                self.assertEqual(client.put('/records/' + key, json={'moves': ['d1']}).status_code, 200)
                self.assertEqual(client.put('/records/unknown', json={'moves': []}).status_code, 404)
                nullable = {'schema_version': None, 'rules_version': None, 'initial': None,
                            'moves': None, 'variations': None, 'result': None}
                self.assertEqual(client.post('/records', json=nullable).status_code, 201)
                self.assertEqual(client.post('/records', json={'initial': {'pawns': ['e1', 'e9'], 'walls': [10, 10], 'turn': 1, 'extra': 3}}).status_code, 201)
            with TestClient(app()) as client:
                self.assertEqual(client.get('/records/' + key).json()['record']['moves'], ['d1'])
                self.assertEqual(len(client.get('/records').json()['records']), 3)
                self.assertEqual(client.delete('/records/' + key).status_code, 200)
                self.assertEqual(client.get('/records/' + key).status_code, 404)
                self.assertEqual(client.delete('/records/' + key).status_code, 404)

    def test_cors(self):
        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(RemoteExpert(data_dir=folder))) as client:
                for origin in ('null', 'http://localhost:5173', 'http://127.0.0.1:8765'):
                    response = client.options('/records', headers={'Origin': origin, 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers['access-control-allow-origin'], origin)
                response = client.options('/records', headers={'Origin': 'https://example.com', 'Access-Control-Request-Method': 'POST'})
                self.assertEqual(response.status_code, 400)
