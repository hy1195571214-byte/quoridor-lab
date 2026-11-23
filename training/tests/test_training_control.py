"""Opening the UI is read-only; only an explicit valid POST can launch a trainer."""
import json
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from quoridor_lab.training_control import TrainingController, install_control_routes
from quoridor_lab.training_monitor import TrainingMonitor, install_training_routes


class TrainingControlTests(unittest.TestCase):
    def setup_controller(self, root, factory):
        dataset = root / 'dataset'
        dataset.mkdir()
        (dataset / 'manifest.json').write_text('{}')
        (dataset / 'episodes.jsonl').write_text('{}\n')
        selection = root / 'active.json'
        selection.write_text(json.dumps({'version': 'fixture', 'manifest': str(dataset / 'manifest.json'),
            'corpus': str(dataset / 'episodes.jsonl'), 'episodes': 30,
            'splits': {'train': 24, 'validation': 3, 'test': 3}}))
        controller = TrainingController(root / 'runs', selection=selection, process_factory=factory)
        controller.devices = lambda: {'mps': True, 'cuda': False, 'cpu': True}
        app = FastAPI()
        install_control_routes(app, controller)
        install_training_routes(app, TrainingMonitor(root / 'runs', dataset / 'episodes.jsonl'))
        return controller, TestClient(app)

    def test_read_only_configuration_validation_and_single_explicit_launch(self):
        calls = []
        class FakeProcess:
            pid = 12345
            def poll(self): return None
        def factory(command, **kwargs):
            calls.append((command, kwargs))
            return FakeProcess()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            controller, client = self.setup_controller(root, factory)
            with client:
                for _ in range(2):
                    config = client.get('/training/config').json()
                    self.assertEqual(config['suggested_name'], '棋谱预训练 · 第1轮')
                    self.assertFalse(config['active'])
                    self.assertEqual(client.get('/training/runs').json()['runs'], [])
                self.assertFalse((root / 'runs').exists())
                self.assertEqual(calls, [])
                self.assertEqual(client.post('/training/start', json={'epochs': 0}).status_code, 422)
                self.assertEqual(client.post('/training/start', json={'architecture': 'arbitrary-command'}).status_code, 422)
                self.assertEqual(client.post('/training/start', json={'device': 'cuda'}).status_code, 400)
                self.assertEqual(calls, [])
                response = client.post('/training/start', json={'name': '棋谱预训练 · 第1轮', 'epochs': 3})
                self.assertEqual(response.status_code, 202, response.text)
                self.assertEqual(response.json()['id'], 'pretrain-001')
                self.assertEqual(len(calls), 1)
                command, kwargs = calls[0]
                self.assertIn('quoridor_lab.supervised', command)
                self.assertEqual(command[command.index('--epochs') + 1], '3')
                self.assertEqual(command[command.index('--name') + 1], '棋谱预训练 · 第1轮')
                self.assertNotIn('shell', kwargs)
                run = client.get('/training/runs/pretrain-001').json()
                self.assertEqual(run['display_name'], '棋谱预训练 · 第1轮')
                self.assertNotIn('command', run)
                self.assertEqual(client.post('/training/start', json={}).status_code, 409)
                self.assertEqual(len(calls), 1)

    def test_launch_failure_is_visible_without_a_fake_running_job(self):
        def fail(*args, **kwargs): raise OSError('fixture launch failure')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, client = self.setup_controller(root, fail)
            with client:
                self.assertEqual(client.post('/training/start', json={}).status_code, 503)
                run = client.get('/training/runs/pretrain-001').json()
                self.assertEqual(run['status'], 'failed')
                self.assertFalse(client.get('/training/config').json()['active'])

    def test_downloaded_dataset_paths_resolve_relative_to_selection_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, client = self.setup_controller(root, lambda *args, **kwargs: self.fail('Must not launch on GET'))
            selection = root / 'active.json'
            active = json.loads(selection.read_text())
            active.update(manifest='dataset/manifest.json', corpus='dataset/episodes.jsonl')
            selection.write_text(json.dumps(active))
            with client:
                response = client.get('/training/config')
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['dataset']['manifest'], str((root / 'dataset/manifest.json').resolve()))

    def test_large_corpus_pagination_and_indexed_detail_keep_test_private(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = root / 'episodes.jsonl'
            entries = []
            with corpus.open('wb') as output:
                for uid, split in [('a', 'train'), ('b', 'train'), ('v', 'validation'), ('t', 'test')]:
                    offset = output.tell()
                    output.write((json.dumps({'split': split, 'dataset_source': 'fixture',
                        'episode': {'episode_id': uid, 'moves': ['e2']}}) + '\n').encode())
                    entries.append({'episode_id': uid, 'source': 'fixture', 'split': split,
                                    'plies': 1, 'winner': None, 'file_offset': offset})
            (root / 'manifest.json').write_text(json.dumps({'episodes': entries}))
            monitor = TrainingMonitor(root / 'runs', corpus)
            app = FastAPI()
            install_training_routes(app, monitor)
            with TestClient(app) as client:
                result = client.get('/training/episodes?limit=1&offset=1').json()
                self.assertEqual(result['total'], 2)
                self.assertEqual(result['episodes'][0]['episode_id'], 'b')
                self.assertNotIn('moves', monitor.episodes()['b'])
                self.assertEqual(client.get('/training/episodes/b').json()['moves'], ['e2'])
                self.assertEqual(client.get('/training/episodes?split=test').status_code, 400)
                self.assertEqual(client.get('/training/episodes/t').status_code, 404)


if __name__ == '__main__':
    unittest.main()
