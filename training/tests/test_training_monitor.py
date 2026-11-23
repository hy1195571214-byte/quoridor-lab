"""Checkpoint-backed legal predictions and replay corpus isolation."""
import json
import hashlib
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
try:
    import torch
except ImportError:
    torch = None

from quoridor_lab.rules import State
from quoridor_lab.training_monitor import TrainingJournal, TrainingMonitor, install_training_routes


@unittest.skipUnless(torch is not None, 'Install learning dependencies for checkpoint tests')
class TrainingMonitorTests(unittest.TestCase):
    def test_real_checkpoint_prediction_and_dataset_boundaries(self):
        from quoridor_lab.learning import network
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            corpus = root / 'episodes.jsonl'
            corpus.write_text('\n'.join(json.dumps({
                'split': split, 'dataset_source': 'fixture',
                'episode': {'episode_id': split, 'moves': ['e2', 'e8']},
            }) for split in ['train', 'validation', 'test']))
            monitor = TrainingMonitor(root / 'runs', corpus)
            app = FastAPI()
            install_training_routes(app, monitor)
            journal = TrainingJournal(root / 'runs/cnn', 'cnn32', 0, 2)
            try:
                with self.assertRaises(ValueError):
                    TrainingJournal(root / 'runs/cnn', 'cnn32', 0, 2, resume=True)
                with TestClient(app) as client:
                    self.assertEqual(client.get('/training/runs').json()['runs'][0]['id'], 'cnn')
                    self.assertEqual(client.post('/training/runs/cnn/predict', json={}).status_code, 409)
                    self.assertEqual(client.get('/training/episodes').json()['episodes'][0]['episode_id'], 'train')
                    self.assertEqual(client.get('/training/episodes?split=test').status_code, 400)
                    self.assertEqual(client.get('/training/episodes/test').status_code, 404)
                    self.assertEqual(client.get('/training/episodes/validation').json()['moves'], ['e2', 'e8'])
                    model = network('cnn32')
                    journal.checkpoint({'architecture': 'cnn32', 'state_dict': model.state_dict(),
                                        'epoch': 1, 'global_step': 4})
                    revision = client.get('/training/runs/cnn').json()['checkpoint']['revision']
                    response = client.post('/training/runs/cnn/predict', json={
                        'moves': ['e2'], 'revision': revision, 'rollout_plies': 8})
                    self.assertEqual(response.status_code, 200, response.text)
                    prediction = response.json()
                    self.assertEqual(prediction['checkpoint_epoch'], 1)
                    self.assertEqual(prediction['checkpoint_step'], 4)
                    self.assertEqual(prediction['revision'], revision)
                    self.assertEqual(len(prediction['predicted_moves']), 8)
                    State.from_moves(['e2'] + prediction['predicted_moves'])
                    for candidate in prediction['top_actions']:
                        State.from_moves(['e2', candidate['notation']])
                        self.assertGreaterEqual(candidate['probability'], 0)
                    self.assertEqual(client.post('/training/runs/cnn/predict', json={'moves': ['e3']}).status_code, 400)
                    self.assertEqual(client.post('/training/runs/cnn/predict', json={'rollout_plies': 25}).status_code, 422)
                    journal.checkpoint({'architecture': 'cnn32', 'state_dict': model.state_dict(), 'epoch': 2})
                    self.assertEqual(client.post('/training/runs/cnn/predict', json={'revision': revision}).status_code, 409)
                    self.assertEqual(client.post('/training/runs/cnn/predict', json={}).json()['checkpoint_epoch'], 2)
            finally:
                journal.close()

    def test_epoch_resume_reproduces_uninterrupted_weights(self):
        from quoridor_lab.supervised import main
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            moves = ['e2', 'e8', 'e3', 'e7']
            corpus = root / 'episodes.jsonl'
            corpus.write_text('\n'.join(json.dumps({'split': split, 'dataset_source': 'fixture',
                'episode': {'episode_id': split, 'moves': moves}}) for split in ['train', 'validation']))
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'episodes_file_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
                'episodes': [{'episode_id': split, 'split': split, 'source': 'fixture',
                    'plies': len(moves), 'sequence_sha256': hashlib.sha256(','.join(moves).encode()).hexdigest()}
                    for split in ['train', 'validation']]}))
            common = ['--manifest', str(manifest), '--architecture', 'mlp', '--device', 'cpu',
                      '--batch-size', '2', '--seed', '0']
            for mode in ['memory', 'stream']:
                with self.subTest(mode=mode):
                    continuous, resumed_path = root / f'continuous-{mode}', root / f'resumed-{mode}'
                    options = common + ['--data-mode', mode, '--shuffle-buffer', '2']
                    with contextlib.redirect_stdout(io.StringIO()):
                        main(options + ['--run-dir', str(continuous), '--epochs', '2'])
                        main(options + ['--run-dir', str(resumed_path), '--epochs', '1'])
                        main(options + ['--run-dir', str(resumed_path), '--epochs', '2', '--resume'])
                    full = torch.load(continuous / 'latest.pt', weights_only=True)
                    resumed = torch.load(resumed_path / 'latest.pt', weights_only=True)
                    self.assertEqual(full['global_step'], resumed['global_step'])
                    for key, weight in full['state_dict'].items():
                        self.assertTrue(torch.equal(weight, resumed['state_dict'][key]), key)


if __name__ == '__main__':
    unittest.main()
