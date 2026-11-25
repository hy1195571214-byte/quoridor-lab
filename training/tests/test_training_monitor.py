"""Checkpoint-backed legal predictions and replay corpus isolation."""
import json
import hashlib
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import errno

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
    def test_phase1_poll_reuses_completed_log_and_reloads_after_append(self):
        from quoridor_lab import phase1_monitor

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "phase1" / "mlp-seed0"
            run.mkdir(parents=True)
            (run / "config.json").write_text(json.dumps({
                "architecture": "mlp", "protocol": "phase1-level1-v1", "demonstrations_sha256": "fixture",
            }))
            metrics = run / "metrics.jsonl"
            metrics.write_text('{"phase":"imitation","epoch":1}\n{"phase":"ppo","update":1,"steps":24}\n')
            app = FastAPI()
            install_training_routes(app, TrainingMonitor(root, root / "episodes.jsonl"))
            client = TestClient(app)
            with patch.object(phase1_monitor, "parse_rows", wraps=phase1_monitor.parse_rows) as parse:
                reply = client.get("/training/runs").json()["runs"][0]
                self.assertEqual((reply["imitation_done"], reply["ppo_done"]), (1, 1))
                self.assertEqual(client.get("/training/runs").json()["runs"][0], reply)
                self.assertEqual(parse.call_count, 1)
                with metrics.open("a") as stream:
                    stream.write('{"phase":"ppo","update":2,"steps":48}')
                self.assertEqual(client.get("/training/runs").json()["runs"][0]["ppo_done"], 1)
                with metrics.open("a") as stream:
                    stream.write("\n")
                self.assertEqual(client.get("/training/runs").json()["runs"][0]["ppo_done"], 2)
                self.assertEqual(parse.call_count, 3)
            detail = client.get("/training/runs/phase1:phase1:mlp-seed0")
            self.assertEqual([row["update"] for row in detail.json()["ppo"]], [1, 2])

    def test_phase1_lists_current_game_count_while_evaluating(self):
        from quoridor_lab.phase1_monitor import Phase1Monitor, PROGRESS

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "phase1" / "mlp-seed0"
            run.mkdir(parents=True)
            (run / "config.json").write_text(json.dumps({
                "architecture": "mlp", "protocol": "phase1-level1-v1", "demonstrations_sha256": "fixture",
            }))
            (run / "metrics.jsonl").write_text(
                '{"phase":"ppo","update":5,"game_units":{"completed":488,"next_evaluation":500}}\n')
            (run / PROGRESS).write_text(json.dumps({"status": "training", "activity": "evaluating",
                                                   "game_units": {"completed": 601, "next_evaluation": 500}}))
            status = Phase1Monitor(root).detail("phase1:phase1:mlp-seed0", include_logs=False)
            self.assertEqual(status["activity"], "evaluating")
            self.assertEqual(status["game_units"]["completed"], 601)

    def test_mid_epoch_io_failure_resumes_to_identical_weights(self):
        from quoridor_lab import supervised
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            examples = [('a', 'train', ['e2', 'e8', 'e3', 'e7']),
                        ('b', 'train', ['d1', 'd9', 'd2', 'd8']),
                        ('c', 'train', ['f1', 'f9', 'f2', 'f8']),
                        ('v', 'validation', ['e2', 'e8', 'e3', 'e7'])]
            corpus = root / 'episodes.jsonl'
            corpus.write_text('\n'.join(json.dumps({'split': split, 'dataset_source': 'fixture',
                'episode': {'episode_id': uid, 'moves': moves}}) for uid, split, moves in examples))
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'episodes_file_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
                'episodes': [{'episode_id': uid, 'split': split, 'source': 'fixture', 'plies': len(moves),
                    'sequence_sha256': hashlib.sha256(','.join(moves).encode()).hexdigest()}
                    for uid, split, moves in examples]}))
            original_batches = supervised.batches
            def disconnected(*args, **kwargs):
                for number, batch in enumerate(original_batches(*args, **kwargs)):
                    if kwargs.get('shuffle') and number == 3:
                        raise OSError(errno.EIO, 'fixture external disk disconnected')
                    yield batch
            common = ['--manifest', str(manifest), '--architecture', 'mlp', '--device', 'cpu',
                      '--batch-size', '2', '--seed', '0', '--epochs', '2', '--checkpoint-every', '2',
                      '--shuffle-buffer', '4']
            for mode in ['memory', 'stream']:
                with self.subTest(mode=mode), contextlib.redirect_stdout(io.StringIO()):
                    full_dir, resumed_dir = root / f'full-{mode}', root / f'interrupted-{mode}'
                    options = common + ['--data-mode', mode]
                    supervised.main(options + ['--run-dir', str(full_dir)])
                    with patch.object(supervised, 'batches', disconnected), self.assertRaises(OSError):
                        supervised.main(options + ['--run-dir', str(resumed_dir)])
                    partial = torch.load(resumed_dir / 'latest.pt', weights_only=True)
                    self.assertEqual(partial['global_step'], 3)
                    self.assertEqual(partial['resume_batch'], 3)
                    self.assertFalse(partial['epoch_complete'])
                    status = json.loads((resumed_dir / 'status.json').read_text())
                    self.assertEqual(status['failure_reason'], 'data_unavailable')
                    self.assertEqual(status['batch'], 3)
                    self.assertEqual(status['global_step'], 3)
                    before = hashlib.sha256((resumed_dir / 'latest.pt').read_bytes()).hexdigest()
                    with self.assertRaisesRegex(ValueError, 'batch size'):
                        supervised.main(options + ['--run-dir', str(resumed_dir), '--resume', '--batch-size', '3'])
                    self.assertEqual(before, hashlib.sha256((resumed_dir / 'latest.pt').read_bytes()).hexdigest())
                    supervised.main(options + ['--run-dir', str(resumed_dir), '--resume'])
                    full = torch.load(full_dir / 'latest.pt', weights_only=True)
                    resumed = torch.load(resumed_dir / 'latest.pt', weights_only=True)
                    self.assertEqual(full['global_step'], resumed['global_step'])
                    for key, value in full['state_dict'].items():
                        self.assertTrue(torch.equal(value, resumed['state_dict'][key]), key)
                    for left, right in zip(full['metrics'], resumed['metrics']):
                        self.assertEqual(left['train_loss'], right['train_loss'])
                        self.assertEqual(left['validation_loss'], right['validation_loss'])
                    self.assertNotIn('error', json.loads((resumed_dir / 'status.json').read_text()))

    def test_validation_io_failure_keeps_completed_training_batches(self):
        from quoridor_lab import supervised
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            moves = ['e2', 'e8', 'e3', 'e7']
            corpus = root / 'episodes.jsonl'
            corpus.write_text('\n'.join(json.dumps({'split': split, 'dataset_source': 'fixture',
                'episode': {'episode_id': split, 'moves': moves}}) for split in ['train', 'validation']))
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'episodes_file_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
                'episodes': [{'episode_id': split, 'split': split, 'source': 'fixture', 'plies': 4,
                    'sequence_sha256': hashlib.sha256(','.join(moves).encode()).hexdigest()}
                    for split in ['train', 'validation']]}))
            options = ['--manifest', str(manifest), '--architecture', 'mlp', '--device', 'cpu',
                       '--batch-size', '2', '--seed', '0', '--epochs', '1', '--data-mode', 'stream']
            with contextlib.redirect_stdout(io.StringIO()):
                supervised.main(options + ['--run-dir', str(root / 'full')])
                with patch.object(supervised, 'evaluate', side_effect=OSError(errno.EIO, 'fixture validation read error')):
                    with self.assertRaises(OSError):
                        supervised.main(options + ['--run-dir', str(root / 'interrupted')])
                partial = torch.load(root / 'interrupted/latest.pt', weights_only=True)
                self.assertEqual(partial['resume_batch'], 2)
                supervised.main(options + ['--run-dir', str(root / 'interrupted'), '--resume'])
            full = torch.load(root / 'full/latest.pt', weights_only=True)
            restored = torch.load(root / 'interrupted/latest.pt', weights_only=True)
            self.assertEqual(restored['global_step'], 2)
            for key, value in full['state_dict'].items():
                self.assertTrue(torch.equal(value, restored['state_dict'][key]), key)

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
