"""Dataset import exclusions, preserved holdout boundaries and streaming parity."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from quoridor_lab.dataset_import import file_sha256, import_corpus, sequence_hash
from quoridor_lab.streaming_dataset import StreamingCorpus
from quoridor_lab.rules import Action, State


class DatasetImportTests(unittest.TestCase):
    def test_selected_external_dataset_does_not_silently_fall_back_when_unmounted(self):
        from quoridor_lab import dataset_paths
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'external/manifest.json'
            (root / 'active-dataset.json').write_text(json.dumps({'manifest': str(manifest)}))
            with patch.object(dataset_paths, 'DATASETS', root):
                with self.assertRaisesRegex(FileNotFoundError, 'connect its disk'):
                    dataset_paths.default_manifest()
                manifest.parent.mkdir()
                manifest.write_text('{}')
                self.assertEqual(dataset_paths.default_manifest(), manifest)

    def test_full_replay_deduplication_and_parent_holdout_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "base"
            base.mkdir()
            old_moves = ['e2', 'e8', 'e3', 'e7', 'e4', 'e6', 'e5', 'e4', 'e6', 'e3', 'e7', 'e2', 'e8', 'e1']
            old = {"dataset_source": "fixture", "split": "test", "episode": {
                "episode_id": "old-test", "moves": old_moves, "winner": State.from_moves(old_moves).winner}}
            corpus = base / "episodes.jsonl"
            corpus.write_text(json.dumps(old) + "\n")
            parent = base / "manifest.json"
            parent.write_text(json.dumps({"episodes_file_sha256": file_sha256(corpus)}))
            moves = ['d1', 'e8', 'd2', 'e7', 'd3', 'e6', 'd4', 'e5', 'd5', 'e4', 'd6', 'e3', 'd7', 'e2', 'd8', 'e1']
            reflected = [chr(105 - (ord(m[0]) - 97)) + m[1] for m in moves]
            archive = root / "cpp.zip"
            encode = lambda values: ' '.join(f'{i // 2 + 1}. {m}' if i % 2 == 0 else m for i, m in enumerate(values))
            with zipfile.ZipFile(archive, "w") as target:
                for name, content in {
                    "pgn/new.pgn": encode(moves), "pgn/duplicate.pgn": encode(moves),
                    "pgn/mirror.pgn": encode(reflected), "pgn/old.pgn": encode(old_moves),
                    "pgn/illegal.pgn": "1. e3", "pgn/unfinished.pgn": "1. e2",
                    "__MACOSX/pgn/._extra.pgn": "invalid", "tsv/new.tsv": "different representation",
                }.items():
                    target.writestr(name, content)
            archive.with_name("download-status.json").write_text(json.dumps({
                "state": "complete", "sha256": file_sha256(archive)}))
            output = root / "combined"
            with contextlib.redirect_stdout(io.StringIO()):
                report = import_corpus(archive, parent, output)
            self.assertEqual(report['import_counts']['pgn_scanned'], 6)
            self.assertEqual(report['import_counts']['cpp_games_added'], 2)
            self.assertEqual(report['import_counts']['duplicate_records_removed'], 2)
            self.assertEqual(report['import_counts']['rejected_records'], 2)
            rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0], old)
            self.assertEqual(rows[1]['split'], rows[2]['split'])
            self.assertEqual(file_sha256(output / 'episodes.jsonl'), report['episodes_file_sha256'])
            with self.assertRaises(FileExistsError):
                import_corpus(archive, parent, output)

    def test_streaming_visits_each_training_sample_once_and_never_replays_test(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder) / 'episodes.jsonl'
            rows, entries = [], []
            for uid, split, moves in [('a', 'train', ['e2', 'e8']), ('b', 'train', ['d1', 'd9']),
                                      ('v', 'validation', ['e2']), ('heldout', 'test', ['e3'])]:
                rows.append({'dataset_source': 'fixture', 'split': split, 'episode': {'episode_id': uid, 'moves': moves}})
                entries.append({'episode_id': uid, 'source': 'fixture', 'split': split,
                                'plies': len(moves), 'sequence_sha256': sequence_hash(moves)})
            corpus.write_text('\n'.join(json.dumps(row) for row in rows))
            stream = StreamingCorpus(corpus, {'episodes': entries}, shuffle_buffer=2)
            batches = list(stream.batches('train', 3, shuffle=True, seed=4, epoch=1))
            self.assertEqual(sum(len(batch['action']) for batch in batches), 4)
            actual = {}
            for batch in batches:
                for i, uid in enumerate(batch['episode']):
                    actual[(str(uid), int(batch['ply'][i]))] = batch['board'][i], batch['mask'][i], batch['action'][i]
            self.assertEqual(len(actual), 4)
            for row in rows[:2]:
                state = State()
                for ply, move in enumerate(row['episode']['moves']):
                    board, mask, action = actual[(row['episode']['episode_id'], ply)]
                    obs = state.observation()
                    np.testing.assert_array_equal(board, obs['board'])
                    np.testing.assert_array_equal(mask, obs['action_mask'])
                    self.assertEqual(action, Action.parse(move).id)
                    state = state.apply(move)
            self.assertEqual(sum(len(b['action']) for b in stream.batches('validation', 2)), 1)


if __name__ == '__main__':
    unittest.main()
