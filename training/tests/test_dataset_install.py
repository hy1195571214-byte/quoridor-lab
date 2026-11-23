"""Published-data installation verifies bytes and cannot write outside its staging directory."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('dataset_installer', Path(__file__).resolve().parents[1] / 'examples/install_training_data.py')
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class DatasetInstallTests(unittest.TestCase):
    def fixture(self, root, unsafe=False):
        manifest, corpus = b'{"version":"fixture"}\n', b'{"split":"test"}\n'
        archive = root / 'data.tar.gz'
        with tarfile.open(archive, 'w:gz') as output:
            for name, data in [('fixture/manifest.json', manifest), ('fixture/episodes.jsonl', corpus)]:
                item = tarfile.TarInfo(name)
                item.size = len(data)
                output.addfile(item, io.BytesIO(data))
            if unsafe:
                item = tarfile.TarInfo('../outside.txt')
                item.size = 3
                output.addfile(item, io.BytesIO(b'bad'))
        catalog = {'dataset_folder': 'fixture', 'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
                   'corpus_sha256': hashlib.sha256(corpus).hexdigest(),
                   'assets': [{'kind': 'frozen_training_validation_and_test_corpus', 'sha256': installer.sha256(archive)}]}
        return archive, catalog

    def test_exact_install_and_existing_modified_data_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, catalog = self.fixture(root)
            data_dir = root / 'installed'
            installed = installer.install(archive, data_dir, catalog)
            self.assertEqual(installer.sha256(installed / 'episodes.jsonl'), catalog['corpus_sha256'])
            self.assertFalse((data_dir / 'active-dataset.json').exists())
            (installed / 'episodes.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'original files were retained'):
                installer.install(archive, data_dir, catalog)
            self.assertEqual((installed / 'episodes.jsonl').read_text(), 'changed')

    def test_checksum_failure_and_archive_traversal_do_not_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, catalog = self.fixture(root, unsafe=True)
            data_dir = root / 'installed'
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                installer.install(archive, data_dir, catalog)
            self.assertFalse((root / 'outside.txt').exists())
            self.assertFalse((data_dir / 'fixture').exists())
            catalog['assets'][0]['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'checksum'):
                installer.install(archive, data_dir, catalog)


if __name__ == '__main__':
    unittest.main()
