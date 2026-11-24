"""Download/verify the published frozen corpus. This command never starts training."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen

DATASETS = Path(__file__).resolve().parents[1] / "data/datasets"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(archive, data_dir, catalog):
    asset = next(row for row in catalog["assets"] if row["kind"] == "frozen_training_validation_and_test_corpus")
    if sha256(archive) != asset["sha256"]:
        raise ValueError("Downloaded archive checksum differs from the published checksum")
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    folder = catalog["dataset_folder"]
    destination = data_dir / folder
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix=".dataset-install-", dir=data_dir) as temporary:
            stage = Path(temporary)
            with tarfile.open(archive, "r:gz") as source:
                for member in source:
                    name = PurePosixPath(member.name)
                    if (name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != folder
                            or not member.isfile() or member.issym() or member.islnk()):
                        raise ValueError("Unsafe dataset archive member")
                    target = stage.joinpath(*name.parts)
                    if target.exists():
                        raise ValueError("Duplicate dataset archive member")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as content, target.open("xb") as output:
                        shutil.copyfileobj(content, output, 4 * 1024 * 1024)
            manifest = stage / folder / "manifest.json"
            corpus = stage / folder / "episodes.jsonl"
            if sha256(manifest) != catalog["manifest_sha256"] or sha256(corpus) != catalog["corpus_sha256"]:
                raise ValueError("Extracted dataset does not match the frozen publication")
            (stage / folder).rename(destination)
    if (sha256(destination / "manifest.json") != catalog["manifest_sha256"]
            or sha256(destination / "episodes.jsonl") != catalog["corpus_sha256"]):
        raise ValueError("An existing dataset differs; original files were retained")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded training-data tar.gz")
    parser.add_argument("--data-dir", type=Path, default=DATASETS, help="Dataset storage directory")
    args = parser.parse_args()
    catalog = json.loads((DATASETS / "release.json").read_text())
    asset = next(row for row in catalog["assets"] if row["kind"] == "frozen_training_validation_and_test_corpus")
    with tempfile.TemporaryDirectory(prefix="quoridor-download-") as temporary:
        archive = args.archive
        if archive is None:
            archive = Path(temporary) / asset["name"]
            request = Request(asset["url"], headers={"User-Agent": "QuoridorLab-dataset-installer"})
            print(f"Downloading {asset['name']} ({asset['bytes']:,} bytes)", flush=True)
            with urlopen(request, timeout=60) as response, archive.open("wb") as target:
                shutil.copyfileobj(response, target, 4 * 1024 * 1024)
        destination = install(archive, args.data_dir, catalog)
    def local_path(path):
        return path.relative_to(DATASETS).as_posix() if path.is_relative_to(DATASETS) else str(path)
    active = {"version": catalog["version"], "manifest": local_path(destination / "manifest.json"),
              "corpus": local_path(destination / "episodes.jsonl"), "manifest_sha256": catalog["manifest_sha256"],
              "corpus_sha256": catalog["corpus_sha256"], "episodes": catalog["episodes"],
              "plies": catalog["plies"], "splits": catalog["splits"], "default_data_mode": "auto"}
    path = DATASETS / "active-dataset.json"
    temporary = path.with_name(".active-dataset.json.installing")
    temporary.write_text(json.dumps(active, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    print(f"Verified {catalog['episodes']:,} games. Selected {destination}. Training has not been started.")


if __name__ == "__main__":
    main()
