"""Package the complete frozen corpus, independent test set, and original sources for GitHub Releases."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "training/data/datasets"
TAG = "training-data-2026-11-23"
REPOSITORY = "hy1195571214-byte/quoridor-lab"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(corpus_dir, output):
    corpus_dir, output = Path(corpus_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    if sha256(corpus_dir / "episodes.jsonl") != manifest["episodes_file_sha256"]:
        raise ValueError("Frozen corpus checksum differs from its manifest")
    assets = []

    def register(path, kind):
        row = {"name": path.name, "kind": kind, "bytes": path.stat().st_size, "sha256": sha256(path),
               "url": f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{path.name}"}
        assets.append(row)
        print(json.dumps(row), flush=True)

    bundle = output / "quoridor-training-data-round2-v1.tar.gz"
    with tarfile.open(bundle, "w:gz", compresslevel=3) as target:
        for name in ["manifest.json", "episodes.jsonl", "duplicates.jsonl", "rejected.jsonl", "progress.json"]:
            target.add(corpus_dir / name, arcname=f"round2-cpp208k-v1/{name}", recursive=False)
    register(bundle, "frozen_training_validation_and_test_corpus")

    tests = output / "quoridor-test-set-round2-v1.jsonl.gz"
    count = plies = 0
    with (corpus_dir / "episodes.jsonl").open("rb") as source, gzip.open(tests, "wb", compresslevel=3) as target:
        for line in source:
            row = json.loads(line)
            if row["split"] == "test":
                target.write(line)
                count += 1
                plies += len(row["episode"]["moves"])
    if (count, plies) != (manifest["splits"]["test"]["episodes"], manifest["splits"]["test"]["plies"]):
        raise ValueError("Independent test export does not match the frozen split")
    register(tests, "independent_test_set")

    originals = output / "quoridor-source-data-20261123.tar.gz"
    excluded_names = {"active-dataset.json", "release.json", "extract-status.json", "download-status.json"}
    with tarfile.open(originals, "w:gz", compresslevel=3) as target:
        for path in sorted(DATASETS.rglob("*")):
            if not path.is_file() or path.is_symlink() or any(part.startswith(".") for part in path.relative_to(DATASETS).parts):
                continue
            if path.name in excluded_names or path.name == "quoridorcpp-208k.zip" or path.suffix in {".npz", ".log", ".pid", ".lock"}:
                continue
            target.add(path, arcname="source-data/" + path.relative_to(DATASETS).as_posix(), recursive=False)
    register(originals, "original_and_intermediate_source_data_except_cpp_zip")

    original_zip = DATASETS / "quoridorcpp-208k/quoridorcpp-208k.zip"
    register(original_zip, "original_quoridorcpp_zip")
    catalog = {
        "release_tag": TAG, "repository": REPOSITORY,
        "release_url": f"https://github.com/{REPOSITORY}/releases/tag/{TAG}",
        "version": manifest["version"], "dataset_folder": "round2-cpp208k-v1",
        "episodes": sum(row["episodes"] for row in manifest["splits"].values()),
        "plies": sum(row["plies"] for row in manifest["splits"].values()),
        "splits": {key: value["episodes"] for key, value in manifest["splits"].items()},
        "manifest_sha256": sha256(corpus_dir / "manifest.json"),
        "corpus_sha256": manifest["episodes_file_sha256"],
        "test_episodes": count, "test_plies": plies, "assets": assets,
        "license_note": "Upstream Quoridor-Zero data licensing is unverified/unknown. Preserve the upstream provenance; the project license does not relicense third-party data.",
    }
    (DATASETS / "release.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    (output / "SHA256SUMS.txt").write_text("".join(f"{row['sha256']}  {row['name']}\n" for row in assets))
    print(json.dumps({"complete": True, "episodes": catalog["episodes"], "test_episodes": count,
                      "asset_bytes": sum(row["bytes"] for row in assets)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.corpus_dir, args.output)
