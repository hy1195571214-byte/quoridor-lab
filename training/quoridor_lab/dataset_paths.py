"""Resolve the locally selected frozen dataset without changing older corpora."""
import json
from pathlib import Path

DATASETS = Path(__file__).resolve().parent.parent / "data/datasets"


def default_manifest():
    selection = DATASETS / "active-dataset.json"
    if selection.is_file():
        manifest = Path(json.loads(selection.read_text())["manifest"])
        if not manifest.is_absolute():
            manifest = selection.parent / manifest
        if not manifest.is_file():
            raise FileNotFoundError(f"Selected dataset is unavailable; connect its disk: {manifest}")
        return manifest
    return DATASETS / "round1-2500/manifest.json"
