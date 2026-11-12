"""Build a portable ZIP containing the offline HTML, source, backend, records, and instructions."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "node_modules", "__pycache__", ".playwright-cli", "output", "releases", "pending", "build"}


def allowed(path):
    rel = path.relative_to(ROOT)
    if any(part in SKIP or part.endswith(".egg-info") for part in rel.parts):
        return False
    if path.name.startswith(".env") or path.name in {".DS_Store", "device-id.txt", "expert-cache.jsonl"}:
        return False
    return ".sqlite3" not in path.name and path.suffix not in {".pyc", ".pt", ".pth", ".log", ".zip"}


def package(output):
    required = ["frontend/offline/index.html", "frontend/dist/client/index.html", "docs/DELIVERY_GUIDE.md"]
    for item in required:
        if not (ROOT / item).exists():
            raise SystemExit(f"Missing {item}. Run npm ci && npm run build in frontend first.")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and allowed(p))
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "Quoridor-Lab/"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            name = file.relative_to(ROOT).as_posix()
            data = file.read_bytes()
            manifest["files"][name] = hashlib.sha256(data).hexdigest()
            archive.writestr(prefix + name, data)
        # Put the double-click entry at the top of the extracted project as well.
        data = (ROOT / "frontend/offline/index.html").read_bytes()
        archive.writestr(prefix + "offline/index.html", data)
        manifest["files"]["offline/index.html"] = hashlib.sha256(data).hexdigest()
        archive.writestr(prefix + "PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        error = archive.testzip()
        if error:
            raise SystemExit(f"Invalid archive member: {error}")
    print(f"{output} ({output.stat().st_size:,} bytes, {len(manifest['files'])} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "releases/Quoridor-Lab.zip")
    parser.add_argument("--build", action="store_true", help="Rebuild frontend before packaging (requires npm ci)")
    args = parser.parse_args()
    if args.build:
        subprocess.run(["npm", "run", "build"], cwd=ROOT / "frontend", check=True)
    package(args.output.resolve())
