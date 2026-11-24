"""Expose local search-guided training logs alongside the supervised runs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re

from fastapi import HTTPException


PREFIX = "phase1:"
PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
PROGRESS = "phase1-status.json"
LOCK = ".phase1.lock"


def read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_rows(path):
    if not path.is_file():
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    rows = []
    # The writer publishes each record with a newline. Ignore an in-flight tail.
    for line in content[:content.rfind("\n") + 1].splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def writer_active(path):
    """Return None when the presence of a writer cannot be verified."""
    try:
        with path.open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return None
    return False


@contextmanager
def phase1_progress(folder, epochs, updates):
    from .training_monitor import atomic_json

    with (folder / LOCK).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This run already has an active training writer") from None
        path = folder / PROGRESS
        plan = {"imitation_total": epochs, "ppo_total": updates}
        atomic_json(path, {**plan, "status": "training"})
        try:
            yield plan
        except BaseException as exc:
            status = "stopped" if isinstance(exc, KeyboardInterrupt) else "failed"
            atomic_json(path, {**plan, "status": status, "error": str(exc)})
            raise
        else:
            atomic_json(path, {**plan, "status": "completed"})


class Phase1Monitor:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def run_folder(self, run_id):
        parts = run_id.removeprefix(PREFIX).split(":")
        if not run_id.startswith(PREFIX) or len(parts) > 2 or not all(
            PART.fullmatch(part) and ".." not in part for part in parts
        ):
            raise HTTPException(400, "Invalid Phase 1 run ID")
        folder = self.root.joinpath(*parts)
        if not folder.is_dir() or folder.resolve() != folder:
            raise HTTPException(404, "Unknown Phase 1 run")
        return folder

    def demonstrations(self):
        summaries = []
        for path in sorted(self.root.glob("*/demonstrations.jsonl")):
            if path.resolve() != path:
                continue
            rows = read_rows(path)
            if rows:
                summaries.append({"batch": path.parent.name, "games": len(rows),
                                  "train": sum(row.get("split") == "train" for row in rows),
                                  "validation": sum(row.get("split") == "validation" for row in rows)})
        return summaries

    def runs(self):
        runs = []
        for config in [*self.root.glob("*/config.json"), *self.root.glob("*/*/config.json")]:
            run_id = PREFIX + ":".join(config.parent.relative_to(self.root).parts)
            try:
                runs.append(self.detail(run_id, include_logs=False))
            except HTTPException:
                continue
        return runs

    def detail(self, run_id, *, include_logs=True):
        folder = self.run_folder(run_id)
        config = read_json(folder / "config.json")
        architecture = config.get("architecture")
        if architecture not in ("mlp", "cnn32") or not config.get("demonstrations_sha256"):
            raise HTTPException(404, "Unknown Phase 1 run")
        metric_path = folder / "metrics.jsonl"
        progress_path = folder / PROGRESS
        progress = read_json(progress_path)
        rows = read_rows(metric_path)
        imitation = [row for row in rows if row.get("phase") == "imitation"]
        ppo = [row for row in rows if row.get("phase") == "ppo" and isinstance(row.get("update"), int)]
        latest = ppo[-1] if ppo else {}
        # Before progress manifests, the run's last planned update wrote this evaluation.
        final_update = bool(latest.get("shortest_eval"))
        status = progress.get("status", "completed" if final_update else "partial")
        if status == "training" and writer_active(folder / LOCK) is False:
            status = "stopped"
        files = (progress_path, metric_path, folder / "config.json")
        updated_at = datetime.fromtimestamp(max(path.stat().st_mtime for path in files if path.is_file()), timezone.utc)
        batch = folder.parent.name if folder.parent != self.root else folder.name
        checkpoint = (folder / "latest.pt").is_file()
        name = "CNN32" if architecture == "cnn32" else "MLP"
        level1 = config.get("protocol") == "phase1-level1-v1"
        result = {
            "id": run_id, "kind": "phase1", "batch": batch, "status": status,
            "display_name": f"Phase 1{' / Level 1' if level1 else ''} · {name}-PPO · seed {config.get('seed', 0)} · {batch}",
            "architecture": architecture, "device": config.get("device"), "seed": config.get("seed"),
            "updated_at": updated_at.isoformat(), "error": progress.get("error"),
            "imitation_done": len(imitation), "imitation_total": progress.get("imitation_total", config.get("bc_epochs")),
            "ppo_done": latest.get("update", 0), "ppo_total": progress.get("ppo_total", latest.get("update") if final_update else None),
            "steps": latest.get("steps", 0), "opponent": latest.get("opponent", config.get("start_level")),
            "warm_start_saved": (folder / "warm_start.pt").is_file(), "checkpoint_saved": checkpoint,
            "has_development_eval": any(row.get("development_eval") for row in ppo),
        }
        if level1:
            result.update(level1=True, qualified=bool(progress.get("qualified")),
                          pbrs_scale=progress.get("pbrs_scale", latest.get("hyperparameters", {}).get("pbrs_scale", 0.)),
                          required_passes=progress.get("required_passes", 3),
                          qualified_update=progress.get("qualified_update"),
                          activity=progress.get("activity"), eval_update=progress.get("eval_update"),
                          pilot_updates=config.get("inheritance", {}).get("pilot_updates"),
                          best_saved=(folder / "best.pt").is_file(),
                          actions_per_second=latest.get("actions_per_second"))
        if include_logs:
            last_eval = next((row for row in reversed(ppo) if row.get("development_eval")), None)
            evaluation_file = (f"ppo_{last_eval['update']:04d}_games.jsonl" if last_eval else "warm_start_games.jsonl")
            games = read_rows(folder / evaluation_file)
            if level1 and last_eval:
                opponents = last_eval["formal_eval"]
                games = [game for name in opponents for game in
                         [row for row in games if row.get("opponent") == name][:16]]
            result.update(imitation=imitation, ppo=ppo, evaluation=last_eval, games=games)
        return result
