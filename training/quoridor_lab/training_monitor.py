"""Atomic training progress and checkpoint-backed replay predictions."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from .phase1_monitor import PREFIX, Phase1Monitor, writer_active
from .rules import Action, State


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_revision(path):
    stats = path.stat()
    return f"{stats.st_mtime_ns}-{stats.st_size}"


class TrainingJournal:
    """A run directory has one training writer and any number of monitor readers."""

    def __init__(self, folder, architecture, seed, epochs, resume=False):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = (self.folder / ".writer.lock").open("a")
        # flock is released by the OS if training exits or is interrupted.
        import fcntl
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("This run already has an active training writer") from None
        status_path = self.folder / "status.json"
        if status_path.exists() and not resume:
            self.lock.close()
            raise ValueError("Run directory already exists; choose a new run or resume it")
        self.state = json.loads(status_path.read_text()) if resume and status_path.exists() else {}
        self.previous_seconds = self.state.get("total_seconds", self.state.get("session_seconds", 0)) if resume else 0
        for name in ["error", "failure_reason", "checkpoint_error"]:
            self.state.pop(name, None)
        self.started = time.monotonic()
        self.state.update(architecture=architecture, seed=seed, total_epochs=epochs,
                          status="preparing", metrics=self.state.get("metrics", []))
        self.update()

    def update(self, **changes):
        self.state.update(changes)
        self.state.update(updated_at=datetime.now(timezone.utc).isoformat(),
                          session_seconds=round(time.monotonic() - self.started, 2),
                          total_seconds=round(self.previous_seconds + time.monotonic() - self.started, 2))
        atomic_json(self.folder / "status.json", self.state)

    def metric(self, row):
        self.state["metrics"].append(row)
        self.update()

    def checkpoint(self, payload):
        import torch
        temporary = self.folder / f".latest.{os.getpid()}.tmp"
        torch.save(payload, temporary)
        temporary.replace(self.folder / "latest.pt")
        self.update(checkpoint_epoch=payload["epoch"],
                    checkpoint_step=payload.get("global_step", 0), checkpoint_batch=payload.get("batch"),
                    checkpoint_epoch_complete=payload.get("epoch_complete", True))

    def close(self):
        self.lock.close()


class PredictionQuery(BaseModel):
    moves: list[str] = Field(default_factory=list, max_length=256)
    rollout_plies: int = Field(default=8, ge=1, le=24)
    revision: str | None = Field(default=None, max_length=100)


class TrainingMonitor:
    def __init__(self, runs_dir, corpus):
        self.runs_dir = Path(runs_dir)
        self.corpus = Path(corpus)
        self._corpus_revision = None
        self._episodes = {}
        self._episode_offsets = {}
        self._corpus_lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._cached_model = None
        self._cached_key = None

    def run_path(self, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id) or ".." in run_id:
            raise HTTPException(400, "Invalid run ID")
        folder = self.runs_dir / run_id
        if not folder.is_dir() or folder.resolve().parent != self.runs_dir.resolve():
            raise HTTPException(404, "Unknown training run")
        return folder

    def detail(self, run_id):
        folder = self.run_path(run_id)
        try:
            status = folder / "status.json"
            state = json.loads((status if status.is_file() else folder / "launch.json").read_text())
            state.pop("command", None)
        except (OSError, ValueError):
            raise HTTPException(503, "Training status is temporarily unavailable") from None
        if status.is_file() and state.get("status") == "training" and writer_active(folder / ".writer.lock") is False:
            state["status"] = "stopped"
        checkpoint = folder / "latest.pt"
        return {**state, "id": run_id, "checkpoint": (
            {"revision": checkpoint_revision(checkpoint),
             "epoch": state.get("checkpoint_epoch"), "step": state.get("checkpoint_step"),
             "batch": state.get("checkpoint_batch"), "epoch_complete": state.get("checkpoint_epoch_complete", True)}
            if checkpoint.is_file() else None)}

    def episodes(self):
        with self._corpus_lock:
            if not self.corpus.is_file():
                raise HTTPException(503, "The frozen training corpus has not been prepared")
            revision = checkpoint_revision(self.corpus)
            if revision != self._corpus_revision:
                episodes = {}
                offsets = {}
                manifest = self.corpus.with_name("manifest.json")
                index = json.loads(manifest.read_text()).get("episodes", []) if manifest.is_file() else []
                if index and all("file_offset" in entry for entry in index):
                    for entry in index:
                        if entry["split"] not in {"train", "validation"}:
                            continue
                        uid = entry["episode_id"]
                        episodes[uid] = {"episode_id": uid, "split": entry["split"],
                                         "source": entry["source"], "winner": entry.get("winner"),
                                         "plies": entry["plies"]}
                        offsets[uid] = entry["file_offset"]
                    self._episodes, self._episode_offsets = episodes, offsets
                    self._corpus_revision = revision
                    return self._episodes
                with self.corpus.open() as source:
                    for line in source:
                        wrapper = json.loads(line)
                        # The dashboard never serves the reserved final-test corpus.
                        if wrapper["split"] not in {"train", "validation"}:
                            continue
                        row = wrapper["episode"]
                        episodes[row["episode_id"]] = {
                            "episode_id": row["episode_id"], "moves": row["moves"],
                            "split": wrapper["split"], "source": wrapper["dataset_source"],
                            "winner": row.get("winner"), "plies": len(row["moves"]),
                        }
                self._episodes = episodes
                self._episode_offsets = {}
                self._corpus_revision = revision
            return self._episodes

    def episode(self, episode_id):
        with self._corpus_lock:
            row = self.episodes().get(episode_id)
            if row is None:
                raise HTTPException(404, "Unknown development episode")
            if episode_id not in self._episode_offsets:
                return row
            with self.corpus.open("rb") as source:
                source.seek(self._episode_offsets[episode_id])
                wrapper = json.loads(source.readline())
            if wrapper["split"] != row["split"] or wrapper["episode"]["episode_id"] != episode_id:
                raise HTTPException(503, "Frozen corpus index no longer matches its episodes")
            return {**row, "moves": wrapper["episode"]["moves"]}

    def run_corpus(self, run_id):
        manifest = self.detail(run_id).get("dataset_manifest") if run_id else None
        return Path(manifest).with_name("episodes.jsonl") if manifest else self.corpus

    def predict(self, run_id, query):
        try:
            import numpy as np
            import torch
            from .learning import network
        except ImportError:
            raise HTTPException(503, "Install the training learning dependencies to enable predictions") from None

        try:
            state = State.from_moves(query.moves)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        if state.winner:
            raise HTTPException(400, "Game already finished")
        if len(query.moves) >= 256:
            raise HTTPException(400, "This position reached the 256-action limit")
        folder = self.run_path(run_id)
        checkpoint = folder / "latest.pt"
        if not checkpoint.is_file():
            raise HTTPException(409, "Waiting for the first saved checkpoint")
        with self._model_lock:
            with checkpoint.open("rb") as source:
                stats = os.fstat(source.fileno())
                revision = f"{stats.st_mtime_ns}-{stats.st_size}"
                if query.revision and query.revision != revision:
                    raise HTTPException(409, "Checkpoint changed; request the latest version")
                key = (run_id, revision)
                if key != self._cached_key:
                    try:
                        saved = torch.load(source, map_location="cpu", weights_only=True)
                        architecture = saved.get("architecture", "mlp")
                        model = network(architecture)
                        model.load_state_dict(saved["state_dict"])
                        model.eval()
                    except (ValueError, KeyError, RuntimeError, TypeError) as error:
                        raise HTTPException(503, f"Checkpoint could not be loaded: {error}") from None
                    self._cached_model = (model, saved)
                    self._cached_key = key
            model, saved = self._cached_model
            predicted = []
            top = []
            with torch.inference_mode():
                for index in range(min(query.rollout_plies, 256 - len(query.moves))):
                    if state.winner:
                        break
                    observation = state.observation()
                    mask = torch.from_numpy(observation["action_mask"].astype(np.bool_))
                    logits = model(torch.from_numpy(observation["board"]).unsqueeze(0))[0]
                    if not torch.isfinite(logits).all():
                        raise HTTPException(503, "Checkpoint produced non-finite predictions")
                    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(0)
                    if index == 0:
                        values, ids = probabilities.topk(min(3, int(mask.sum())))
                        top = [{"notation": Action.from_id(int(action_id)).notation,
                                "action_id": int(action_id), "probability": float(value)}
                               for value, action_id in zip(values, ids)]
                    action = Action.from_id(int(probabilities.argmax()))
                    predicted.append(action.notation)
                    state = state.apply(action)
            return {"run_id": run_id, "revision": revision,
                    "checkpoint_epoch": saved.get("epoch", 0),
                    "checkpoint_step": saved.get("global_step", 0),
                    "architecture": saved.get("architecture", "mlp"),
                    "moves": query.moves, "top_actions": top,
                    "predicted_moves": predicted, "winner": state.winner,
                    "truncated": state.winner is None and len(state.history) >= 256,
                    "method": "masked_argmax_both_players"}


def install_training_routes(app, monitor):
    phase1 = Phase1Monitor(monitor.runs_dir)
    corpora = {monitor.corpus: monitor}
    corpora_lock = threading.Lock()

    def corpus_monitor(run_id):
        corpus = monitor.run_corpus(run_id)
        with corpora_lock:
            if corpus not in corpora:
                corpora[corpus] = TrainingMonitor(monitor.runs_dir, corpus)
            return corpora[corpus]

    @app.get("/training/runs")
    def list_runs():
        runs = phase1.runs()
        for folder in sorted(monitor.runs_dir.glob("*")):
            if folder.is_dir() and any((folder / name).is_file() for name in ["status.json", "launch.json"]):
                try:
                    detail = monitor.detail(folder.name)
                    detail.pop("metrics", None)
                    runs.append(detail)
                except HTTPException:
                    continue
        return {"runs": sorted(runs, key=lambda item: item.get("updated_at", ""), reverse=True),
                "phase1_demonstrations": phase1.demonstrations()}

    @app.get("/training/runs/{run_id}")
    def run_detail(run_id: str):
        if run_id.startswith(PREFIX):
            return phase1.detail(run_id)
        return monitor.detail(run_id)

    @app.post("/training/runs/{run_id}/predict")
    def predict(run_id: str, query: PredictionQuery):
        return monitor.predict(run_id, query)

    @app.get("/training/episodes")
    def episodes(split: str = "train", run: str | None = None, offset: int = Query(default=0, ge=0),
                 limit: int = Query(default=200, ge=1, le=500)):
        if split not in {"train", "validation"}:
            raise HTTPException(400, "Choose train or validation")
        rows = [row for row in corpus_monitor(run).episodes().values() if row["split"] == split]
        return {"episodes": [{key: value for key, value in row.items() if key != "moves"}
                             for row in rows[offset:offset + limit]],
                "total": len(rows), "offset": offset, "limit": limit}

    @app.get("/training/episodes/{episode_id}")
    def episode(episode_id: str, run: str | None = None):
        return corpus_monitor(run).episode(episode_id)
