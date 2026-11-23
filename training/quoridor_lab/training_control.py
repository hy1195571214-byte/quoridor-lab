"""Explicit, local-only training launches. Reading configuration never starts a job."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field

from .dataset_paths import DATASETS, default_manifest
from .training_monitor import atomic_json

TRAINING_ROOT = Path(__file__).resolve().parent.parent


class StartTraining(BaseModel):
    name: str = Field(default="", max_length=60)
    architecture: Literal["cnn32", "mlp"] = "cnn32"
    epochs: int = Field(default=40, ge=1, le=200)
    batch_size: Literal[64, 128, 256, 512] = 256
    seed: int = Field(default=0, ge=0, le=2147483647)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.1)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"


class TrainingController:
    def __init__(self, runs_dir, *, selection=DATASETS / "active-dataset.json", process_factory=subprocess.Popen):
        self.runs_dir = Path(runs_dir).resolve()
        self.selection = Path(selection)
        self.process_factory = process_factory
        self.processes = {}
        self.lock = threading.RLock()

    def dataset(self):
        try:
            if self.selection.is_file():
                active = json.loads(self.selection.read_text())
                manifest = Path(active["manifest"])
                corpus = Path(active["corpus"])
                if not manifest.is_absolute():
                    manifest = self.selection.parent / manifest
                if not corpus.is_absolute():
                    corpus = self.selection.parent / corpus
                if not manifest.is_file() or not corpus.is_file():
                    raise FileNotFoundError("Selected dataset disk is unavailable")
                return {**active, "name": f"合并棋谱（{active['episodes']:,} 盘）",
                        "manifest": str(manifest.resolve()), "corpus": str(corpus.resolve())}
            manifest = default_manifest()
            content = json.loads(manifest.read_text())
            splits = {key: row["episodes"] for key, row in content["splits"].items()}
            return {"name": f"棋谱数据（{sum(splits.values()):,} 盘）", "manifest": str(manifest),
                    "version": content["version"], "episodes": sum(splits.values()), "splits": splits}
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(503, "训练数据暂不可用，请检查 HY 硬盘是否已连接。") from error

    @staticmethod
    def devices():
        try:
            import torch
            return {"mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(), "cpu": True}
        except ImportError:
            raise HTTPException(503, "训练依赖尚未安装。") from None

    def active(self):
        if any(process.poll() is None for process in self.processes.values()):
            return True
        # Also respect a trainer launched in another terminal or before a server restart.
        for path in self.runs_dir.glob("*/.writer.lock"):
            with path.open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
        return False

    def next_number(self):
        numbers = [int(match[1]) for path in self.runs_dir.glob("pretrain-*")
                   if (match := re.fullmatch(r"pretrain-(\d+)", path.name))]
        return max(numbers, default=0) + 1

    def configuration(self):
        with self.lock:
            return {"dataset": self.dataset(), "devices": self.devices(), "active": self.active(),
                    "suggested_name": f"棋谱预训练 · 第{self.next_number()}轮"}

    def start(self, query):
        with self.lock:
            if self.active():
                raise HTTPException(409, "已有训练正在运行，请等待它结束后再开始。")
            dataset = self.dataset()
            devices = self.devices()
            if query.device != "auto" and not devices[query.device]:
                raise HTTPException(400, "所选训练设备当前不可用。")
            number = self.next_number()
            run_id = f"pretrain-{number:03d}"
            display_name = query.name.strip() or f"棋谱预训练 · 第{number}轮"
            folder = self.runs_dir / run_id
            folder.mkdir(parents=True)
            command = [sys.executable, "-u", "-m", "quoridor_lab.supervised",
                       "--run-dir", str(folder), "--manifest", dataset["manifest"], "--name", display_name,
                       "--architecture", query.architecture, "--epochs", str(query.epochs),
                       "--batch-size", str(query.batch_size), "--seed", str(query.seed),
                       "--learning-rate", str(query.learning_rate), "--device", query.device]
            if sys.platform == "darwin" and Path("/usr/bin/caffeinate").is_file():
                command = ["/usr/bin/caffeinate", "-i", *command]
            launch = {"display_name": display_name, "status": "preparing", "architecture": query.architecture,
                      "seed": query.seed, "total_epochs": query.epochs, "device": query.device,
                      "dataset_version": dataset["version"], "dataset_name": dataset["name"],
                      "dataset_manifest": dataset["manifest"], "metrics": [],
                      "updated_at": datetime.now(timezone.utc).isoformat(), "command": command}
            atomic_json(folder / "launch.json", launch)
            try:
                with (folder / "console.log").open("ab") as log:
                    process = self.process_factory(command, cwd=str(TRAINING_ROOT), stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.processes[run_id] = process
                launch["process_id"] = process.pid
                atomic_json(folder / "launch.json", launch)
            except OSError as error:
                atomic_json(folder / "status.json", {**launch, "status": "failed", "error": str(error)})
                raise HTTPException(503, "训练进程未能启动，请查看运行日志。") from error
            return {"id": run_id, "display_name": display_name, "status": "preparing"}


def install_control_routes(app, controller):
    @app.get("/training/config")
    def configuration():
        return controller.configuration()

    @app.post("/training/start", status_code=202)
    def start(query: StartTraining):
        return controller.start(query)
