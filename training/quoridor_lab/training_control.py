"""Explicit, local-only training launches. Reading configuration never starts a job."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
from typing import Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from .dataset_paths import DATASETS, default_manifest
from .training_monitor import atomic_json
from .fast_observation import INPUT_PIPELINE

TRAINING_ROOT = Path(__file__).resolve().parent.parent


class StartTraining(BaseModel):
    name: str = Field(default="", max_length=60)
    architecture: Literal["cnn32", "mlp"] = "cnn32"
    epochs: int = Field(default=40, ge=1, le=200)
    batch_size: Literal[64, 128, 256, 512] = 256
    seed: int = Field(default=0, ge=0, le=2147483647)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.1)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    checkpoint_every: int = Field(default=100, ge=1, le=10000)


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
                    "suggested_name": f"棋谱预训练 · 第{self.next_number()}轮", "input_pipeline": INPUT_PIPELINE}

    def run_path(self, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id) or ".." in run_id:
            raise HTTPException(400, "Invalid run ID")
        folder = self.runs_dir / run_id
        if not folder.is_dir() or folder.resolve().parent != self.runs_dir:
            raise HTTPException(404, "Unknown training run")
        return folder

    def pause(self, run_id):
        with self.lock:
            folder = self.run_path(run_id)
            lock_path = folder / ".writer.lock"
            if not lock_path.is_file():
                raise HTTPException(409, "此训练当前没有运行。")
            with lock_path.open("a") as writer:
                try:
                    fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise HTTPException(409, "此训练当前没有运行。")
            try:
                launch = json.loads((folder / "launch.json").read_text())
                pid = int(launch["process_id"])
                if pid < 2 or os.getpgid(pid) != pid:
                    raise ValueError("Unrecognized process group")
                command = subprocess.check_output(["/bin/ps", "-ww", "-p", str(pid), "-o", "args="],
                                                  text=True, timeout=3)
                if "quoridor_lab.supervised" not in command or str(folder) not in command:
                    raise ValueError("Process does not belong to this run")
                # Only this explicitly requested, verified launch group receives SIGINT.
                os.killpg(pid, signal.SIGINT)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                raise HTTPException(409, "无法确认此训练进程，请刷新进度后重试。") from None
            return {"id": run_id, "status": "stopping"}

    def launch(self, run_id, folder, command, launch):
        if sys.platform == "darwin" and Path("/usr/bin/caffeinate").is_file():
            command = ["/usr/bin/caffeinate", "-i", *command]
        launch = {**launch, "command": command, "status": "preparing",
                  "updated_at": datetime.now(timezone.utc).isoformat()}
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
        return {"id": run_id, "display_name": launch["display_name"], "status": "preparing"}

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
                       "--learning-rate", str(query.learning_rate), "--device", query.device,
                       "--checkpoint-every", str(query.checkpoint_every)]
            launch = {"display_name": display_name, "status": "preparing", "architecture": query.architecture,
                      "seed": query.seed, "total_epochs": query.epochs, "device": query.device,
                      "dataset_version": dataset["version"], "dataset_name": dataset["name"],
                      "dataset_manifest": dataset["manifest"], "metrics": [],
                      "settings": query.model_dump()}
            return self.launch(run_id, folder, command, launch)

    def resume(self, run_id):
        with self.lock:
            if self.active():
                raise HTTPException(409, "已有训练正在运行，请等待它结束后再继续。")
            folder = self.run_path(run_id)
            if not (folder / "latest.pt").is_file():
                raise HTTPException(409, "此运行没有可恢复的模型，请在新训练设置中重新开始。")
            try:
                state = json.loads((folder / "status.json").read_text())
                launch = json.loads((folder / "launch.json").read_text())
            except (OSError, ValueError):
                raise HTTPException(503, "原训练配置暂不可用。") from None
            manifest = Path(state.get("dataset_manifest", launch.get("dataset_manifest", "")))
            if not manifest.is_file() or not manifest.with_name("episodes.jsonl").is_file():
                raise HTTPException(503, "训练数据暂不可用，请检查 HY 硬盘是否已连接。")
            settings = StartTraining(**(launch.get("settings") or {
                "architecture": state["architecture"], "epochs": state["total_epochs"],
                "batch_size": state.get("batch_size", 256), "seed": state["seed"],
                "device": state.get("device", "auto"), "learning_rate": state.get("learning_rate", 0.0003)}))
            if settings.device != "auto" and not self.devices()[settings.device]:
                raise HTTPException(400, "原训练设备当前不可用。")
            command = [sys.executable, "-u", "-m", "quoridor_lab.supervised", "--resume",
                       "--run-dir", str(folder), "--manifest", str(manifest),
                       "--name", state.get("display_name", run_id), "--architecture", settings.architecture,
                       "--epochs", str(settings.epochs), "--batch-size", str(settings.batch_size),
                       "--seed", str(settings.seed), "--learning-rate", str(settings.learning_rate),
                       "--device", settings.device, "--checkpoint-every", str(settings.checkpoint_every),
                       "--data-mode", state.get("data_mode", "auto"),
                       "--shuffle-buffer", str(state.get("shuffle_buffer", 8192))]
            launch["resume_count"] = launch.get("resume_count", 0) + 1
            return self.launch(run_id, folder, command, launch)


def install_control_routes(app, controller):
    def explicit_local_action(request):
        origin = request.headers.get("origin")
        if origin is not None and not re.fullmatch(r"https?://(?:127\.0\.0\.1|localhost)(?::\d+)?", origin):
            raise HTTPException(403, "请在本地训练网页中执行此操作。")
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(415, "训练控制需要明确的 JSON 请求。")

    @app.get("/training/config")
    def configuration():
        return controller.configuration()

    @app.post("/training/start", status_code=202)
    def start(query: StartTraining, request: Request):
        explicit_local_action(request)
        return controller.start(query)

    @app.post("/training/runs/{run_id}/resume", status_code=202)
    def resume(run_id: str, request: Request):
        explicit_local_action(request)
        return controller.resume(run_id)

    @app.post("/training/runs/{run_id}/pause", status_code=202)
    def pause(run_id: str, request: Request):
        explicit_local_action(request)
        return controller.pause(run_id)
