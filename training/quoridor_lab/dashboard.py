"""Local collection monitor and single-worker controller."""
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import threading
import time
from typing import Literal
from urllib.parse import urlencode
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .collect import _atomic, collect
from .opponents import RemoteExpert
from .rules import State


def rows(path):
    if path.exists():
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.endswith("\n"):
                    break
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSONL record in {path.name}") from None


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def replay_link(base, game):
    params = {"moves": ",".join(game["moves"])}
    sides = game.get("config", {}).get("remote_sides")
    if sides:
        params["ai"] = ",".join(sides)
    if game["status"] in {"completed", "truncated"}:
        params["result"] = json.dumps({"winner": game["winner"], "reason": game["finished_reason"]})
    return base.rstrip("/") + "/analysis?" + urlencode(params)


def sample_stats(path, games):
    last_position = {}
    positions = Counter()
    demonstrations = Counter()
    by_level = Counter()
    requests = {}
    opening_by_game = {game["episode_id"]: game["config"].get("opening", []) for game in games}
    cached = 0
    for row in rows(path):
        episode = row["episode_id"]
        if episode not in opening_by_game or not row["is_teacher"]:
            continue
        before = last_position.get(episode)
        if before is None:
            before = State.from_moves(opening_by_game[episode]).to_dict()
        last_position[episode] = row["after"]
        position = json.dumps(
            [before[key] for key in ("pawns", "walls", "remaining_walls", "turn")],
            separators=(",", ":"),
        )
        level = row["metadata"]["difficulty"]
        positions[position] += 1
        demonstrations[position, level, row["action_id"]] += 1
        by_level[level] += 1
        cached += bool(row["cached"])
        if not row["cached"]:
            key = (tuple(row["history"]), level)
            requests.setdefault(key, Counter())[row["notation"]] += 1
    total = sum(by_level.values())
    repeated_requests = [answers for answers in requests.values() if sum(answers.values()) > 1]
    return {"games": len(games), "samples": total, "cached": cached, "by_level": by_level,
            "positions": len(positions), "repeated_positions": total - len(positions),
            "unique_demonstrations": len(demonstrations),
            "repeated_demonstrations": total - len(demonstrations),
            "repeated_request_histories": len(repeated_requests),
            "changed_request_histories": sum(len(answers) > 1 for answers in repeated_requests)}


Level = Literal["easy", "medium", "hard", "expert", "master"]


class RunConfig(BaseModel):
    red: Level = "expert"
    blue: Level = "expert"
    games: int = Field(default=2, ge=1, le=1000)
    concurrent_games: int = Field(default=1, ge=1, le=32)
    max_plies: int = Field(default=256, ge=2, le=256)
    opening: list[str] = Field(default_factory=list, max_length=255)
    use_cache: bool = True


def create_dashboard(root, *, data_dir, replay_base, expert_factory=RemoteExpert):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.RLock()
    worker = None

    @asynccontextmanager
    async def lifespan(app):
        for path in root.glob("*/run.json"):
            run = read(path)
            if run and run["status"] == "running":
                run.update(status="paused", error="Monitor stopped before this run finished.")
                _atomic(path, run)
        yield

    app = FastAPI(lifespan=lifespan)

    def run_collection(folder, run, resume):
        started = time.monotonic()
        try:
            config = run["config"]
            completed = sum(1 for _ in rows(folder / "episodes.jsonl"))
            remaining = config["games"] - completed
            if remaining > 0:
                expert = expert_factory(data_dir=data_dir, cache=config.get("use_cache", True))
                try:
                    result = collect(
                        expert, output=folder, games=remaining, resume=resume,
                        remote_sides=(config["red"], config["blue"]),
                        max_plies=config["max_plies"], opening=config["opening"],
                        concurrent_games=config.get("concurrent_games", 1),
                        progress=lambda _: None,
                    )
                finally:
                    expert.close()
                if result["error"]:
                    run.update(status="paused", error=result["error"]["message"])
                else:
                    run.update(status="finished", error=None)
            else:
                run.update(status="finished", error=None)
        except Exception as exc:
            run.update(status="failed", error=str(exc))
        run["elapsed_seconds"] = run.get("elapsed_seconds", 0) + time.monotonic() - started
        _atomic(folder / "run.json", run)

    def launch(folder, run, resume):
        nonlocal worker
        if worker and worker.is_alive():
            raise HTTPException(409, "A collection run is already active.")
        run.update(status="running", error=None)
        _atomic(folder / "run.json", run)
        worker = threading.Thread(target=run_collection, args=(folder, run, resume), daemon=True)
        worker.start()
        return {"id": run["id"]}

    def folder_for(run_id):
        if len(run_id) != 32 or any(ch not in "0123456789abcdef" for ch in run_id):
            raise HTTPException(404, "Unknown run")
        folder = root / run_id
        if not (folder / "run.json").is_file():
            raise HTTPException(404, "Unknown run")
        return folder

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).with_name("dashboard.html"), headers={"Cache-Control": "no-store"})

    @app.post("/api/runs", status_code=201)
    def start(config: RunConfig):
        if config.concurrent_games > config.games:
            raise HTTPException(400, "同时运行盘数不能超过总盘数。")
        try:
            state = State.from_moves(config.opening)
            if state.winner or len(state.history) >= config.max_plies:
                raise ValueError("Opening must leave room for further play.")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        with lock:
            if worker and worker.is_alive():
                raise HTTPException(409, "A collection run is already active.")
            run_id = uuid.uuid4().hex
            folder = root / run_id
            folder.mkdir()
            run = {"id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
                   "config": config.model_dump()}
            return launch(folder, run, False)

    @app.post("/api/runs/{run_id}/resume")
    def resume_run(run_id: str):
        with lock:
            folder = folder_for(run_id)
            run = read(folder / "run.json")
            if run["status"] not in {"paused", "failed"}:
                raise HTTPException(409, "This run cannot be resumed.")
            return launch(folder, run, True)

    @app.get("/api/runs")
    def runs():
        return sorted((run for path in root.glob("*/run.json") if (run := read(path))),
                      key=lambda run: run["created_at"], reverse=True)

    @app.get("/api/runs/{run_id}")
    def details(run_id: str):
        folder = folder_for(run_id)
        run = read(folder / "run.json")
        games = list(rows(folder / "episodes.jsonl"))
        seen = Counter(tuple(game["moves"]) for game in games)
        for game in games:
            game["duplicate"] = seen[tuple(game["moves"])] > 1
            game["replay"] = replay_link(replay_base, game)
        active = []
        finished_ids = {game["episode_id"] for game in games}
        for path in (folder / "pending").glob("*.json"):
            pending = read(path)
            if pending and pending["episode_id"] not in finished_ids:
                moves = pending["moves"]
                game = {"game_number": pending.get("game_number", len(games) + 1),
                        "ply": len(moves), "turn": 1 + len(moves) % 2}
                if moves:
                    game["replay"] = replay_link(
                        replay_base, {"moves": moves, "status": "running", "config": pending["config"]},
                    )
                active.append(game)
        active.sort(key=lambda game: game["game_number"])
        games.sort(key=lambda game: game.get("game_number", 0))
        run.update(games=games, active=active, duplicates=len(games) - len(seen),
                   unique_games=len(seen), directory=str(folder.resolve()))
        return run

    @app.get("/api/runs/{run_id}/stats")
    def stats(run_id: str):
        folder = folder_for(run_id)
        return sample_stats(folder / "transitions.jsonl", list(rows(folder / "episodes.jsonl")))

    @app.get("/api/runs/{run_id}/files/{filename}")
    def download(run_id: str, filename: Literal["episodes.jsonl", "transitions.jsonl"]):
        path = folder_for(run_id) / filename
        if not path.is_file():
            raise HTTPException(404, "File has not been captured yet")
        return FileResponse(path, filename=filename, media_type="application/x-ndjson")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--output", default="data/collection/dashboard")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--replay-base", default="http://127.0.0.1:4173")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_dashboard(args.output, data_dir=args.data_dir, replay_base=args.replay_base),
                host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
