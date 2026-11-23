"""Local collection monitor and single-worker controller."""
from collections import Counter
from contextlib import asynccontextmanager, ExitStack
from datetime import datetime, timezone
from math import ceil
import argparse
import json
from pathlib import Path
import threading
import time
from typing import Literal
from urllib.parse import urlencode
import uuid

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .collect import _atomic, _retry_file_access, collect
from .opponents import MIN_REQUEST_INTERVAL, RemoteExpert, RequestGate
from .rules import State
from .quality import QUALITY_VERSION, geometric_review


DIAGNOSTIC_FILES = ("rejected-moves.jsonl", "deferred-games.jsonl", "last-error.json", "e2-events.jsonl")


def rows(path):
    if path.exists():
        with _retry_file_access(path.open, encoding="utf-8") as file:
            for line in file:
                if not line.endswith("\n"):
                    break
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSONL record in {path.name}") from None


def read(path):
    try:
        return json.loads(_retry_file_access(path.read_text, encoding="utf-8"))
    except FileNotFoundError:
        return None


def replay_link(base, game):
    params = {"moves": ",".join(game["moves"])}
    sides = game.get("config", {}).get("remote_sides")
    if sides:
        params["ai"] = ",".join(sides)
    if game["status"] in {"completed", "truncated", "abandoned"}:
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
Transport = Literal["polling", "websocket", "upgrade"]
LEVELS = ("easy", "medium", "hard", "expert", "master")


class RunConfig(BaseModel):
    red: Level = "expert"
    blue: Level = "expert"
    games: int = Field(default=2, ge=1, le=10000)
    concurrent_games: int = Field(default=1, ge=1, le=100)
    max_plies: int = Field(default=256, ge=2, le=256)
    time_limit_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    games_per_connection: int | None = Field(default=None, ge=1, le=100)
    request_interval: float = Field(default=1.0, ge=MIN_REQUEST_INTERVAL, allow_inf_nan=False)
    request_transport: Transport = "websocket"
    proxy_env: bool = False
    illegal_move_retries: int = Field(default=0, ge=0, le=2)
    opening: list[str] = Field(default_factory=list, max_length=255)
    use_cache: bool = True
    drain_pending: bool = False
    abandon_after_retries: int | None = Field(default=None, ge=0)
    completed_only: bool = False


class CampaignConfig(BaseModel):
    games_per_matchup: int = Field(default=10000, ge=2, le=10000, multiple_of=2)
    connections: int = Field(default=6, ge=1, le=100)
    request_interval: float = Field(default=1.0, ge=MIN_REQUEST_INTERVAL, allow_inf_nan=False)
    request_transport: Transport = "websocket"
    proxy_env: bool = False
    illegal_move_retries: int = Field(default=0, ge=0, le=2)


class ResumeConfig(BaseModel):
    concurrent_games: int | None = Field(default=None, ge=1, le=100)
    request_interval: float | None = Field(default=None, ge=MIN_REQUEST_INTERVAL, allow_inf_nan=False)
    request_transport: Transport | None = None
    proxy_env: bool | None = None
    drain_pending: bool | None = None
    abandon_after_retries: int | None = Field(default=None, ge=0)
    completed_only: bool | None = None


def campaign_runs(config):
    for index in reversed(range(len(LEVELS))):
        for weaker in reversed(LEVELS[:index + 1]):
            stronger = LEVELS[index]
            matchup = f"{weaker}-{stronger}"
            count = config.games_per_matchup
            directions = [(stronger, weaker, count)] if stronger == weaker else [
                (stronger, weaker, count // 2), (weaker, stronger, count // 2),
            ]
            for first, second, games in directions:
                yield matchup, RunConfig(red=first, blue=second, games=games,
                    concurrent_games=min(games, config.connections), games_per_connection=1,
                    request_interval=config.request_interval, illegal_move_retries=config.illegal_move_retries,
                    request_transport=config.request_transport,
                    proxy_env=config.proxy_env,
                    use_cache=False, drain_pending=True, completed_only=True).model_dump()


def create_dashboard(root, *, data_dir, replay_base, expert_factory=RemoteExpert):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.RLock()
    worker = None
    active_run_id = None
    stop_requested = None

    @asynccontextmanager
    async def lifespan(app):
        for path in root.glob("*/run.json"):
            run = read(path)
            if run and run["status"] == "running":
                run.update(status="paused", error="Monitor stopped before this run finished.")
                _atomic(path, run)
        yield

    app = FastAPI(lifespan=lifespan)

    def run_collection(folder, run, resume, stop):
        started = time.monotonic()
        previous_sampling = run.get("sampling_seconds", 0)
        sampling_end = None
        try:
            config = run["config"]
            campaign = bool(run.get("campaign"))
            completed_only = campaign or config.get("completed_only", False)
            if campaign:
                config["completed_only"] = True
            limit = config.get("time_limit_seconds")
            deadline = started + max(0, limit - previous_sampling) if limit is not None else None
            completed = sum(1 for game in rows(folder / "episodes.jsonl")
                            if not completed_only or game["status"] == "completed")
            remaining = config["games"] - completed
            if remaining > 0:
                group_size = config.get("games_per_connection")
                connection_count = ceil(min(remaining, config.get("concurrent_games", 1)) / group_size) if group_size else 1
                with ExitStack() as connections:
                    interval = config.get("request_interval", 1.0)
                    gate = RequestGate(interval=interval, stopped=stop)
                    drain_pending = config.get("drain_pending", False)
                    abandon_after_retries = config.get("abandon_after_retries")
                    retry_failed_games = campaign or drain_pending or abandon_after_retries is not None
                    use_lanes = connection_count > 1 or retry_failed_games or completed_only
                    experts = []
                    for _ in range(connection_count):
                        options = {"request_gate": gate}
                        if "request_transport" in config:
                            options["transport"] = config["request_transport"]
                        if "proxy_env" in config:
                            options["proxy_env"] = config["proxy_env"]
                        if interval != 1.0:
                            options["interval"] = interval
                        expert = expert_factory(data_dir=data_dir, cache=config.get("use_cache", True), **options)
                        connections.callback(expert.close)
                        if experts:
                            expert.cache = experts[0].cache
                        experts.append(expert)
                    result = collect(
                        experts[0], output=folder, games=remaining, resume=resume,
                        remote_sides=(config["red"], config["blue"]),
                        max_plies=config["max_plies"], opening=config["opening"],
                        concurrent_games=config.get("concurrent_games", 1),
                        deadline=deadline,
                        request_lanes=experts if use_lanes else (),
                        games_per_connection=(group_size or 1) if use_lanes else group_size,
                        illegal_move_retries=config.get("illegal_move_retries", 0),
                        defer_illegal_responses=retry_failed_games,
                        deferred_retry_limit=5 if campaign else 0,
                        retry_transient=campaign,
                        drain_pending=drain_pending,
                        abandon_after_retries=abandon_after_retries,
                        completed_only=completed_only,
                        progress=lambda _: None,
                    )
                run["connections"] = connection_count
                run["peak_inflight"] = gate.peak_inflight if use_lanes else int(result["network_requests"] > 0)
                sampling_end = deadline if result["time_limit_reached"] else time.monotonic()
                run["network_requests"] = run.get("network_requests", 0) + result["network_requests"]
                if result["error"]:
                    failure = result["error"]
                    message = "已暂停采集。" if failure["kind"] == "stopped" else failure["message"]
                    if failure['kind'] == 'suspect_e2':
                        message = '远端持续返回 e2 异常回退或重复盘面候选；已保留断点暂停，该走法未写入棋谱。'
                    elif failure['kind'] == 'e2_service_unhealthy':
                        message = '多条新连接连续收到 6 次非法 e2，已保留断点暂停，防止持续弃局重开。'
                    run.update(status="paused", error=message, error_kind=failure["kind"])
                elif result["time_limit_reached"]:
                    run.update(status="finished", error=None, stop_reason="time_limit")
                else:
                    run.update(status="finished", error=None)
            else:
                run.update(status="finished", error=None)
        except Exception as exc:
            run.update(status="failed", error=str(exc))
        run["sampling_seconds"] = previous_sampling + (sampling_end or time.monotonic()) - started
        run["elapsed_seconds"] = run.get("elapsed_seconds", 0) + time.monotonic() - started
        _atomic(folder / "run.json", run)

    def launch(folder, run, resume):
        nonlocal worker, active_run_id, stop_requested
        if worker and worker.is_alive():
            raise HTTPException(409, "A collection run is already active.")
        if run.get("campaign"):
            run["config"]["completed_only"] = True
        run.update(status="running", error=None, error_kind=None)
        _atomic(folder / "run.json", run)
        stop_requested = threading.Event()
        active_run_id = run["id"]
        worker = threading.Thread(target=run_queue, args=(folder, run, resume, stop_requested), daemon=True)
        worker.start()
        return {"id": run["id"]}

    def run_queue(folder, run, resume, stop):
        nonlocal active_run_id
        while True:
            run_collection(folder, run, resume, stop)
            with lock:
                if stop.is_set() or not run.get("campaign") or run["status"] != "finished" or run.get("stop_reason"):
                    break
                waiting = [item for path in root.glob("*/run.json")
                           if (item := read(path)) and item.get("campaign") == run["campaign"]
                           and item["status"] == "queued"]
                if not waiting:
                    break
                run = min(waiting, key=lambda item: item["campaign_order"])
                folder = root / run["id"]
                run.update(status="running", error=None, error_kind=None)
                _atomic(folder / "run.json", run)
                active_run_id = run["id"]
                resume = False

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
    def resume_run(run_id: str, settings: ResumeConfig | None = None):
        with lock:
            folder = folder_for(run_id)
            run = read(folder / "run.json")
            if run["status"] == "queued" and run.get("campaign"):
                unfinished = [item for path in root.glob("*/run.json")
                              if (item := read(path)) and item.get("campaign") == run["campaign"]
                              and item["status"] != "finished"]
                if run["id"] != min(unfinished, key=lambda item: item["campaign_order"])["id"]:
                    raise HTTPException(409, "先恢复本次正式采集中排在前面的批次。")
            elif run["status"] not in {"paused", "failed"}:
                raise HTTPException(409, "This run cannot be resumed.")
            if settings:
                overrides = settings.model_dump(exclude_none=True)
                if "abandon_after_retries" in settings.model_fields_set:
                    overrides["abandon_after_retries"] = settings.abandon_after_retries
                if overrides.get("concurrent_games", 1) > run["config"]["games"]:
                    raise HTTPException(400, "同时运行盘数不能超过总盘数。")
                run["config"].update(overrides)
            return launch(folder, run, True)

    @app.post("/api/runs/{run_id}/pause", status_code=202)
    def pause_run(run_id: str):
        with lock:
            folder = folder_for(run_id)
            if not worker or not worker.is_alive() or active_run_id != run_id or read(folder / "run.json")["status"] != "running":
                raise HTTPException(409, "此批次未在采集。")
            stop_requested.set()
            return {"id": run_id}

    @app.post("/api/campaigns", status_code=201)
    def start_campaign(config: CampaignConfig):
        with lock:
            if worker and worker.is_alive():
                raise HTTPException(409, "A collection run is already active.")
            if any((item := read(path)) and item.get("campaign")
                   and item["status"] in {"queued", "paused", "failed"}
                   for path in root.glob("*/run.json")):
                raise HTTPException(409, "Resume the existing campaign first.")
            campaign_id = uuid.uuid4().hex
            created = datetime.now(timezone.utc).isoformat()
            planned = []
            for order, (matchup, settings) in enumerate(campaign_runs(config), 1):
                run_id = uuid.uuid4().hex
                folder = root / run_id
                folder.mkdir()
                run = {"id": run_id, "created_at": created, "config": settings,
                       "status": "queued", "campaign": campaign_id,
                       "campaign_order": order, "matchup": matchup}
                _atomic(folder / "run.json", run)
                planned.append((folder, run))
            start = launch(*planned[0], False)
            return {**start, "campaign": campaign_id, "batches": len(planned),
                    "games": sum(run["config"]["games"] for _, run in planned)}

    @app.get("/api/runs")
    def runs():
        return sorted((run for path in root.glob("*/run.json") if (run := read(path))),
                      key=lambda run: (run["created_at"], -run.get("campaign_order", 0)), reverse=True)

    @app.get("/api/runs/{run_id}")
    def details(run_id: str, offset: int = Query(0, ge=0),
                limit: int | None = Query(None, ge=1, le=250)):
        folder = folder_for(run_id)
        run = read(folder / "run.json")
        games = list(rows(folder / "episodes.jsonl"))
        abandoned = list(rows(folder / "abandoned-games.jsonl"))
        seen = Counter(tuple(game["moves"]) for game in games)
        active = []
        finished_ids = {game["episode_id"] for game in games + abandoned}
        for path in (folder / "pending").glob("*.json"):
            try:
                pending = read(path)
            except PermissionError:
                # Windows may still be replacing or deleting this checkpoint.
                # Let the next poll pick it up if it is still pending.
                continue
            if pending and pending["episode_id"] not in finished_ids:
                moves = pending["moves"]
                game = {"game_number": pending.get("game_number", len(games) + 1),
                        "ply": len(moves), "turn": 1 + len(moves) % 2,
                        "deferred": bool(pending.get("deferred_error"))}
                retry_state = pending.get("retry_state")
                if retry_state and retry_state.get("ply") == len(moves):
                    game["retry_state"] = retry_state
                if moves:
                    game["replay"] = replay_link(
                        replay_base, {"moves": moves, "status": "running", "config": pending["config"]},
                    )
                active.append(game)
        active.sort(key=lambda game: game["game_number"])
        games.sort(key=lambda game: game.get("game_number", 0))
        page = games[offset:offset + limit] if limit is not None else games
        for game in page:
            game["duplicate"] = seen[tuple(game["moves"])] > 1
            game["replay"] = replay_link(replay_base, game)
        outcomes = Counter(game["status"] for game in games)
        completed_only = bool(run.get("campaign")) or run["config"].get("completed_only", False)
        opening_plies = len(run["config"].get("opening", []))
        abandoned_samples = sum(max(0, game["ply_count"] - len(game["config"].get("opening", [])))
                                for game in abandoned)
        run.update(games=page, games_saved=len(games), completed=outcomes["completed"],
                   completed_only=completed_only,
                   target_games_saved=outcomes["completed"] if completed_only else len(games),
                   truncated=outcomes["truncated"], samples_saved=sum(
                       game["ply_count"] - len(game["config"].get("opening", [])) for game in games
                   ) + sum(max(0, game["ply"] - opening_plies) for game in active) + abandoned_samples,
                   abandoned=len(abandoned), abandoned_samples=abandoned_samples,
                   active=active, deferred=sum(game["deferred"] for game in active),
                   diagnostic_files=[name for name in DIAGNOSTIC_FILES if (folder / name).is_file()],
                   duplicates=len(games) - len(seen),
                   unique_games=len(seen), directory=str(folder.resolve()))
        run["can_pause"] = bool(run["status"] == "running" and active_run_id == run_id and not stop_requested.is_set())
        e2_events = list(rows(folder / 'e2-events.jsonl'))
        run['e2_health'] = {
            'quality_version': QUALITY_VERSION,
            'scope': 'events_since_guard_deployment',
            'events': len(e2_events),
            'by_status': dict(Counter(event['status'] for event in e2_events)),
            'by_category': dict(Counter(event['quality']['category'] for event in e2_events)),
            'unverified_reviews': sum(event['status'] == 'accepted' and geometric_review(event['quality'])
                                      and not event['quality'].get('independent_confirmation')
                                      for event in e2_events),
        }
        return run

    @app.get("/api/runs/{run_id}/abandoned")
    def abandoned_games(run_id: str, offset: int = Query(0, ge=0),
                        limit: int = Query(100, ge=1, le=250)):
        folder = folder_for(run_id)
        games = list(rows(folder / "abandoned-games.jsonl"))
        games.sort(key=lambda game: game.get("game_number", 0))
        page = games[offset:offset + limit]
        for game in page:
            game["replay"] = replay_link(replay_base, game)
        return {"total": len(games), "games": page}

    @app.get("/api/runs/{run_id}/stats")
    def stats(run_id: str):
        folder = folder_for(run_id)
        return sample_stats(folder / "transitions.jsonl", list(rows(folder / "episodes.jsonl")))

    @app.get("/api/runs/{run_id}/files/{filename}")
    def download(run_id: str, filename: Literal[
        "episodes.jsonl", "transitions.jsonl", "abandoned-games.jsonl",
        "rejected-moves.jsonl", "deferred-games.jsonl", "last-error.json",
        "e2-events.jsonl",
    ]):
        path = folder_for(run_id) / filename
        if not path.is_file():
            raise HTTPException(404, "File has not been captured yet")
        media_type = "application/json" if filename == "last-error.json" else "application/x-ndjson"
        return FileResponse(path, filename=filename, media_type=media_type)

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
