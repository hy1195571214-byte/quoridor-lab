"""Loopback REST facade for notebooks and model code. No browser is needed."""
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .rules import Action, State
from .env import QuoridorEnv
from .opponents import RemoteExpert, LocalPolicy, RemoteError
from .records import RecordStore


class Query(BaseModel):
    moves: list[str] = Field(default_factory=list, max_length=512)


class NewGame(BaseModel):
    learner_player: int = Field(default=1, ge=1, le=2)
    opponent: str = "remote"
    max_plies: int = Field(default=256, ge=2, le=512)
    seed: int = 0


class Step(BaseModel):
    action_id: int | None = Field(default=None, ge=0, le=208)
    notation: str | None = None


@dataclass
class Session:
    env: QuoridorEnv
    lock: threading.RLock = field(default_factory=threading.RLock)


def create_app(remote=None, agents=None, record_dir=None):
    remote = remote or RemoteExpert()
    sessions = {}
    sessions_lock = threading.RLock()
    # Additional local policies can be injected as {id: policy}, each exposing
    # choose(State) -> Decision. Model slots are deliberately unavailable by default.
    custom_ids = set(agents or {})
    policies = {"shortest": LocalPolicy("shortest"), "random": LocalPolicy("random")}
    policies.update(agents or {})
    policy_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        remote.close()

    app = FastAPI(title="Quoridor Lab Local Training API", version="0.1.5", lifespan=lifespan)
    # Offline file:// pages have the Origin "null"; loopback web dev servers
    # may run on any port. The server CLI binds only to 127.0.0.1.
    app.add_middleware(CORSMiddleware, allow_origins=["null"],
                       allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
                       allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type"])
    records = RecordStore(record_dir or remote.data_dir)

    @app.get("/records")
    def list_records():
        return {"records": records.list()}

    @app.get("/records/{record_id}")
    def get_record(record_id: str):
        item = records.get(record_id)
        if item is None:
            raise HTTPException(404, "Unknown record")
        return item

    def save_record(record, record_id=None):
        try:
            item = records.put(record, record_id)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        if item is None:
            raise HTTPException(404, "Unknown record")
        return item

    @app.post("/records", status_code=201)
    def create_record(record: dict):
        return save_record(record)

    @app.put("/records/{record_id}")
    def update_record(record_id: str, record: dict):
        return save_record(record, record_id)

    @app.delete("/records/{record_id}")
    def delete_record(record_id: str):
        if not records.delete(record_id):
            raise HTTPException(404, "Unknown record")
        return {"deleted": record_id}

    @app.exception_handler(RemoteError)
    async def remote_error(request, error):
        from fastapi.responses import JSONResponse
        status = 429 if error.kind == "rate_limited" else 502
        return JSONResponse(status_code=status, content={"error": error.kind,
            "message": str(error), "retryable": error.retryable,
            "note": "No move was substituted. A game step may have committed the learner move; inspect its state before reset."})

    def find(game_id):
        with sessions_lock:
            if game_id not in sessions:
                raise HTTPException(404, "Unknown local game")
            return sessions[game_id]

    def snapshot(session):
        obs = session.env.state.observation(session.env.max_plies)
        if session.env._done:
            obs["action_mask"].fill(0)
        return {"state": session.env.state.to_dict(), "learner_player": session.env.learner_player,
                "done": session.env._done,
                "observation": obs["board"].tolist(), "action_mask": obs["action_mask"].tolist()}

    @app.get("/health")
    def health():
        return {"ok": True, "difficulty": remote.difficulty, "action_count": 209,
                "observation_shape": [8, 9, 9], "network_requests": remote.network_requests,
                "cache_hits": remote.cache_hits}

    @app.post("/expert/move")
    def expert_move(query: Query):
        try:
            state = State.from_moves(query.moves)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if state.winner:
            raise HTTPException(400, "Game already finished")
        return remote.choose(state).to_dict()

    @app.post("/state")
    def state(query: Query):
        try:
            current = State.from_moves(query.moves)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        obs = current.observation()
        return {"state": current.to_dict(), "action_mask": obs["action_mask"].tolist(),
                "observation": obs["board"].tolist()}

    @app.get("/agents")
    def list_agents():
        ids = list(dict.fromkeys(["shortest", "random", "dl", "rl", "search", *policies]))
        return {"agents": [{"id": name, "available": name in policies,
                            "kind": "baseline" if name in {"shortest", "random"} and name not in custom_ids else "custom"}
                           for name in ids]}

    @app.post("/agents/{agent_id}/move")
    def agent_move(agent_id: str, query: Query):
        if agent_id not in policies:
            raise HTTPException(503 if agent_id in {"dl", "rl", "search"} else 404,
                                "Agent is not configured")
        try:
            current = State.from_moves(query.moves)
            if current.winner is not None:
                raise ValueError("Game already finished")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        with policy_lock:
            decision = policies[agent_id].choose(current)
        action = getattr(decision, "action", None)
        if not isinstance(action, Action) or current.error(action) or not callable(getattr(decision, "to_dict", None)):
            raise HTTPException(502, "Agent returned an invalid or illegal action")
        result = decision.to_dict()
        if not isinstance(result, dict):
            raise HTTPException(502, "Agent returned an invalid decision")
        # The validated Action is authoritative, even for custom policies.
        return {**result, "action_id": action.id, "notation": action.notation}

    @app.post("/games")
    def new_game(config: NewGame):
        if config.opponent not in {"remote", "shortest", "random"}:
            raise HTTPException(400, "opponent must be remote, shortest or random")
        env = QuoridorEnv(remote if config.opponent == "remote" else LocalPolicy(config.opponent, config.seed),
                           config.learner_player, config.max_plies)
        obs, info = env.reset(seed=config.seed)
        game_id, session = uuid.uuid4().hex, Session(env)
        with sessions_lock:
            sessions[game_id] = session
        return {"game_id": game_id, **snapshot(session), "transitions": info["transitions"]}

    @app.get("/games/{game_id}")
    def get_game(game_id: str):
        session = find(game_id)
        with session.lock:
            return snapshot(session)

    @app.post("/games/{game_id}/step")
    def step(game_id: str, action: Step):
        if (action.action_id is None) == (action.notation is None):
            raise HTTPException(400, "Supply exactly one of action_id or notation")
        session = find(game_id)
        with session.lock:
            try:
                action_id = action.action_id if action.action_id is not None else Action.parse(action.notation).id
                _, reward, terminated, truncated, info = session.env.step(action_id)
            except (ValueError, RuntimeError) as exc:
                if isinstance(exc, RemoteError):
                    raise
                raise HTTPException(400, str(exc)) from exc
            return {**snapshot(session), "reward": reward, "terminated": terminated,
                    "truncated": truncated, "transitions": info["transitions"]}

    @app.post("/games/{game_id}/reset")
    def reset(game_id: str):
        session = find(game_id)
        with session.lock:
            _, info = session.env.reset()
            return {**snapshot(session), "transitions": info["transitions"]}

    @app.delete("/games/{game_id}")
    def delete(game_id: str):
        session = find(game_id)
        with session.lock, sessions_lock:
            sessions.pop(game_id, None)
        return {"deleted": game_id}

    return app
