"""Loopback REST facade for notebooks and model code. No browser is needed."""
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .rules import Action, State
from .env import BarricadeEnv
from .opponents import RemoteExpert, LocalPolicy, RemoteError


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
    env: BarricadeEnv
    lock: threading.RLock = field(default_factory=threading.RLock)


def create_app(remote=None):
    remote = remote or RemoteExpert()
    sessions = {}
    sessions_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        remote.close()

    app = FastAPI(title="Barricade Local Training API", version="0.1.0", lifespan=lifespan)

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
        return {"state": current.to_dict(), "action_mask": current.action_mask().tolist()}

    @app.post("/games")
    def new_game(config: NewGame):
        if config.opponent not in {"remote", "shortest", "random"}:
            raise HTTPException(400, "opponent must be remote, shortest or random")
        env = BarricadeEnv(remote if config.opponent == "remote" else LocalPolicy(config.opponent, config.seed),
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
