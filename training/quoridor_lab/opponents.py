"""One persistent expert connection, explicit upstream failures, and local baselines."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import threading
import time
import uuid

import requests
import socketio

from .rules import Action, State


class RemoteError(RuntimeError):
    def __init__(self, message: str, *, kind="upstream_error", retryable=False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


@dataclass
class Decision:
    action: Action
    source: str
    latency_seconds: float = 0.0
    cached: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {"action_id": self.action.id, "notation": self.action.notation,
                "source": self.source, "latency_seconds": self.latency_seconds,
                "cached": self.cached, "metadata": self.metadata}


class LocalPolicy:
    def __init__(self, name="shortest", seed=0):
        if name not in {"shortest", "random"}:
            raise ValueError("Local policy must be shortest or random")
        self.name = name
        self.rng = random.Random(seed)

    def choose(self, state: State) -> Decision:
        if self.name == "random":
            action = self.rng.choice(state.legal_actions())
        else:
            # A deliberately simple, documented baseline: no strategic wall search.
            candidates = [Action("move", *p) for p in sorted(state.legal_destinations())]
            scored = [(a, state.apply(a).distance(state.turn)) for a in candidates]
            best = min(value for _, value in scored)
            action = self.rng.choice([a for a, value in scored if value == best])
        return Decision(action, f"local:{self.name}")

    def close(self):
        pass


class RemoteExpert:
    """Same /ai namespace and ai:get_move event as the public web client.

    No game-save, ranked-match, account or billing endpoints are called.
    Rate limits and access denials stop the caller; identities are not rotated.
    """
    difficulties = {"easy", "medium", "hard", "expert", "master"}

    def __init__(self, *, difficulty="expert", data_dir="data", interval=1.0,
                 timeout=30.0, proxy_env=False, cache=True,
                 base_url="https://api.barricade.gg"):
        if difficulty not in self.difficulties:
            raise ValueError("Unknown difficulty")
        if interval < 0.25 or timeout <= 0:
            raise ValueError("interval must be >= 0.25 seconds; timeout must be positive")
        self.difficulty, self.base_url = difficulty, base_url.rstrip("/")
        self.interval, self.timeout = interval, timeout
        self.proxy_env, self.use_cache = proxy_env, cache
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        identity_path = self.data_dir / "device-id.txt"
        try:
            with identity_path.open("x") as file:
                file.write(str(uuid.uuid4()))
        except FileExistsError:
            pass
        self.device_id = identity_path.read_text().strip()
        if not self.device_id:
            raise ValueError("Persistent device-id.txt is empty")
        self.cache_path = self.data_dir / "expert-cache.jsonl"
        self.cache = {}
        if cache and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry
                except (ValueError, KeyError):
                    continue  # A crash may leave an incomplete final cache record.
        self.client = None
        self._lock = threading.RLock()
        self._last_request = 0.0
        self.network_requests = 0
        self.cache_hits = 0

    def _connect(self):
        if self.client and self.client.connected:
            return
        session = requests.Session()
        session.trust_env = self.proxy_env
        client = socketio.Client(http_session=session, reconnection=False,
                                 request_timeout=self.timeout)
        try:
            client.connect(self.base_url, namespaces=["/ai"],
                           auth={"deviceId": self.device_id},
                           headers={"Origin": "https://barricade.gg"},
                           transports=["polling"], wait_timeout=self.timeout)
        except Exception as exc:
            client.disconnect()
            raise RemoteError(f"Expert connection failed: {exc}",
                              kind="connection", retryable=True) from exc
        self.client = client

    def choose(self, state: State) -> Decision:
        if state.winner is not None:
            raise ValueError("Cannot query the expert after the game has ended")
        with self._lock:
            key = hashlib.sha256(json.dumps([self.base_url, self.difficulty,
                                             list(state.history)]).encode()).hexdigest()
            if self.use_cache and key in self.cache:
                item = self.cache[key]
                action = Action.parse(item["move"])
                if state.error(action):
                    raise RemoteError("Cached expert move is illegal", kind="illegal_response")
                self.cache_hits += 1
                return Decision(action, "barricade:" + self.difficulty, cached=True,
                                metadata={"difficulty": self.difficulty,
                                          "observed_at": item["observed_at"],
                                          "provider_response": item["response"]})
            self._connect()
            time.sleep(max(0, self.interval - (time.monotonic() - self._last_request)))
            correlation = str(uuid.uuid4())
            self._last_request = time.monotonic()
            self.network_requests += 1
            start = time.monotonic()
            try:
                response = self.client.call("ai:get_move", {
                    "moves": ",".join(state.history), "difficulty": self.difficulty,
                    "correlationId": correlation, "deviceId": self.device_id,
                }, namespace="/ai", timeout=self.timeout)
            except Exception as exc:
                self.close()
                raise RemoteError(f"Expert request failed: {exc}", kind="transport",
                                  retryable=True) from exc
            latency = time.monotonic() - start
            if not isinstance(response, dict):
                raise RemoteError("Expert returned a non-object acknowledgement", kind="protocol")
            if response.get("error") or response.get("ok") is False:
                kind = str(response.get("error", "rejected"))
                raise RemoteError(f"Expert rejected request: {kind}", kind=kind,
                                  retryable=False)
            if response.get("correlationId", correlation) != correlation:
                raise RemoteError("Expert correlationId mismatch", kind="protocol")
            try:
                action = Action.parse(response.get("move"))
            except ValueError as exc:
                raise RemoteError(str(exc), kind="protocol") from exc
            if state.error(action):
                raise RemoteError(f"Illegal expert response: {action.notation}",
                                  kind="illegal_response")
            observed = datetime.now(timezone.utc).isoformat()
            item = {"key": key, "move": action.notation, "response": response,
                    "observed_at": observed, "history": list(state.history),
                    "difficulty": self.difficulty, "latency_seconds": latency}
            if self.use_cache:
                with self.cache_path.open("a") as file:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
                self.cache[key] = item
            return Decision(action, "barricade:" + self.difficulty, latency, False, {
                "difficulty": self.difficulty, "observed_at": observed,
                "provider_response": response,
            })

    def close(self):
        with self._lock:
            if self.client:
                try:
                    self.client.disconnect()
                except Exception:
                    pass
                self.client = None
