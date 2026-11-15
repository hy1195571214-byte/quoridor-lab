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
from .quality import e2_quality, geometric_review


MIN_REQUEST_INTERVAL = 0.05
E2_FAULT_LIMIT = 6


class RemoteError(RuntimeError):
    def __init__(self, message: str, *, kind="upstream_error", retryable=False, diagnostics=None):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.diagnostics = diagnostics


class RequestGate:
    def __init__(self, interval=1.0, stopped=None):
        self.interval = interval
        self.lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.stopped = stopped if stopped is not None else threading.Event()
        self.last_started = 0.0
        self.inflight = 0
        self.peak_inflight = 0
        self.e2_fault_streak = 0

    def wait(self, deadline):
        with self.lock:
            delay = max(0, self.last_started + self.interval - time.monotonic())
            if deadline is not None:
                delay = min(delay, max(0, deadline - time.monotonic()))
            if self.stopped.wait(delay):
                raise RemoteError("Collection stopped after a request error", kind="stopped")
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise RemoteError("Collection time limit reached", kind="time_limit")
            self.last_started = now
            with self.cache_lock:
                self.inflight += 1
                self.peak_inflight = max(self.peak_inflight, self.inflight)

    def done(self):
        with self.cache_lock:
            self.inflight -= 1


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
                 base_url="https://api.barricade.gg", request_gate=None, transport="websocket"):
        if difficulty not in self.difficulties:
            raise ValueError("Unknown difficulty")
        if interval < MIN_REQUEST_INTERVAL or timeout <= 0:
            raise ValueError(f"interval must be >= {MIN_REQUEST_INTERVAL} seconds; timeout must be positive")
        if transport not in {"polling", "websocket", "upgrade"}:
            raise ValueError("transport must be polling, websocket or upgrade")
        self.transport = transport
        self._recovery_transport = None
        self.difficulty, self.base_url = difficulty, base_url.rstrip("/")
        self.interval, self.timeout = interval, timeout
        self.request_gate = request_gate
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
        self._connection_id = None
        self._connection_request = 0
        self.e2_fault_streak = 0

    def _record_e2_fault(self, diagnostics):
        counter = self.request_gate or self
        lock = self.request_gate.cache_lock if self.request_gate else self._lock
        with lock:
            counter.e2_fault_streak += 1
            if counter is not self:
                self.e2_fault_streak += 1
            diagnostics['e2_fault_streak'] = counter.e2_fault_streak
            diagnostics['e2_lane_fault_streak'] = self.e2_fault_streak
            diagnostics['e2_fault_limit'] = E2_FAULT_LIMIT
            return max(counter.e2_fault_streak, self.e2_fault_streak) >= E2_FAULT_LIMIT

    def _record_e2_success(self):
        counter = self.request_gate or self
        lock = self.request_gate.cache_lock if self.request_gate else self._lock
        with lock:
            counter.e2_fault_streak = 0
            self.e2_fault_streak = 0

    def _connect(self, timeout=None):
        if self.client and self.client.connected:
            return
        timeout = self.timeout if timeout is None else timeout
        session = requests.Session()
        session.trust_env = self.proxy_env
        websocket_options = {"suppress_origin": True}
        if not self.proxy_env:
            # websocket-client reads proxy environment variables independently of
            # requests.Session.trust_env. Keep this flag consistent for both paths.
            websocket_options["http_no_proxy"] = ["*"]
        client = socketio.Client(http_session=session, reconnection=False,
                                 request_timeout=timeout,
                                 # The explicit Origin header below must appear only once.
                                 websocket_extra_options=websocket_options)
        try:
            client.connect(self.base_url, namespaces=["/ai"],
                           auth={"deviceId": self.device_id},
                           headers={"Origin": "https://barricade.gg"},
                           transports=(["polling", "websocket"] if (self._recovery_transport or self.transport) == "upgrade"
                                       else [self._recovery_transport or self.transport]), wait_timeout=timeout)
        except Exception as exc:
            client.disconnect()
            raise RemoteError(f"Expert connection failed: {str(exc) or type(exc).__name__}",
                              kind="connection", retryable=True) from exc
        self.client = client
        self._connection_id = uuid.uuid4().hex
        self._connection_request = 0

    def choose(self, state: State, *, difficulty=None, deadline=None, _health_check=False,
               _confirming_e2=False) -> Decision:
        difficulty = self.difficulty if difficulty is None else difficulty
        if difficulty not in self.difficulties:
            raise ValueError("Unknown difficulty")
        if state.winner is not None:
            raise ValueError("Cannot query the expert after the game has ended")

        def remaining():
            if deadline is None:
                return self.timeout
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise RemoteError("Collection time limit reached", kind="time_limit")
            return min(seconds, self.timeout)

        with self._lock:
            remaining()
            key = hashlib.sha256(json.dumps([self.base_url, difficulty,
                                             list(state.history)]).encode()).hexdigest()
            cache_bypassed = None
            if self.use_cache and not _health_check and not _confirming_e2 and key in self.cache:
                item = self.cache[key]
                action = Action.parse(item["move"])
                quality = e2_quality(state, action)
                if state.error(action) or quality.get('quarantine') or quality.get('category') == 'review_e2':
                    cache_bypassed = {'move': action.notation, 'quality': quality,
                                      'reason': 'illegal_or_suspect_cached_move'}
                    # Preserve the append-only cache file, but do not repeat a suspect hit.
                    self.cache.pop(key)
                else:
                    self.cache_hits += 1
                    return Decision(action, "barricade:" + difficulty, cached=True,
                                    metadata={"difficulty": difficulty, "quality": quality,
                                              "observed_at": item["observed_at"],
                                              "provider_response": item["response"]})
            self._connect(remaining())
            if self.request_gate is not None:
                self.request_gate.wait(deadline)
            else:
                delay = max(0, self.interval - (time.monotonic() - self._last_request))
                if deadline is not None:
                    delay = min(delay, max(0, deadline - time.monotonic()))
                time.sleep(delay)
            try:
                timeout = remaining()
                correlation = str(uuid.uuid4())
                self._last_request = time.monotonic()
                self.network_requests += 1
                self._connection_request += 1
                start = time.monotonic()
                requested_at = datetime.now(timezone.utc).isoformat()
                try:
                    payload = {
                        "moves": ",".join(state.history), "difficulty": difficulty,
                        "correlationId": correlation, "deviceId": self.device_id,
                    }
                    response = self.client.call("ai:get_move", payload, namespace="/ai", timeout=timeout)
                except Exception as exc:
                    self.close()
                    remaining()
                    raise RemoteError(f"Expert request failed: {str(exc) or type(exc).__name__}", kind="transport",
                                      retryable=True) from exc
            finally:
                if self.request_gate is not None:
                    self.request_gate.done()
            remaining()
            latency = time.monotonic() - start
            response_diagnostics = {
                "provider_response": response,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "latency_seconds": latency,
                "request": {field: payload[field] for field in ("moves", "difficulty", "correlationId")},
                "requested_at": requested_at,
                "connection_id": self._connection_id,
                "connection_request": self._connection_request,
                "requested_transport": self.transport,
                "recovery_transport": self._recovery_transport,
                "proxy_env": self.proxy_env,
                "transport": self.client.transport() if callable(getattr(self.client, "transport", None)) else None,
                "player": state.turn,
                "ply": len(state.history),
                "purpose": 'session_check' if _health_check else 'move',
                "correlation_verified": isinstance(response, dict) and response.get("correlationId") == correlation,
            }
            if not isinstance(response, dict):
                raise RemoteError("Expert returned a non-object acknowledgement", kind="protocol",
                                  diagnostics=response_diagnostics)
            if response.get("error") or response.get("ok") is False:
                kind = str(response.get("error", "rejected"))
                raise RemoteError(f"Expert rejected request: {kind}", kind=kind,
                                  retryable=False, diagnostics=response_diagnostics)
            if response.get("correlationId", correlation) != correlation:
                raise RemoteError("Expert correlationId mismatch", kind="protocol",
                                  diagnostics=response_diagnostics)
            try:
                action = Action.parse(response.get("move"))
            except ValueError as exc:
                raise RemoteError(str(exc), kind="protocol", diagnostics=response_diagnostics) from exc
            rule_error = state.error(action)
            quality = e2_quality(state, action)
            if rule_error:
                unhealthy = False
                if action.notation == 'e2':
                    if self.transport != 'websocket':
                        self._recovery_transport = 'websocket'
                    response_diagnostics['recovery'] = {'connection_reset': True,
                                                       'next_transport': self._recovery_transport or self.transport}
                    self.close()  # Retry on a new session, retaining the same identity and history.
                    unhealthy = self._record_e2_fault(response_diagnostics)
                raise RemoteError('Repeated illegal e2 across fresh sessions; collection must pause' if unhealthy
                                  else f"Illegal expert response: {action.notation}",
                                  kind='e2_service_unhealthy' if unhealthy else "illegal_response", diagnostics={
                                      **response_diagnostics,
                                      "move": action.notation,
                                      "rule_error": rule_error,
                                      "correlation_verified": response.get("correlationId") == correlation,
                                      "quality": quality,
                                  })
            legacy_review = quality.get('category') == 'review_e2' and response_diagnostics['transport'] != 'websocket'
            # Polling-upgraded sessions share the same initialization risk even
            # when their current wire reports WebSocket.
            legacy_review = legacy_review or (quality.get('category') == 'review_e2'
                                               and self.transport == 'upgrade' and self._recovery_transport is None)
            if quality.get('quarantine') or legacy_review:
                if self.transport != 'websocket':
                    self._recovery_transport = 'websocket'
                response_diagnostics['recovery'] = {'connection_reset': True,
                                                   'next_transport': self._recovery_transport or self.transport,
                                                   'reason': 'legacy_transport_review' if legacy_review else quality['category']}
                self.close()
                raise RemoteError('Expert e2 response requires a fresh WebSocket session before retry',
                                  kind='suspect_e2', diagnostics={**response_diagnostics, 'quality': quality,
                                                                'cache_bypassed': cache_bypassed})
            if quality.get('category') == 'review_e2' and not _health_check:
                if geometric_review(quality) and not _confirming_e2:
                    # A separate easy position cannot certify this candidate. Ask
                    # the original position again on a fresh session, once only.
                    candidate = {**response_diagnostics, 'quality': quality,
                                 'cache_bypassed': cache_bypassed}
                    self.close()
                    try:
                        confirmed = self.choose(state, difficulty=difficulty, deadline=deadline,
                                                _confirming_e2=True)
                    except RemoteError as exc:
                        raise RemoteError(str(exc), kind=exc.kind, retryable=exc.retryable,
                                          diagnostics={**(exc.diagnostics or {}),
                                                       'e2_reconsideration': {'candidate': candidate,
                                                                             'outcome': 'failed'}}) from exc
                    same = confirmed.action.notation == 'e2'
                    confirmed.metadata['e2_reconsideration'] = {
                        'candidate': candidate, 'outcome': 'confirmed' if same else 'replaced'}
                    if same:
                        confirmed.metadata['quality']['independent_confirmation'] = True
                    confirmed.latency_seconds += latency
                    return confirmed
                # A legal retreat can be tactical. Test the session's handling of a
                # distinct, known position instead of rejecting on distance alone.
                # The check is fresh and never saved as a played move or cached.
                try:
                    check = self.choose(State.from_moves(['e2']), difficulty=difficulty,
                                        deadline=deadline, _health_check=True)
                except RemoteError as exc:
                    details = {**response_diagnostics, 'quality': quality,
                               'verification_error': {'kind': exc.kind, 'diagnostics': exc.diagnostics}}
                    if exc.kind == 'illegal_response':
                        if self.transport != 'websocket':
                            self._recovery_transport = 'websocket'
                        details['recovery'] = {'connection_reset': True,
                                               'next_transport': self._recovery_transport or self.transport}
                        self.close()
                        raise RemoteError('Legal e2 candidate failed independent session check; reconnect before retry',
                                          kind='suspect_e2', diagnostics=details) from exc
                    # Explicit rejection, timeout, or protocol errors keep their stop semantics.
                    raise RemoteError(str(exc), kind=exc.kind, retryable=exc.retryable, diagnostics=details) from exc
                response_diagnostics['verification'] = check.metadata['request_diagnostics']
                quality['session_check'] = 'passed'
            if not _health_check:
                self._record_e2_success()
            observed = datetime.now(timezone.utc).isoformat()
            item = {"key": key, "move": action.notation, "response": response,
                    "observed_at": observed, "history": list(state.history),
                    "difficulty": difficulty, "latency_seconds": latency}
            if self.use_cache and not _health_check:
                cache_lock = self.request_gate.cache_lock if self.request_gate is not None else self._lock
                with cache_lock:
                    with self.cache_path.open("a") as file:
                        file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    self.cache[key] = item
            return Decision(action, "barricade:" + difficulty, latency, False, {
                "difficulty": difficulty, "observed_at": observed,
                "provider_response": response,
                "request_diagnostics": response_diagnostics,
                "quality": quality,
                "cache_bypassed": cache_bypassed,
            })

    def close(self):
        with self._lock:
            if self.client:
                try:
                    self.client.disconnect()
                except Exception:
                    pass
                self.client = None
