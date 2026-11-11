import {
  actionError,
  actionId,
  applyAction,
  distance,
  legalActions,
  legalDestinations,
  notation,
  parseAction,
} from "./rules.js";

// Policies receive {state, signal, requestId}; they return an action, id, notation,
// or {action_id|notation|action}. No policy is assumed to be trained.
export class AgentRegistry {
  constructor() {
    this.agents = new Map();
    this.register("shortest", shortest);
    this.register("random", random);
  }

  register(id, choose, metadata = {}) {
    if (!id || typeof choose !== "function") throw new Error("invalid-agent");
    this.agents.set(id, { ...metadata, id, choose });
    return this;
  }

  list() {
    return [...this.agents.values()].map(({ choose, ...metadata }) => metadata);
  }

  get(id) {
    return this.agents.get(id);
  }
}

export function random({ state }) {
  const choices = legalActions(state);
  if (!choices.length) throw new Error("no-legal-action");
  return choices[Math.floor(Math.random() * choices.length)];
}

export function shortest({ state }) {
  const player = state.turn;
  const choices = legalDestinations(state).map(([row, col]) => ({
    kind: "move",
    row,
    col,
  }));
  if (!choices.length || state.winner !== null)
    throw new Error("no-legal-action");
  let bestDistance = Infinity;
  let bestActions = [];
  for (const action of choices) {
    const steps = distance(applyAction(state, action), player);
    if (steps < bestDistance) {
      bestDistance = steps;
      bestActions = [action];
    } else if (steps === bestDistance) {
      bestActions.push(action);
    }
  }
  return bestActions[Math.floor(Math.random() * bestActions.length)];
}

function abortError() {
  const error = new Error("Agent request cancelled or superseded");
  error.name = "AbortError";
  return error;
}

// One runner per game: a new request or cancel() invalidates any earlier response,
// including policies that cannot actually stop their computation.
export function createAgentRunner(registry = new AgentRegistry()) {
  let version = 0;
  let controller = null;
  return {
    registry,
    cancel() {
      version++;
      controller?.abort();
      controller = null;
    },
    async choose(id, { state, signal, requestId, ...options }) {
      this.cancel();
      const currentVersion = version;
      controller = new AbortController();
      const localController = controller;
      const position = structuredClone(state);
      const positionKey = JSON.stringify(position);
      const onAbort = () => localController.abort();
      if (signal?.aborted) localController.abort();
      else signal?.addEventListener("abort", onAbort, { once: true });
      try {
        if (localController.signal.aborted) throw abortError();
        const agent = registry.get(id);
        if (!agent) throw new Error("unknown-agent");
        const result = await agent.choose({
          state: structuredClone(position),
          signal: localController.signal,
          requestId,
          ...options,
        });
        if (currentVersion !== version || localController.signal.aborted || JSON.stringify(state) !== positionKey)
          throw abortError();
        const raw =
          result && typeof result === "object" && !("kind" in result)
            ? (result.action ?? result.action_id ?? result.notation)
            : result;
        const action = parseAction(raw);
        const reason = actionError(position, action);
        if (reason) {
          const error = new Error(`Agent returned illegal action: ${reason}`);
          error.code = reason;
          throw error;
        }
        return {
          action,
          action_id: actionId(action),
          notation: notation(action),
          requestId,
        };
      } catch (error) {
        if (currentVersion !== version || localController.signal.aborted || JSON.stringify(state) !== positionKey)
          throw abortError();
        throw error;
      } finally {
        signal?.removeEventListener("abort", onAbort);
        if (currentVersion === version) controller = null;
      }
    },
  };
}

export function httpAgent(url, fetchImpl = fetch) {
  return async ({ state, signal, requestId }) => {
    const response = await fetchImpl(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ moves: state.history, request_id: requestId }),
      signal,
    });
    if (!response.ok) throw new Error(`Agent HTTP ${response.status}`);
    return response.json();
  };
}

// Worker replies with {requestId, action|action_id|notation} or {requestId,error}.
export function workerAgent(worker) {
  return ({ state, signal, requestId }) =>
    new Promise((resolve, reject) => {
      const token = `${Date.now()}-${Math.random()}`;
      const cleanup = () => {
        worker.removeEventListener("message", onMessage);
        worker.removeEventListener("error", onError);
        signal?.removeEventListener("abort", onAbort);
      };
      const onMessage = (event) => {
        if (event.data?.requestId !== token) return;
        cleanup();
        if (event.data.error) reject(new Error(event.data.error));
        else resolve(event.data);
      };
      const onError = (event) => {
        cleanup();
        reject(new Error(event.message || "Worker failed"));
      };
      const onAbort = () => {
        cleanup();
        reject(abortError());
      };
      if (signal?.aborted) {
        onAbort();
        return;
      }
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", onError);
      signal?.addEventListener("abort", onAbort, { once: true });
      worker.postMessage({
        state,
        requestId: token,
        externalRequestId: requestId,
      });
    });
}

export const agentRegistry = new AgentRegistry();
