import assert from "node:assert/strict";
import { initialState, actionError } from "../src/game/rules.js";
import {
  AgentRegistry,
  createAgentRunner,
  httpAgent,
} from "../src/game/agents.js";
const state = initialState(),
  registry = new AgentRegistry(),
  runner = createAgentRunner(registry);
for (const name of ["random", "shortest"])
  assert.equal(
    actionError(state, (await runner.choose(name, { state })).action),
    null,
  );
registry.register("invalid", async () => "a9");
await assert.rejects(runner.choose("invalid", { state }), /illegal/);
let finish;
registry.register("slow", () => new Promise((resolve) => (finish = resolve)));
const stale = runner.choose("slow", { state });
await runner.choose("shortest", { state });
finish("e2");
await assert.rejects(stale, (e) => e.name === "AbortError");
registry.register(
  "http",
  httpAgent("/test", async (_url, options) => ({
    ok: true,
    json: async () => ({
      notation: JSON.parse(options.body).moves.length ? "e8" : "e2",
    }),
  })),
);
assert.equal(
  (await runner.choose("http", { state, requestId: "1" })).notation,
  "e2",
);
console.log("agent tests passed");

// A plugin receives a copy and cannot change the authoritative input position.
registry.register("mutating", async ({ state: candidate }) => {
  candidate.pawns[0] = [5, 4];
  candidate.history.push("e4");
  return "e5";
});
await assert.rejects(runner.choose("mutating", { state }), /illegal/);
assert.deepEqual(state, initialState());

// Cancellation also masks a late rejection, and in-place position changes
// invalidate a reply even if the caller forgot to cancel explicitly.
let failLate;
registry.register("late-failure", () => new Promise((_, reject) => { failLate = reject; }));
const failed = runner.choose("late-failure", { state });
runner.cancel();
failLate(new Error("network failed"));
await assert.rejects(failed, (e) => e.name === "AbortError");
const changing = initialState();
let finishChanged;
registry.register("changed", () => new Promise((resolve) => { finishChanged = resolve; }));
const outdated = runner.choose("changed", { state: changing });
changing.history.push("e2");
finishChanged("e2");
await assert.rejects(outdated, (e) => e.name === "AbortError");
