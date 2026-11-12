import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("Vercel deploys the built frontend and resolves direct application links", async () => {
  const config = JSON.parse(await readFile(new URL("../vercel.json", import.meta.url), "utf8"));
  assert.equal(config.installCommand, "npm ci");
  assert.equal(config.buildCommand, "npm run build");
  assert.equal(config.outputDirectory, "dist/client");
  assert.deepEqual(config.rewrites, [
    { source: "/local", destination: "/index.html" },
    { source: "/computer", destination: "/index.html" },
    { source: "/computer/intro", destination: "/index.html" },
    { source: "/analysis", destination: "/index.html" },
    { source: "/rules", destination: "/index.html" },
    { source: "/settings", destination: "/index.html" },
  ]);
});
