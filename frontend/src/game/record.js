import { nonselectedSample } from "./nonselectedSample.js";
import { selectedSample } from "./selectedSample.js";
import LZString from "lz-string";
import { replay, notation } from "./rules.js";

export const replaySamples = [
  { key: "selected", label: "精选棋谱抽样", games: selectedSample },
  { key: "nonselected", label: "非精选棋谱抽样", games: nonselectedSample },
];

const MAX_INPUT = 128_000;
const MAX_MOVES = 1000;
const MAX_VARIATIONS = 100;
const INITIAL = { pawns: ["e1", "e9"], walls: [10, 10], turn: 1 };
const AI_DIFFICULTIES = new Set(["easy", "medium", "hard", "expert", "master"]);
const fail = (message) => {
  throw new Error(message);
};
const normalizeMoves = (moves, label) => {
  if (!Array.isArray(moves) || moves.length > MAX_MOVES)
    fail(`${label}: invalid move list`);
  return moves.map((move) => {
    if (typeof move !== "string" || move.length > 3)
      fail(`${label}: invalid move`);
    return notation(move);
  });
};

/** Validate and normalize a complete record; all branches are replayed from their anchor. */
export function parseRecord(input) {
  if (typeof input === "string") {
    if (input.length > MAX_INPUT) fail("Record is too large");
    try {
      input = JSON.parse(input);
    } catch {
      fail("Invalid record JSON");
    }
  } else if (JSON.stringify(input)?.length > MAX_INPUT)
    fail("Record is too large");
  if (!input || typeof input !== "object" || Array.isArray(input))
    fail("Invalid record");
  if (input.schema_version != null && input.schema_version !== 1)
    fail("Unsupported record version");
  if (input.rules_version != null && input.rules_version !== "classic-9x9-v1")
    fail("Unsupported rules version");
  if (
    input.initial != null &&
    (JSON.stringify(input.initial.pawns) !== JSON.stringify(INITIAL.pawns) ||
      JSON.stringify(input.initial.walls) !== JSON.stringify(INITIAL.walls) ||
      input.initial.turn !== 1)
  )
    fail("Unsupported starting position");
  const moves = normalizeMoves(input.moves ?? [], "Main line");
  const state = replay(moves);
  const variations = input.variations ?? [];
  if (!Array.isArray(variations) || variations.length > MAX_VARIATIONS)
    fail("Invalid variations");
  let total = moves.length;
  const normalizedVariations = variations.map((branch) => {
    if (
      !branch ||
      !Number.isInteger(branch.anchor) ||
      branch.anchor < -1 ||
      branch.anchor >= moves.length
    )
      fail("Invalid variation anchor");
    const branchMoves = normalizeMoves(branch.moves, "Variation");
    if (!branchMoves.length) fail("Empty variation");
    total += branchMoves.length;
    if (total > MAX_MOVES) fail("Too many moves");
    replay([...moves.slice(0, branch.anchor + 1), ...branchMoves]);
    return { anchor: branch.anchor, moves: branchMoves };
  });
  let result = input.result ?? null;
  if (result !== null) {
    if (
      typeof result !== "object" ||
      ![1, 2, null].includes(result.winner) ||
      typeof result.reason !== "string" ||
      !result.reason ||
      result.reason.length > 64
    )
      fail("Invalid result");
    if (state.winner && result.winner !== state.winner)
      fail("Result contradicts final position");
    if (result.reason === "goal" && (!state.winner || result.winner !== state.winner))
      fail("Goal result contradicts final position");
    result = { winner: result.winner, reason: result.reason };
  }
  const ai = input.ai;
  if (ai != null && (!Array.isArray(ai) || ai.length !== 2 || !ai.every(level => AI_DIFFICULTIES.has(level)))) {
    fail("Invalid Barricade AI difficulties");
  }
  return {
    schema_version: 1,
    rules_version: "classic-9x9-v1",
    initial: { pawns: [...INITIAL.pawns], walls: [...INITIAL.walls], turn: 1 },
    moves,
    result,
    variations: normalizedVariations,
    ...(ai && { ai: [...ai] }),
  };
}

export function createRecord(moves = [], result = null, variations = []) {
  return parseRecord({ moves, result, variations });
}

/** Optional deep links use ply as the number of actions already played. */
export function replaySelection(record, search = "") {
  const params = new URLSearchParams(search);
  const fallback = {
    cursor: record.moves.length - 1,
    branchId: record.variations.length ? record.variations.length - 1 : null,
    branchCursor: (record.variations.at(-1)?.moves.length ?? 0) - 1,
  };
  if (!/^\d+$/.test(params.get("ply") ?? "")) return fallback;
  const ply = Number(params.get("ply"));
  const branchValue = params.get("branch");
  if (/^\d+$/.test(branchValue ?? "")) {
    const branchId = Number(branchValue);
    const branch = record.variations[branchId];
    if (branch) return {
      cursor: branch.anchor, branchId,
      branchCursor: Math.max(-1, Math.min(branch.moves.length - 1, ply - branch.anchor - 2)),
    };
  }
  return { cursor: Math.max(-1, Math.min(record.moves.length - 1, ply - 1)),
    branchId: null, branchCursor: -1 };
}

/** Original links use LZ-String URI-compressed comma-delimited notation. */
export function encodeShare(input) {
  const record = parseRecord(input);
  const params = new URLSearchParams();
  if (record.moves.length)
    params.set(
      "moves",
      LZString.compressToEncodedURIComponent(record.moves.join(",")),
    );
  const groups = new Map();
  for (const { anchor, moves } of record.variations) {
    if (!groups.has(anchor)) groups.set(anchor, []);
    groups.get(anchor).push(moves.join(","));
  }
  if (groups.size)
    params.set(
      "var",
      [...groups]
        .map(([anchor, lines]) => `${anchor}:${lines.join("|")}`)
        .join(";"),
    );
  if (record.result) params.set("result", JSON.stringify(record.result));
  if (record.ai) params.set("ai", record.ai.join(","));
  return `/analysis${params.size ? `?${params}` : ""}`;
}

export function decodeShare(search = "") {
  if (typeof search !== "string" || search.length > MAX_INPUT)
    fail("Share link is too large");
  const query = search.includes("?")
    ? search.slice(search.indexOf("?") + 1).split("#")[0]
    : search.replace(/^\?/, "");
  const params = new URLSearchParams(query);
  const collection = replaySamples.find(({ key }) => params.has(key));
  if (collection && !params.has("moves")) {
    const number = params.get(collection.key);
    const sample = /^[1-9]\d*$/.test(number) && collection.games[Number(number) - 1];
    if (!sample) fail("Unknown sample game");
    params.set("moves", sample.moves);
    params.set("result", JSON.stringify({ winner: sample.winner, reason: "goal" }));
  }
  const encoded = params.get("moves");
  let moves = [];
  if (encoded) {
    let plain = encoded;
    if (!encoded.includes(","))
      plain = LZString.decompressFromEncodedURIComponent(encoded) || encoded;
    if (plain.length > MAX_INPUT) fail("Move list is too large");
    moves = plain.split(",").filter(Boolean);
  }
  const variations = [];
  const variationText = params.get("var");
  if (variationText)
    for (const group of variationText.split(";")) {
      const match = /^(-?\d+):(.+)$/.exec(group);
      if (!match) fail("Invalid variation link");
      for (const line of match[2].split("|"))
        variations.push({ anchor: Number(match[1]), moves: line.split(",") });
      if (variations.length > MAX_VARIATIONS) fail("Too many variations");
    }
  let result = null;
  if (params.has("result")) {
    try {
      result = JSON.parse(params.get("result"));
    } catch {
      fail("Invalid result");
    }
  }
  const ai = params.has("ai") ? params.get("ai").split(",") : null;
  return parseRecord({ moves, result, variations, ai });
}

export function exportRecord(record) {
  return JSON.stringify(parseRecord(record), null, 2);
}
