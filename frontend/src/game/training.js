import { createRecord, encodeShare } from "./record.js";
import { actionError, parseAction, replay } from "./rules.js";

export function chooseTrainingRun(runs, currentId, manualSelection = false) {
  const current = runs.find(run => run.id === currentId);
  const active = runs.find(run => ["preparing", "training"].includes(run.status));
  if (current && (manualSelection || ["preparing", "training"].includes(current.status))) return current.id;
  return active?.id ?? current?.id ?? runs[0]?.id ?? "";
}

export function trainingBatchProgress(run) {
  const batches = Math.max(0, Number(run?.batches) || 0);
  const batch = Math.min(batches, Math.max(0, Number(run?.batch) || 0));
  const totalSamples = Math.max(0, Number(run?.train_samples) || 0);
  const processedSamples = Math.min(totalSamples, batch * (Number(run?.batch_size) || 0));
  return { batch, batches, percentage: batches ? batch / batches * 100 : 0,
    processedSamples, totalSamples };
}

export function validatePrediction(result, { runId, revision, moves }) {
  if (
    result.run_id !== runId || result.revision !== revision ||
    JSON.stringify(result.moves) !== JSON.stringify(moves) ||
    !Array.isArray(result.predicted_moves) || result.predicted_moves.length > 24
  ) throw new Error("Prediction does not match the selected checkpoint and position");
  const state = replay(moves);
  if (!Array.isArray(result.top_actions) || !result.top_actions.length)
    throw new Error("Prediction has no legal candidates");
  for (const item of result.top_actions) {
    if (!Number.isFinite(item.probability) || item.probability < 0 || item.probability > 1 ||
        actionError(state, parseAction(item.notation)))
      throw new Error("Prediction contains an invalid candidate");
  }
  replay([...moves, ...result.predicted_moves]);
  return result;
}

export function trainingReplayUrl(originalMoves, ply, prediction = null, predictionPly = 0) {
  const variations = prediction?.predicted_moves.length
    ? [{ anchor: ply - 1, moves: prediction.predicted_moves }]
    : [];
  const record = createRecord(originalMoves, null, variations);
  const url = new URL(encodeShare(record), "http://localhost");
  url.searchParams.set("ply", String(ply + (variations.length ? predictionPly : 0)));
  if (variations.length) url.searchParams.set("branch", "0");
  return url.pathname + url.search;
}
