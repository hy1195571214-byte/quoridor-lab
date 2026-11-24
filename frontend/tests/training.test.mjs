import test from 'node:test';
import assert from 'node:assert/strict';
import { createRecord, decodeShare, replaySelection } from '../src/game/record.js';
import { chooseTrainingRun, trainingBatchProgress, trainingReplayUrl, validatePrediction } from '../src/game/training.js';

test('the active run is preferred without overriding an explicit history selection', () => {
  const runs = [{ id: 'old', status: 'completed' }, { id: 'current', status: 'training' }];
  assert.equal(chooseTrainingRun(runs, ''), 'current');
  assert.equal(chooseTrainingRun(runs, 'old'), 'current');
  assert.equal(chooseTrainingRun(runs, 'old', true), 'old');
  assert.equal(chooseTrainingRun([], 'old'), '');
});

test('batch progress shows fractional first-epoch work and caps the final partial batch', () => {
  const progress = trainingBatchProgress({ batch: 97, batches: 25625, batch_size: 256, train_samples: 6559801 });
  assert.equal(progress.processedSamples, 24832);
  assert.ok(progress.percentage > 0.37 && progress.percentage < 0.39);
  assert.equal(trainingBatchProgress({ batch: 4, batches: 3, batch_size: 4, train_samples: 10 }).processedSamples, 10);
  assert.equal(trainingBatchProgress({ batch: 4, batches: 3 }).percentage, 100);
  assert.equal(trainingBatchProgress(null).percentage, 0);
});

test('prediction replay preserves the source game and the displayed branch position', () => {
  const original = ['e2', 'e8', 'e3'];
  const url = trainingReplayUrl(original, 1, { predicted_moves: ['d9', 'e3'] }, 1);
  const search = new URL(url, 'http://localhost').search;
  const record = decodeShare(search);
  assert.deepEqual(record.moves, original);
  assert.deepEqual(record.variations, [{ anchor: 0, moves: ['d9', 'e3'] }]);
  assert.deepEqual(replaySelection(record, search), { cursor: 0, branchId: 0, branchCursor: 0 });
  const opening = new URL(trainingReplayUrl(original, 0, { predicted_moves: ['d1'] }), 'http://localhost');
  assert.deepEqual(replaySelection(decodeShare(opening.search), opening.search),
    { cursor: -1, branchId: 0, branchCursor: -1 });
});

test('source replay links open at the chosen ply and old links keep their defaults', () => {
  const record = createRecord(['e2', 'e8'], null, [{ anchor: 0, moves: ['d9'] }]);
  assert.deepEqual(replaySelection(record), { cursor: 1, branchId: 0, branchCursor: 0 });
  const search = new URL(trainingReplayUrl(record.moves, 1), 'http://localhost').search;
  assert.deepEqual(replaySelection(decodeShare(search), search), { cursor: 0, branchId: null, branchCursor: -1 });
  assert.equal(replaySelection(record, '?ply=999&branch=999').cursor, 1);
});

test('stale checkpoint, mismatched position and illegal predictions are rejected', () => {
  const input = { runId: 'cnn', revision: 'v1', moves: ['e2'] };
  const result = { run_id: 'cnn', revision: 'v1', moves: ['e2'],
    top_actions: [{ notation: 'e8', probability: 0.4 }], predicted_moves: ['e8', 'e3'] };
  assert.equal(validatePrediction(result, input), result);
  assert.throws(() => validatePrediction({ ...result, revision: 'v0' }, input));
  assert.throws(() => validatePrediction({ ...result, moves: [] }, input));
  assert.throws(() => validatePrediction({ ...result, predicted_moves: ['e7'] }, input));
  assert.throws(() => validatePrediction({ ...result, top_actions: [{ notation: 'e8', probability: NaN }] }, input));
});
