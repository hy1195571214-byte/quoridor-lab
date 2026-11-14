import test from 'node:test';
import assert from 'node:assert/strict';
import { createRecord, decodeShare, encodeShare, exportRecord, parseRecord } from '../src/game/record.js';

test('share links preserve a game, resignation and multiple variations', () => {
  const original = createRecord(['e2', 'e8', 'he4'], { winner: 2, reason: 'resign' }, [
    { anchor: 0, moves: ['f9', 'e3', 'f8'] },
    { anchor: 0, moves: ['d9'] },
  ]);
  assert.deepEqual(decodeShare(encodeShare(original)), original);
  assert.deepEqual(decodeShare('?moves=e2,e8,he4').moves, original.moves);
  assert.deepEqual(decodeShare('?moves=KYJgNMAcYBbALEA').moves, original.moves);
});

test('Barricade difficulty labels survive links and exported records', () => {
  const record = decodeShare('?moves=e2,e8&ai=expert,easy');
  assert.deepEqual(record.ai, ['expert', 'easy']);
  assert.deepEqual(decodeShare(encodeShare(record)), record);
  assert.deepEqual(parseRecord(exportRecord(record)), record);
  assert.equal(decodeShare('?moves=e2,e8').ai, undefined);
  assert.throws(() => decodeShare('?ai=expert,unknown'));
});

test('records reject invalid moves, branches and unsupported starting positions', () => {
  assert.throws(() => createRecord(['e3']));
  assert.throws(() => createRecord(['e2'], null, [{ anchor: 0, moves: ['e2'] }]));
  assert.throws(() => parseRecord({ moves: [], initial: { pawns: ['a1', 'e9'], walls: [10, 10], turn: 1 } }));
  assert.throws(() => decodeShare('?var=0:e8'));
  assert.throws(() => createRecord([], {winner: 1, reason: 'goal'}));
});
