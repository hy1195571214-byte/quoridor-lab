import test from 'node:test';
import assert from 'node:assert/strict';
import { createRecord, decodeShare, encodeShare, parseRecord } from '../src/game/record.js';

test('share links preserve a game, resignation and multiple variations', () => {
  const original = createRecord(['e2', 'e8', 'he4'], { winner: 2, reason: 'resign' }, [
    { anchor: 0, moves: ['f9', 'e3', 'f8'] },
    { anchor: 0, moves: ['d9'] },
  ]);
  assert.deepEqual(decodeShare(encodeShare(original)), original);
  assert.deepEqual(decodeShare('?moves=e2,e8,he4').moves, original.moves);
  assert.deepEqual(decodeShare('?moves=KYJgNMAcYBbALEA').moves, original.moves);
});

test('records reject invalid moves, branches and unsupported starting positions', () => {
  assert.throws(() => createRecord(['e3']));
  assert.throws(() => createRecord(['e2'], null, [{ anchor: 0, moves: ['e2'] }]));
  assert.throws(() => parseRecord({ moves: [], initial: { pawns: ['a1', 'e9'], walls: [10, 10], turn: 1 } }));
  assert.throws(() => decodeShare('?var=0:e8'));
  assert.throws(() => createRecord([], {winner: 1, reason: 'goal'}));
});
