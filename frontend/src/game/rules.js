// 9×9 Quoridor rules, aligned with training/quoridor_lab/rules.py.
export const ACTION_COUNT = 209;

export function initialState() {
  return { pawns: [[8, 4], [0, 4]], walls: [], remaining: [10, 10], turn: 1, winner: null, history: [] };
}

export function parseAction(input) {
  let kind, row, col;
  if (Number.isInteger(input)) {
    if (input < 0 || input >= ACTION_COUNT) throw new Error('invalid-action');
    if (input < 81) {
      kind = 'move';
      row = Math.floor(input / 9);
      col = input % 9;
    } else {
      const offset = input - 81;
      kind = offset < 64 ? 'h' : 'v';
      row = Math.floor((offset % 64) / 8);
      col = offset % 8;
    }
  } else if (typeof input === 'string') {
    if (/^[a-i][1-9]$/.test(input)) {
      kind = 'move';
      row = 9 - Number(input[1]);
      col = input.charCodeAt(0) - 97;
    } else {
      if (/^[a-h][1-8][hv]$/.test(input)) input = input[2] + input.slice(0, 2);
      if (!/^[hv][a-h][1-8]$/.test(input)) throw new Error('invalid-action');
      kind = input[0];
      row = 8 - Number(input[2]);
      col = input.charCodeAt(1) - 97;
    }
  } else if (input && typeof input === 'object') {
    ({ kind, row, col } = input);
  }
  const limit = kind === 'move' ? 9 : 8;
  if (!['move', 'h', 'v'].includes(kind) || !Number.isInteger(row) || !Number.isInteger(col) || row < 0 || col < 0 || row >= limit || col >= limit) {
    throw new Error('invalid-action');
  }
  return { kind, row, col };
}

export function notation(input) {
  const { kind, row, col } = parseAction(input);
  const column = String.fromCharCode(97 + col);
  return kind === 'move' ? `${column}${9 - row}` : `${kind}${column}${8 - row}`;
}

export function actionId(input) {
  const { kind, row, col } = parseAction(input);
  return kind === 'move' ? row * 9 + col : 81 + (kind === 'v' ? 64 : 0) + row * 8 + col;
}

const inside = (row, col) => row >= 0 && row < 9 && col >= 0 && col < 9;
const edgeKey = (row, col, nextRow, nextCol) => [row * 9 + col, nextRow * 9 + nextCol].sort((a, b) => a - b).join(':');
const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];

function blockedEdges(walls) {
  const edges = new Set();
  for (const [kind, row, col] of walls) {
    for (let offset = 0; offset < 2; offset++) {
      edges.add(kind === 'h'
        ? edgeKey(row, col + offset, row + 1, col + offset)
        : edgeKey(row + offset, col, row + offset, col + 1));
    }
  }
  return edges;
}

// Breadth-first search ignores the other pawn, as required for wall reachability.
export function distance(state, player) {
  const edges = blockedEdges(state.walls);
  const start = state.pawns[player - 1];
  const goalRow = player === 1 ? 0 : 8;
  const seen = new Set([start.join(',')]);
  const queue = [[...start, 0]];
  for (let index = 0; index < queue.length; index++) {
    const [row, col, steps] = queue[index];
    if (row === goalRow) return steps;
    for (const [deltaRow, deltaCol] of directions) {
      const nextRow = row + deltaRow, nextCol = col + deltaCol;
      const squareKey = `${nextRow},${nextCol}`;
      if (inside(nextRow, nextCol) && !seen.has(squareKey) && !edges.has(edgeKey(row, col, nextRow, nextCol))) {
        seen.add(squareKey);
        queue.push([nextRow, nextCol, steps + 1]);
      }
    }
  }
  return 999;
}

export function legalDestinations(state, player = state.turn) {
  const [row, col] = state.pawns[player - 1];
  const [opponentRow, opponentCol] = state.pawns[2 - player];
  const edges = blockedEdges(state.walls);
  const destinations = [];
  const free = (fromRow, fromCol, toRow, toCol) => inside(toRow, toCol) && !edges.has(edgeKey(fromRow, fromCol, toRow, toCol));
  for (const [deltaRow, deltaCol] of directions) {
    const nextRow = row + deltaRow, nextCol = col + deltaCol;
    if (!free(row, col, nextRow, nextCol)) continue;
    if (nextRow !== opponentRow || nextCol !== opponentCol) {
      destinations.push([nextRow, nextCol]);
    } else if (free(nextRow, nextCol, nextRow + deltaRow, nextCol + deltaCol)) {
      destinations.push([nextRow + deltaRow, nextCol + deltaCol]);
    } else {
      for (const [sideRow, sideCol] of [[-deltaCol, deltaRow], [deltaCol, -deltaRow]]) {
        if (free(nextRow, nextCol, nextRow + sideRow, nextCol + sideCol)) {
          destinations.push([nextRow + sideRow, nextCol + sideCol]);
        }
      }
    }
  }
  return destinations.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

export function wallError(state, input) {
  const { kind, row, col } = parseAction(input);
  if (state.remaining[state.turn - 1] <= 0) return 'no-walls';
  for (const [existingKind, existingRow, existingCol] of state.walls) {
    if (existingKind !== kind && existingRow === row && existingCol === col) return 'crossing';
    if (existingKind === kind && ((kind === 'h' && existingRow === row && Math.abs(existingCol - col) < 2) ||
      (kind === 'v' && existingCol === col && Math.abs(existingRow - row) < 2))) return 'overlap';
  }
  const candidate = { ...state, walls: [...state.walls, [kind, row, col]] };
  return [1, 2].some(player => distance(candidate, player) === 999) ? 'blocks-path' : null;
}

export function actionError(state, input) {
  const action = parseAction(input);
  if (state.winner !== null) return 'game-over';
  if (action.kind !== 'move') return wallError(state, action);
  return legalDestinations(state).some(([row, col]) => row === action.row && col === action.col) ? null : 'illegal-move';
}

export function applyAction(state, input) {
  const action = parseAction(input);
  const reason = actionError(state, action);
  if (reason) {
    const error = new Error(`${notation(action)}: ${reason}`);
    error.code = reason;
    throw error;
  }
  const next = {
    pawns: state.pawns.map(pawn => [...pawn]),
    walls: state.walls.map(wall => [...wall]),
    remaining: [...state.remaining],
    turn: 3 - state.turn,
    winner: null,
    history: [...state.history, notation(action)],
  };
  if (action.kind === 'move') {
    next.pawns[state.turn - 1] = [action.row, action.col];
    if (action.row === (state.turn === 1 ? 0 : 8)) next.winner = state.turn;
  } else {
    next.walls.push([action.kind, action.row, action.col]);
    next.walls.sort((a, b) => a[0].localeCompare(b[0]) || a[1] - b[1] || a[2] - b[2]);
    next.remaining[state.turn - 1]--;
  }
  return next;
}

export function replay(moves) {
  let state = initialState();
  for (let index = 0; index < moves.length; index++) {
    try {
      state = applyAction(state, moves[index]);
    } catch (cause) {
      const error = new Error(`Invalid history at ply ${index + 1}: ${cause.message}`);
      error.code = cause.code;
      throw error;
    }
  }
  return state;
}
export const stateFromMoves = replay;

export function legalActions(state) {
  if (state.winner !== null) return [];
  const actions = legalDestinations(state).map(([row, col]) => ({ kind: 'move', row, col }));
  if (state.remaining[state.turn - 1]) {
    for (const kind of ['h', 'v']) {
      for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
          const action = { kind, row, col };
          if (!wallError(state, action)) actions.push(action);
        }
      }
    }
  }
  return actions;
}

export function actionMask(state) {
  const mask = Array(ACTION_COUNT).fill(0);
  for (const action of legalActions(state)) mask[actionId(action)] = 1;
  return mask;
}

export function observation(state, maxPlies = 256) {
  const board = Array.from({ length: 8 }, () => Array.from({ length: 9 }, () => Array(9).fill(0)));
  state.pawns.forEach(([row, col], player) => { board[player][row][col] = 1; });
  for (const [kind, row, col] of state.walls) {
    for (let offset = 0; offset < 2; offset++) {
      board[kind === 'h' ? 2 : 3][row + (kind === 'v' ? offset : 0)][col + (kind === 'h' ? offset : 0)] = 1;
    }
  }
  const planes = [[4, state.remaining[0] / 10], [5, state.remaining[1] / 10], [6, state.turn - 1], [7, Math.min(1, state.history.length / maxPlies)]];
  for (const [index, value] of planes) for (const row of board[index]) row.fill(value);
  return { board, action_mask: actionMask(state) };
}
