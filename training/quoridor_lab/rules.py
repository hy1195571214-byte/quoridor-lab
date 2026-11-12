"""Independent, strict 9x9 rules. Coordinates: row 0 at the top, col 0 at a."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache
import re

import numpy as np

ACTION_COUNT = 209  # 81 destinations + 64 horizontal + 64 vertical wall anchors
Square = tuple[int, int]
Wall = tuple[str, int, int]


@dataclass(frozen=True)
class Action:
    kind: str
    row: int
    col: int

    def __post_init__(self):
        limit = 9 if self.kind == "move" else 8
        if self.kind not in {"move", "h", "v"} or not all(
            type(n) is int and 0 <= n < limit for n in (self.row, self.col)
        ):
            raise ValueError("Action coordinates or kind are invalid")

    @property
    def id(self) -> int:
        return self.row * 9 + self.col if self.kind == "move" else (
            81 + (64 if self.kind == "v" else 0) + self.row * 8 + self.col
        )

    @property
    def notation(self) -> str:
        return f"{chr(97 + self.col)}{9 - self.row}" if self.kind == "move" else (
            f"{self.kind}{chr(97 + self.col)}{8 - self.row}"
        )

    @classmethod
    def from_id(cls, action_id: int) -> Action:
        if type(action_id) is not int or not 0 <= action_id < ACTION_COUNT:
            raise ValueError("action_id must be an integer in [0, 208]")
        if action_id < 81:
            return cls("move", *divmod(action_id, 9))
        offset = action_id - 81
        return cls("h" if offset < 64 else "v", *divmod(offset % 64, 8))

    @classmethod
    def parse(cls, text: str) -> Action:
        if not isinstance(text, str):
            raise ValueError("Action notation must be a string")
        if re.fullmatch(r"[a-i][1-9]", text):
            return cls("move", 9 - int(text[1]), ord(text[0]) - 97)
        if re.fullmatch(r"[a-h][1-8][hv]", text):
            text = text[2] + text[:2]
        if re.fullmatch(r"[hv][a-h][1-8]", text):
            return cls(text[0], 8 - int(text[2]), ord(text[1]) - 97)
        raise ValueError(f"Invalid action notation: {text!r}")


def _inside(p: Square) -> bool:
    return 0 <= p[0] < 9 and 0 <= p[1] < 9


@lru_cache(maxsize=4096)
def _blocked(walls: tuple[Wall, ...]) -> frozenset:
    result = set()
    for kind, row, col in walls:
        for d in (0, 1):
            a, b = ((row, col + d), (row + 1, col + d)) if kind == "h" else (
                (row + d, col), (row + d, col + 1)
            )
            result.add(frozenset((a, b)))
    return frozenset(result)


@lru_cache(maxsize=32768)
def _distance(walls: tuple[Wall, ...], start: Square, goal_row: int) -> int:
    """Wall reachability ignores temporary pawn occupancy, as required by the rules."""
    queue = deque([(start, 0)])
    seen = {start}
    blocked = _blocked(walls)
    while queue:
        p, dist = queue.popleft()
        if p[0] == goal_row:
            return dist
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (p[0] + dr, p[1] + dc)
            if _inside(q) and q not in seen and frozenset((p, q)) not in blocked:
                seen.add(q)
                queue.append((q, dist + 1))
    return 999


@dataclass(frozen=True)
class State:
    pawns: tuple[Square, Square] = ((8, 4), (0, 4))
    walls: tuple[Wall, ...] = ()
    remaining: tuple[int, int] = (10, 10)
    turn: int = 1
    winner: int | None = None
    history: tuple[str, ...] = ()

    def distance(self, player: int) -> int:
        return _distance(self.walls, self.pawns[player - 1], 0 if player == 1 else 8)

    def legal_destinations(self, player: int | None = None) -> set[Square]:
        player = player or self.turn
        p, opponent = self.pawns[player - 1], self.pawns[2 - player]
        blocked = _blocked(self.walls)
        free = lambda a, b: _inside(b) and frozenset((a, b)) not in blocked
        moves = set()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (p[0] + dr, p[1] + dc)
            if not free(p, q):
                continue
            if q != opponent:
                moves.add(q)
                continue
            jump = (q[0] + dr, q[1] + dc)
            if free(q, jump):
                moves.add(jump)
            else:
                for sr, sc in ((-dc, dr), (dc, -dr)):
                    side = (q[0] + sr, q[1] + sc)
                    if free(q, side):
                        moves.add(side)
        return moves

    def wall_error(self, action: Action) -> str | None:
        if self.remaining[self.turn - 1] <= 0:
            return "no-walls"
        k, r, c = action.kind, action.row, action.col
        for wk, wr, wc in self.walls:
            if wk != k and (wr, wc) == (r, c):
                return "crossing"
            if wk == k and ((k == "h" and wr == r and abs(wc - c) < 2) or (
                k == "v" and wc == c and abs(wr - r) < 2
            )):
                return "overlap"
        candidate = tuple(sorted((*self.walls, (k, r, c))))
        if any(_distance(candidate, self.pawns[p - 1], 0 if p == 1 else 8) == 999 for p in (1, 2)):
            return "blocks-path"
        return None

    def error(self, action: Action) -> str | None:
        if self.winner is not None:
            return "game-over"
        if action.kind == "move":
            return None if (action.row, action.col) in self.legal_destinations() else "illegal-move"
        return self.wall_error(action)

    def legal_actions(self) -> list[Action]:
        if self.winner is not None:
            return []
        moves = [Action("move", r, c) for r, c in sorted(self.legal_destinations())]
        if self.remaining[self.turn - 1]:
            moves += [a for k in ("h", "v") for r in range(8) for c in range(8)
                      if self.wall_error(a := Action(k, r, c)) is None]
        return moves

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        for action in self.legal_actions():
            mask[action.id] = 1
        return mask

    def apply(self, action: Action | str | int) -> State:
        if isinstance(action, str):
            action = Action.parse(action)
        elif type(action) is int:
            action = Action.from_id(action)
        reason = self.error(action)
        if reason:
            raise ValueError(f"{action.notation}: {reason}")
        pawns, walls, remaining = list(self.pawns), self.walls, list(self.remaining)
        winner = None
        if action.kind == "move":
            pawns[self.turn - 1] = (action.row, action.col)
            if action.row == (0 if self.turn == 1 else 8):
                winner = self.turn
        else:
            walls = tuple(sorted((*walls, (action.kind, action.row, action.col))))
            remaining[self.turn - 1] -= 1
        return State(tuple(pawns), walls, tuple(remaining), 3 - self.turn, winner,
                     (*self.history, action.notation))

    @classmethod
    def from_moves(cls, moves) -> State:
        state = cls()
        for index, move in enumerate(moves):
            try:
                state = state.apply(move)
            except ValueError as exc:
                raise ValueError(f"Invalid history at ply {index + 1}: {exc}") from exc
        return state

    def observation(self, max_plies=256) -> dict:
        board = np.zeros((8, 9, 9), dtype=np.float32)
        for i, (r, c) in enumerate(self.pawns):
            board[i, r, c] = 1
        for k, r, c in self.walls:
            if k == "h":
                board[2, r, c:c + 2] = 1
            else:
                board[3, r:r + 2, c] = 1
        board[4].fill(self.remaining[0] / 10)
        board[5].fill(self.remaining[1] / 10)
        board[6].fill(self.turn - 1)
        board[7].fill(min(1, len(self.history) / max_plies))
        return {"board": board, "action_mask": self.action_mask()}

    def to_dict(self) -> dict:
        square = lambda p: f"{chr(97 + p[1])}{9 - p[0]}"
        return {"pawns": list(map(square, self.pawns)), "walls": [list(w) for w in self.walls],
                "remaining_walls": list(self.remaining), "turn": self.turn,
                "winner": self.winner, "finished_reason": "goal" if self.winner else None,
                "ply": len(self.history), "moves": list(self.history)}
