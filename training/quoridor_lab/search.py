"""Quoridor search policies: easy (greedy baseline) and medium/hard (minimax).

Contract: ``SearchAgent.choose(state) -> Decision``.  All search actions are
taken from ``state.legal_actions()``; distances use ``state.distance`` only.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass

from quoridor_lab.opponents import Decision
from quoridor_lab.rules import Action, Square, State, _blocked, _inside

WIN_SCORE = 10_000
UNREACHABLE = 999
BLOCKED_MARGIN = 8_000


@dataclass(frozen=True)
class DifficultyConfig:
    name: str
    mode: str  # "greedy" | "minimax"
    min_depth: int = 1
    max_depth: int = 1
    wall_path_filter: bool = False
    use_action_order: bool = False
    random_prob: float = 0.0
    max_wall_candidates: int = 0
    time_limit_seconds: float | None = None

    def __post_init__(self):
        if self.mode not in {"greedy", "minimax"}:
            raise ValueError("mode must be greedy or minimax")
        if not 0 <= self.random_prob <= 1:
            raise ValueError("random_prob must be in [0, 1]")
        if self.mode == "minimax" and self.min_depth < 1:
            raise ValueError("min_depth must be >= 1 for minimax")


DIFFICULTY_PRESETS: dict[str, DifficultyConfig] = {
    "easy": DifficultyConfig(
        name="easy",
        mode="greedy",
        random_prob=0.15,
    ),
    "medium": DifficultyConfig(
        name="medium",
        mode="minimax",
        min_depth=2,
        max_depth=3,
        wall_path_filter=True,
        use_action_order=False,
        max_wall_candidates=16,
    ),
    "hard": DifficultyConfig(
        name="hard",
        mode="minimax",
        min_depth=3,
        max_depth=4,
        wall_path_filter=True,
        use_action_order=True,
        max_wall_candidates=16,
        time_limit_seconds=2.0,
    ),
}


class TimeBudgetExceeded(Exception):
    """Raised when iterative deepening runs out of time (handled internally)."""


def _distance_margin(state: State, me: int, opp: int) -> float:
    d_me, d_opp = state.distance(me), state.distance(opp)
    if d_me >= UNREACHABLE and d_opp >= UNREACHABLE:
        return 0.0
    if d_me >= UNREACHABLE:
        return -float(BLOCKED_MARGIN)
    if d_opp >= UNREACHABLE:
        return float(BLOCKED_MARGIN)
    return float(d_opp - d_me)


def _leaf_value(state: State) -> float:
    """score = distance(对手) - distance(当前行动方) at ``state.turn``."""
    me, opp = state.turn, 3 - state.turn
    return _distance_margin(state, me, opp)


def _terminal_value(state: State, root_player: int) -> float | None:
    if state.winner is None:
        return None
    return float(WIN_SCORE if state.winner == root_player else -WIN_SCORE)


# --- shortest-path region (wall pruning only; not used for scoring) -------------

def _goal_row(player: int) -> int:
    return 0 if player == 1 else 8


def _bfs_forward(state: State, player: int) -> dict[Square, int]:
    start = state.pawns[player - 1]
    goal_row = _goal_row(player)
    blocked = _blocked(state.walls)
    dist: dict[Square, int] = {start: 0}
    queue: deque[Square] = deque([start])
    while queue:
        p = queue.popleft()
        if p[0] == goal_row:
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (p[0] + dr, p[1] + dc)
            if _inside(q) and q not in dist and frozenset((p, q)) not in blocked:
                dist[q] = dist[p] + 1
                queue.append(q)
    return dist


def _bfs_backward(state: State, player: int) -> dict[Square, int]:
    goal_row = _goal_row(player)
    blocked = _blocked(state.walls)
    dist: dict[Square, int] = {}
    queue: deque[Square] = deque()
    for col in range(9):
        g = (goal_row, col)
        dist[g] = 0
        queue.append(g)
    while queue:
        p = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (p[0] + dr, p[1] + dc)
            if _inside(q) and frozenset((q, p)) not in blocked:
                if q not in dist:
                    dist[q] = dist[p] + 1
                    queue.append(q)
    return dist


def _shortest_path_squares(state: State, player: int) -> set[Square]:
    best = state.distance(player)
    if best >= UNREACHABLE:
        return {state.pawns[player - 1]}
    forward = _bfs_forward(state, player)
    backward = _bfs_backward(state, player)
    result: set[Square] = set()
    for square, steps in forward.items():
        back = backward.get(square, UNREACHABLE)
        if back < UNREACHABLE and steps + back == best:
            result.add(square)
    return result or {state.pawns[player - 1]}


def _path_adjacent_region(state: State) -> set[Square]:
    path: set[Square] = set()
    for player in (1, 2):
        path |= _shortest_path_squares(state, player)
    region = set(path)
    for row, col in path:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + dr, col + dc)
            if _inside(neighbor):
                region.add(neighbor)
    return region


def _wall_cells(action: Action) -> set[Square]:
    k, r, c = action.kind, action.row, action.col
    if k == "h":
        return {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
    return {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}


def _wall_gain(state: State, action: Action) -> float:
    me, opp = state.turn, 3 - state.turn
    before = _distance_margin(state, me, opp)
    after = _distance_margin(state.apply(action), me, opp)
    return after - before


def _rank_walls(state: State, walls: list[Action], limit: int) -> list[Action]:
    scored = sorted(((-_wall_gain(state, w), w.notation, w) for w in walls))
    return [w for *_, w in scored[:limit]]


def candidate_actions(state: State, config: DifficultyConfig) -> list[Action]:
    legal = state.legal_actions()
    moves = [a for a in legal if a.kind == "move"]
    walls = [a for a in legal if a.kind != "move"]
    if config.wall_path_filter:
        region = _path_adjacent_region(state)
        near = [w for w in walls if _wall_cells(w) & region]
        walls = near if near else walls
    if config.max_wall_candidates and len(walls) > config.max_wall_candidates:
        walls = _rank_walls(state, walls, config.max_wall_candidates)
    return moves + walls


# --- minimax -------------------------------------------------------------------

@dataclass
class SearchStats:
    nodes: int = 0
    cutoffs: int = 0
    completed_depth: int = 0


def _order_actions(state: State, actions: list[Action], root_player: int) -> list[Action]:
    moves = [a for a in actions if a.kind == "move"]
    walls = [a for a in actions if a.kind != "move"]

    def move_score(action: Action) -> float:
        child = state.apply(action)
        terminal = _terminal_value(child, root_player)
        return terminal if terminal is not None else _leaf_value(child)

    moves.sort(key=lambda a: (-move_score(a), a.notation))
    walls.sort(key=lambda a: (-_wall_gain(state, a), a.notation))
    return moves + walls


def _prepare_actions(state: State, config: DifficultyConfig, root_player: int) -> list[Action]:
    actions = candidate_actions(state, config)
    if config.use_action_order:
        return _order_actions(state, actions, root_player)
    moves = [a for a in actions if a.kind == "move"]
    walls = [a for a in actions if a.kind != "move"]
    return moves + walls


def _check_time(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeBudgetExceeded()


def _search(
    state: State,
    depth: int,
    alpha: float,
    beta: float,
    root_player: int,
    config: DifficultyConfig,
    stats: SearchStats,
    deadline: float | None,
) -> float:
    _check_time(deadline)
    stats.nodes += 1
    terminal = _terminal_value(state, root_player)
    if terminal is not None:
        return terminal
    if depth == 0:
        return _leaf_value(state)

    maximizing = state.turn == root_player
    actions = _prepare_actions(state, config, root_player)
    if not actions:
        return _leaf_value(state)

    if maximizing:
        value = -math.inf
        for action in actions:
            _check_time(deadline)
            value = max(
                value,
                _search(
                    state.apply(action), depth - 1, alpha, beta, root_player, config, stats, deadline
                ),
            )
            alpha = max(alpha, value)
            if beta <= alpha:
                stats.cutoffs += 1
                break
        return value

    value = math.inf
    for action in actions:
        _check_time(deadline)
        value = min(
            value,
            _search(
                state.apply(action), depth - 1, alpha, beta, root_player, config, stats, deadline
            ),
        )
        beta = min(beta, value)
        if beta <= alpha:
            stats.cutoffs += 1
            break
    return value


def _search_root(
    state: State, depth: int, config: DifficultyConfig, deadline: float | None
) -> tuple[Action, SearchStats, float]:
    root_player = state.turn
    stats = SearchStats(completed_depth=depth)
    actions = _prepare_actions(state, config, root_player)
    if not actions:
        raise ValueError("No legal actions in a non-terminal position")

    best_action, best_value = actions[0], -math.inf
    alpha, beta = -math.inf, math.inf
    for action in actions:
        _check_time(deadline)
        value = _search(
            state.apply(action), depth - 1, alpha, beta, root_player, config, stats, deadline
        )
        if value > best_value:
            best_value, best_action = value, action
        alpha = max(alpha, best_value)
    return best_action, stats, best_value


def _choose_minimax(
    state: State, config: DifficultyConfig
) -> tuple[Action, SearchStats, float | None]:
    deadline = (
        time.monotonic() + config.time_limit_seconds if config.time_limit_seconds else None
    )
    legal = state.legal_actions()
    fallback = legal[0]
    if any(a.kind == "move" for a in legal):
        moves = [a for a in legal if a.kind == "move"]
        best = min(state.apply(a).distance(state.turn) for a in moves)
        fallback = next(a for a in moves if state.apply(a).distance(state.turn) == best)

    best_action, best_stats, best_score = fallback, SearchStats(), None
    for depth in range(config.min_depth, config.max_depth + 1):
        try:
            action, stats, score = _search_root(state, depth, config, deadline)
            best_action, best_stats, best_score = action, stats, score
            best_stats.completed_depth = depth
        except TimeBudgetExceeded:
            break
    return best_action, best_stats, best_score


def _choose_greedy(state: State, config: DifficultyConfig, rng: random.Random) -> Action:
    legal = state.legal_actions()
    if config.random_prob and rng.random() < config.random_prob:
        return rng.choice(legal)
    moves = [a for a in legal if a.kind == "move"]
    pool = moves if moves else legal
    best_dist = min(state.apply(a).distance(state.turn) for a in pool)
    return rng.choice([a for a in pool if state.apply(a).distance(state.turn) == best_dist])


class SearchAgent:
    difficulties = frozenset(DIFFICULTY_PRESETS)

    def __init__(self, difficulty: str = "medium", *, seed: int = 0, config: DifficultyConfig | None = None):
        if config is not None:
            self.config = config
        elif difficulty in DIFFICULTY_PRESETS:
            self.config = DIFFICULTY_PRESETS[difficulty]
        else:
            raise ValueError(f"Unknown difficulty {difficulty!r}")
        self.difficulty = self.config.name
        self.rng = random.Random(seed)

    def choose(self, state: State) -> Decision:
        if state.winner is not None:
            raise ValueError("Cannot choose after the game has ended")
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available")

        start = time.monotonic()
        if self.config.mode == "greedy":
            action = _choose_greedy(state, self.config, self.rng)
            stats_meta = {"mode": "greedy", "random_fallback": False}
        else:
            action, stats, score = _choose_minimax(state, self.config)
            stats_meta = {
                "mode": "minimax",
                "nodes": stats.nodes,
                "cutoffs": stats.cutoffs,
                "score": score,
                "completed_depth": stats.completed_depth,
            }

        legal_ids = {a.id for a in legal}
        if action.id not in legal_ids:
            raise RuntimeError(f"Search produced action outside legal set: {action.notation}")
        if state.error(action):
            raise RuntimeError(f"Search produced illegal action {action.notation}")

        return Decision(
            action,
            f"search:{self.difficulty}",
            time.monotonic() - start,
            metadata=stats_meta,
        )

    def close(self) -> None:
        pass