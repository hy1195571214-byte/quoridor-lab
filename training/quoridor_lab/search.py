"""Minimax search opponent — ``choose(State) -> Decision``, no training deps.

All candidate actions come from ``state.legal_actions()``. Path length uses
``state.distance`` only (rules.py BFS). ``distance == 999`` is mapped to a large
finite score so evaluation never returns NaN.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from quoridor_lab.opponents import Decision
from quoridor_lab.rules import Action, State

WIN_SCORE = 10_000
UNREACHABLE = 999
BLOCKED_MARGIN = 8_000


# --- difficulty ----------------------------------------------------------------

@dataclass(frozen=True)
class EvalWeights:
    wall_balance: float = 0.0


@dataclass(frozen=True)
class DifficultyConfig:
    name: str
    depth: int
    wall_filter: bool
    random_prob: float
    max_wall_candidates: int = 0
    eval_weights: EvalWeights = EvalWeights()

    def __post_init__(self):
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if not 0 <= self.random_prob <= 1:
            raise ValueError("random_prob must be in [0, 1]")


DIFFICULTY_PRESETS: dict[str, DifficultyConfig] = {
    "easy": DifficultyConfig(
        name="easy",
        depth=2,
        wall_filter=False,
        random_prob=0.15,
        max_wall_candidates=8,
    ),
    "medium": DifficultyConfig(
        name="medium",
        depth=3,
        wall_filter=True,
        random_prob=0.0,
        max_wall_candidates=10,
    ),
    "hard": DifficultyConfig(
        name="hard",
        depth=4,
        wall_filter=True,
        random_prob=0.0,
        max_wall_candidates=12,
        eval_weights=EvalWeights(wall_balance=0.3),
    ),
}


# --- evaluation ----------------------------------------------------------------

def _distance_margin(state: State, me: int, opp: int) -> float:
    d_me, d_opp = state.distance(me), state.distance(opp)
    if d_me >= UNREACHABLE and d_opp >= UNREACHABLE:
        return 0.0
    if d_me >= UNREACHABLE:
        return -float(BLOCKED_MARGIN)
    if d_opp >= UNREACHABLE:
        return float(BLOCKED_MARGIN)
    return float(d_opp - d_me)


def terminal_value(state: State, root_player: int) -> float | None:
    if state.winner is None:
        return None
    return float(WIN_SCORE if state.winner == root_player else -WIN_SCORE)


def evaluate(state: State, root_player: int, weights: EvalWeights = EvalWeights()) -> float:
    terminal = terminal_value(state, root_player)
    if terminal is not None:
        return terminal
    me, opp = root_player, 3 - root_player
    score = _distance_margin(state, me, opp)
    if weights.wall_balance:
        score += weights.wall_balance * (state.remaining[opp - 1] - state.remaining[me - 1])
    return score


# --- action pruning (subset of legal_actions only) -----------------------------

def _wall_gain(state: State, action: Action) -> float:
    me, opp = state.turn, 3 - state.turn
    before = _distance_margin(state, me, opp)
    after_state = state.apply(action)
    after = _distance_margin(after_state, me, opp)
    return after - before


def _rank_walls(state: State, walls: list[Action], limit: int) -> list[Action]:
    scored = sorted(((-_wall_gain(state, w), w.notation, w) for w in walls))
    return [w for *_, w in scored[:limit]]


def candidate_actions(state: State, config: DifficultyConfig) -> list[Action]:
    legal = state.legal_actions()
    moves = [a for a in legal if a.kind == "move"]
    walls = [a for a in legal if a.kind != "move"]
    if config.wall_filter:
        improved = [w for w in walls if _wall_gain(state, w) > 0]
        walls = improved if improved else walls
    if config.max_wall_candidates and len(walls) > config.max_wall_candidates:
        walls = _rank_walls(state, walls, config.max_wall_candidates)
    return moves + walls


# --- minimax -------------------------------------------------------------------

@dataclass
class SearchStats:
    nodes: int = 0
    cutoffs: int = 0


def _order_actions(state: State, actions: list[Action], root_player: int, config: DifficultyConfig) -> list[Action]:
    moves = [a for a in actions if a.kind == "move"]
    walls = [a for a in actions if a.kind != "move"]

    def after_score(action: Action) -> float:
        child = state.apply(action)
        terminal = terminal_value(child, root_player)
        return terminal if terminal is not None else evaluate(child, root_player, config.eval_weights)

    moves.sort(key=lambda a: (-after_score(a), a.notation))
    walls.sort(key=lambda a: (-_wall_gain(state, a), a.notation))
    return moves + walls


def _search(
    state: State,
    depth: int,
    alpha: float,
    beta: float,
    root_player: int,
    config: DifficultyConfig,
    stats: SearchStats,
) -> float:
    stats.nodes += 1
    terminal = terminal_value(state, root_player)
    if terminal is not None:
        return terminal
    if depth == 0:
        return evaluate(state, root_player, config.eval_weights)

    maximizing = state.turn == root_player
    actions = _order_actions(state, candidate_actions(state, config), root_player, config)
    if not actions:
        return evaluate(state, root_player, config.eval_weights)

    if maximizing:
        value = -math.inf
        for action in actions:
            value = max(
                value,
                _search(state.apply(action), depth - 1, alpha, beta, root_player, config, stats),
            )
            alpha = max(alpha, value)
            if beta <= alpha:
                stats.cutoffs += 1
                break
        return value

    value = math.inf
    for action in actions:
        value = min(
            value,
            _search(state.apply(action), depth - 1, alpha, beta, root_player, config, stats),
        )
        beta = min(beta, value)
        if beta <= alpha:
            stats.cutoffs += 1
            break
    return value


def choose_action(state: State, config: DifficultyConfig) -> tuple[Action, SearchStats, float]:
    root_player = state.turn
    stats = SearchStats()
    actions = _order_actions(state, candidate_actions(state, config), root_player, config)
    if not actions:
        raise ValueError("No legal actions in a non-terminal position")

    best_action, best_value = actions[0], -math.inf
    alpha, beta = -math.inf, math.inf
    for action in actions:
        value = _search(state.apply(action), config.depth - 1, alpha, beta, root_player, config, stats)
        if value > best_value:
            best_value, best_action = value, action
        alpha = max(alpha, best_value)
    return best_action, stats, best_value


# --- public agent --------------------------------------------------------------

class SearchAgent:
    """Alpha-beta Quoridor search; same contract as ``LocalPolicy``."""

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
        if self.config.random_prob and self.rng.random() < self.config.random_prob:
            action = self.rng.choice(legal)
            stats_meta = {"random_fallback": True, "nodes": 0, "cutoffs": 0, "score": None}
        else:
            action, stats, score = choose_action(state, self.config)
            stats_meta = {
                "random_fallback": False,
                "nodes": stats.nodes,
                "cutoffs": stats.cutoffs,
                "score": score,
                "depth": self.config.depth,
            }

        if state.error(action):
            raise RuntimeError(f"Search produced illegal action {action.notation}")
        return Decision(
            action,
            f"search:{self.difficulty}",
            time.monotonic() - start,
            metadata={"difficulty": self.difficulty, **stats_meta},
        )

    def close(self) -> None:
        pass