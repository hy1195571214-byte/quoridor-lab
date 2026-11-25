"""Frozen local Quoridor Level 1 policies, games and relative Elo."""
from dataclasses import dataclass
from functools import lru_cache
import math
import random

import numpy as np
import torch

from .fast_observation import ALL, BOTTOM, H_EDGES, LEFT, RIGHT, TOP, V_EDGES
from .fast_observation import _goal_path, _neighbors, fast_action_mask, training_observation
from .rules import Action, State
from .search import SearchAgent
from .search_demonstrations import SEARCH_LEVELS

POOL = ("level1",) * 7 + ("bfs",) * 2 + ("randomized_search",)
ANCHORS = ("random", "bfs", "level1")
THRESHOLDS = {"random": .95, "bfs": .80, "level1": .70}
MAX_PLIES = 256
POLICIES = {
    "random": {"strategy": "uniform over all legal actions"},
    "bfs": {"strategy": "shortest-distance legal pawn move; seeded ties"},
    "level1": {"depth": 1, "wall_candidates": 6, "wall_penalty": .35, "random_probability": 0},
    "randomized_search": {"depth": 1, "wall_candidates": 6, "wall_penalty": .35, "random_probability": .25},
}
DEEP_SEARCH = {"level2": SEARCH_LEVELS["medium"], "level3": SEARCH_LEVELS["hard"]}
SEARCH_ACTIONS = tuple(Action.from_id(i) for i in range(209))
WALL_CELLS = tuple(0b11000000011 << (row * 9 + col) for row in range(8) for col in range(8))


@lru_cache(maxsize=4096)
def edges_for(walls):
    up, down, left, right = ALL ^ TOP, ALL ^ BOTTOM, ALL ^ LEFT, ALL ^ RIGHT
    for kind, row, col in walls:
        anchor = row * 8 + col
        if kind == "h":
            footprint = H_EDGES[anchor]
            up, down = up & ~(footprint << 9), down & ~footprint
        else:
            footprint = V_EDGES[anchor]
            left, right = left & ~(footprint << 1), right & ~footprint
    return up, down, left, right


def pawn_bit(square):
    return 1 << (square[0] * 9 + square[1])


def bit_distance(start, goal, edges):
    seen = frontier = start
    distance = 0
    while frontier:
        if frontier & goal:
            return distance
        frontier = _neighbors(frontier, edges) & ~seen
        seen |= frontier
        distance += 1
    return 999


@lru_cache(maxsize=32768)
def search_distance_layers(walls, goal):
    """A wall layout can share goal distances across all pawn locations."""
    edges = edges_for(walls)
    layers = []
    frontier = seen = goal
    while frontier:
        layers.append(frontier)
        frontier = _neighbors(frontier, edges) & ~seen
        seen |= frontier
    return tuple(layers)


@dataclass(frozen=True)
class SearchPosition(State):
    """Use the equivalent batched legality mask and bitset distances in search."""

    def distance(self, player):
        start = pawn_bit(self.pawns[player - 1])
        goal = TOP if player == 1 else BOTTOM
        for distance, layer in enumerate(search_distance_layers(self.walls, goal)):
            if start & layer:
                return distance
        return 999

    @lru_cache(maxsize=8192)
    def legal_actions(self):
        return [SEARCH_ACTIONS[code] for code in np.flatnonzero(fast_action_mask(self))]

    def rank_legal_walls(self, walls, limit):
        up, down, left, right = edges = edges_for(self.walls)
        own, other = (pawn_bit(self.pawns[i - 1]) for i in (self.turn, 3 - self.turn))
        own_goal, other_goal = (TOP, BOTTOM) if self.turn == 1 else (BOTTOM, TOP)
        own_path = _goal_path(own, own_goal, edges)
        other_path = _goal_path(other, other_goal, edges)
        own_distance, other_distance = self.distance(self.turn), self.distance(3 - self.turn)

        def rank(wall):
            # Removing edges outside an existing shortest path preserves its distance.
            anchor = wall.row * 8 + wall.col
            if wall.kind == "h":
                removed = H_EDGES[anchor]
                component = 0
                candidate = (up & ~(removed << 9), down & ~removed, left, right)
            else:
                removed = V_EDGES[anchor]
                component = 1
                candidate = (up, down, left & ~(removed << 1), right & ~removed)
            ours = bit_distance(own, own_goal, candidate) if own_path[component] & removed else own_distance
            theirs = bit_distance(other, other_goal, candidate) if other_path[component] & removed else other_distance
            return ours - theirs, wall.notation

        return sorted(walls, key=rank)[:limit]

    def walls_near_shortest_paths(self, walls):
        edges = edges_for(self.walls)
        path = 0
        for square, goal in zip(self.pawns, (TOP, BOTTOM), strict=True):
            start = pawn_bit(square)
            layers = search_distance_layers(self.walls, goal)
            distance = next((i for i, layer in enumerate(layers) if start & layer), None)
            if distance is None:
                path |= start
                continue
            frontier = seen = start
            for layer in reversed(layers[:distance + 1]):
                path |= frontier & layer
                frontier = _neighbors(frontier, edges) & ~seen
                seen |= frontier
        region = path | _neighbors(path, edges_for(()))
        return [wall for wall in walls if WALL_CELLS[wall.row * 8 + wall.col] & region]

    def apply(self, action):
        # Search expands only actions taken from its generated legal list. The
        # actual game still applies the returned action through the rules engine.
        pawns, walls, remaining = list(self.pawns), self.walls, list(self.remaining)
        winner = None
        if action.kind == "move":
            pawns[self.turn - 1] = (action.row, action.col)
            if action.row == (0 if self.turn == 1 else 8):
                winner = self.turn
        else:
            walls = tuple(sorted((*walls, (action.kind, action.row, action.col))))
            remaining[self.turn - 1] -= 1
        return SearchPosition(tuple(pawns), walls, tuple(remaining), 3 - self.turn, winner)


@lru_cache(maxsize=32768)
def depth_search_action(position, name):
    return SearchAgent(config=DEEP_SEARCH[name]).choose(position).action.id


def wall_trials(state, edges, target, goal):
    if not state.remaining[state.turn - 1]:
        return []
    vertical, horizontal = _goal_path(pawn_bit(target), goal, edges)
    row, col = target
    proposals = []
    for boundary, footprints, start in ((vertical, H_EDGES, 81), (horizontal, V_EDGES, 145)):
        for anchor, footprint in enumerate(footprints):
            if boundary & footprint:
                r, c = divmod(anchor, 8)
                proposals.append((abs(r - row) + abs(c - col), start + anchor))
    if not proposals:
        return []
    mask = fast_action_mask(state)
    return [code for _, code in sorted(proposals) if mask[code]][:6]


def bot_action(state, name, seed):
    if name in DEEP_SEARCH:
        position = SearchPosition(state.pawns, state.walls, state.remaining, state.turn, state.winner)
        return depth_search_action(position, name)
    rng = random.Random(seed ^ (len(state.history) * 2654435761))
    if name == "random":
        return int(rng.choice(np.flatnonzero(fast_action_mask(state))))
    edges = edges_for(state.walls)
    own_goal, other_goal = (TOP, BOTTOM) if state.turn == 1 else (BOTTOM, TOP)
    moves = sorted(state.legal_destinations())
    move_distances = [(row * 9 + col, bit_distance(pawn_bit((row, col)), own_goal, edges))
                      for row, col in moves]
    best_dist = min(distance for _, distance in move_distances)
    if best_dist == 0 or name == "bfs":
        return rng.choice([code for code, distance in move_distances if distance == best_dist])
    other = state.pawns[2 - state.turn]
    own = pawn_bit(state.pawns[state.turn - 1])
    other_bit = pawn_bit(other)
    other_dist = bit_distance(other_bit, other_goal, edges)
    candidates = [(code, other_dist - distance) for code, distance in move_distances]
    for code in wall_trials(state, edges, other, other_goal):
        action = Action.from_id(code)
        candidate_edges = edges_for(tuple(sorted((*state.walls, (action.kind, action.row, action.col)))))
        score = bit_distance(other_bit, other_goal, candidate_edges) - bit_distance(own, own_goal, candidate_edges)
        candidates.append((code, score - .35))
    if name == "randomized_search" and rng.random() < .25:
        return rng.choice(candidates)[0]
    best_score = max(score for _, score in candidates)
    return rng.choice([code for code, score in candidates if score == best_score])


@dataclass
class Game:
    state: State
    player: int
    opponent: str
    seed: int
    opening: tuple
    index: int = 0

    @classmethod
    def start(cls, opening, player, opponent, seed, index=0, resolve=bot_action):
        state = State.from_moves(opening)
        if state.turn != player:
            state = state.apply(resolve(state, opponent, seed))
        return cls(state, player, opponent, seed, opening, index)

    def step(self, action, resolve=bot_action):
        self.state = self.state.apply(action)
        result = self.outcome()
        if result is None:
            self.state = self.state.apply(resolve(self.state, self.opponent, self.seed))
            result = self.outcome()
        return result

    def outcome(self):
        if self.state.winner is not None:
            return "wins" if self.state.winner == self.player else "losses"
        if len(self.state.history) >= MAX_PLIES:
            return "truncated"
        return None

    def record(self, result):
        return {"opponent": self.opponent, "seed": self.seed, "learner_player": self.player,
                "result": result, "opening": self.opening, "moves": self.state.history}


def step_batch(games, actions, resolve_batch):
    results = []
    pending = []
    for index, (game, action) in enumerate(zip(games, actions, strict=True)):
        if action is None:
            results.append("illegal")
            continue
        game.state = game.state.apply(action)
        result = game.outcome()
        results.append(result)
        if result is None:
            pending.append(index)
    requests = [(games[index].state, games[index].opponent, games[index].seed) for index in pending]
    for index, reply in zip(pending, resolve_batch(requests), strict=True):
        game = games[index]
        game.state = game.state.apply(reply)
        results[index] = game.outcome()
    return results


def observations(games):
    rows = [training_observation(game.state, MAX_PLIES) for game in games]
    return (torch.from_numpy(np.stack([row["board"] for row in rows])),
            torch.from_numpy(np.stack([row["action_mask"] for row in rows])).bool())


def balance_action_families(logits, mask, strength):
    """Correct for the difference in how many pawn and wall actions are available."""
    if not strength:
        return logits
    pawn_offset = mask[:, :81].sum(dim=1, keepdim=True).clamp_min(1).log() * strength
    wall_offset = mask[:, 81:].sum(dim=1, keepdim=True).clamp_min(1).log() * strength
    return torch.cat((logits[:, :81] - pawn_offset, logits[:, 81:] - wall_offset), dim=1)


def evaluation_games(openings, opponent, seed, resolve=bot_action):
    return [Game.start(opening, player, opponent, seed + index, index, resolve)
            for index, opening in enumerate(openings) for player in (1, 2)]


def play_evaluation(games, device, *, model=None, bot=None, family_balance=0., resolve=bot_action):
    results = []
    if model is not None:
        model.eval()
    with torch.inference_mode():
        while games:
            remaining = []
            for start in range(0, len(games), 96):
                group = games[start:start + 96]
                if model is None:
                    actions = [resolve(game.state, bot, game.seed ^ 40391) for game in group]
                    legal = None
                else:
                    boards, legal = observations(group)
                    device_mask = legal.to(device)
                    logits, _ = model(boards.to(device), device_mask)
                    logits = balance_action_families(logits, device_mask, family_balance)
                    actions = logits.argmax(dim=-1).tolist()
                actions = [action if legal is None or legal[index, action] else None
                           for index, action in enumerate(actions)]
                if (resolve_batch := getattr(resolve, "batch", None)) is not None:
                    outcomes = step_batch(group, actions, resolve_batch)
                else:
                    outcomes = [game.step(action, resolve) if action is not None else "illegal"
                               for game, action in zip(group, actions)]
                for game, result in zip(group, outcomes):
                    if result:
                        results.append(game.record(result))
                    else:
                        remaining.append(game)
            games = remaining
    return results


def score_games(games):
    names = ("wins", "losses", "truncated", "illegal")
    counts = {name: sum(game["result"] == name for game in games) for name in names}
    total = len(games)
    return {**counts, "games": total, "win_rate": counts["wins"] / total,
            "truncation_rate": counts["truncated"] / total, "illegal_rate": counts["illegal"] / total,
            "red_games": sum(game["learner_player"] == 1 for game in games),
            "blue_games": sum(game["learner_player"] == 2 for game in games),
            "wins_red": sum(game["result"] == "wins" and game["learner_player"] == 1 for game in games),
            "wins_blue": sum(game["result"] == "wins" and game["learner_player"] == 2 for game in games)}


def passed(scores, thresholds=THRESHOLDS, truncation_limit=.05, *, truncation_limits=None):
    limits = truncation_limits or {}
    return all(scores[name]["win_rate"] >= threshold and scores[name]["illegal"] == 0
               and scores[name]["truncation_rate"] < limits.get(name, truncation_limit)
               for name, threshold in thresholds.items())


def estimate_elo(matches):
    """Fit a local Bradley–Terry scale with BFS fixed at 1500.

    matches maps (learner, opponent) to counts. Only natural wins/losses count.
    A half-win and half-loss per pairing keep separated fits finite.
    """
    names = sorted({name for pair in matches for name in pair} - {"bfs"})
    indices = {name: index for index, name in enumerate(names)}
    design, won, totals = [], [], []
    for (left, right), score in matches.items():
        row = np.zeros(len(names))
        if left != "bfs":
            row[indices[left]] += 1
        if right != "bfs":
            row[indices[right]] -= 1
        design.append(row)
        won.append(score["wins"] + .5)
        totals.append(score["wins"] + score["losses"] + 1)
    matrix, won, totals = np.asarray(design), np.asarray(won), np.asarray(totals)
    weights = np.zeros(len(names))
    for _ in range(80):
        probabilities = 1 / (1 + np.exp(-np.clip(matrix @ weights, -30, 30)))
        curvature = (totals * probabilities * (1 - probabilities)).clip(1e-9)
        gradient = matrix.T @ (won - totals * probabilities)
        shift = np.linalg.solve((matrix.T * curvature) @ matrix, gradient)
        weights += np.clip(shift, -2, 2)
        if np.max(np.abs(shift)) < 1e-8:
            break
    return {"bfs": 1500., **{name: round(1500 + weights[i] * 400 / math.log(10), 2)
                            for name, i in indices.items()}}


def model_elo(scores, anchor_ratings):
    # Freeze the reference scale as training progresses.
    left, right = -2000., 5000.
    for _ in range(50):
        current = (left + right) / 2
        balance = 0.
        for name, row in scores.items():
            score = 1 / (1 + 10 ** ((anchor_ratings[name] - current) / 400))
            balance += row["wins"] + .5 - (row["wins"] + row["losses"] + 1) * score
        if balance > 0:
            left = current
        else:
            right = current
    return round((left + right) / 2, 2)
