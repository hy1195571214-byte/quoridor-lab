"""Equivalent training masks using 81-bit reachability and existing-path certificates.

The public rules engine remains the reference implementation. A candidate wall
which misses a player's existing goal path cannot disconnect that player.
Only candidates touching that path need another reachability search.
"""
import numpy as np

INPUT_PIPELINE = "bitset-path-v1"

ALL = (1 << 81) - 1
ANCHORS = (1 << 64) - 1
TOP = (1 << 9) - 1
BOTTOM = TOP << 72
LEFT = sum(1 << (row * 9) for row in range(9))
RIGHT = LEFT << 8
ANCHOR_LEFT = sum(1 << (row * 8) for row in range(8))
ANCHOR_RIGHT = ANCHOR_LEFT << 7
H_EDGES = tuple((1 << (row * 9 + col)) | (1 << (row * 9 + col + 1))
                for row in range(8) for col in range(8))
V_EDGES = tuple((1 << (row * 9 + col)) | (1 << ((row + 1) * 9 + col))
                for row in range(8) for col in range(8))


def _neighbors(frontier, edges):
    up, down, left, right = edges
    return ((frontier & up) >> 9) | ((frontier & down) << 9) | ((frontier & left) >> 1) | ((frontier & right) << 1)


def _reachable(start, goal, edges):
    frontier = seen = start
    while frontier:
        if frontier & goal:
            return True
        frontier = _neighbors(frontier, edges) & ~seen
        seen |= frontier
    return False


def _goal_path(start, goal, edges):
    frontier = seen = start
    layers = [start]
    while not frontier & goal:
        frontier = _neighbors(frontier, edges) & ~seen
        if not frontier:
            return None
        seen |= frontier
        layers.append(frontier)
    current = frontier & goal
    current &= -current
    vertical = horizontal = 0
    for layer in reversed(layers[:-1]):
        previous = _neighbors(current, edges) & layer
        previous &= -previous
        a, b = current.bit_length() - 1, previous.bit_length() - 1
        if abs(a - b) == 9:
            vertical |= 1 << min(a, b)
        else:
            horizontal |= 1 << min(a, b)
        current = previous
    return vertical, horizontal


def fast_action_mask(state):
    mask = np.zeros(209, dtype=np.int8)
    if state.winner is not None:
        return mask
    for row, col in state.legal_destinations():
        mask[row * 9 + col] = 1
    if state.remaining[state.turn - 1] <= 0:
        return mask
    up, down, left, right = ALL ^ TOP, ALL ^ BOTTOM, ALL ^ LEFT, ALL ^ RIGHT
    h_walls = v_walls = 0
    for kind, row, col in state.walls:
        anchor = row * 8 + col
        if kind == "h":
            h_walls |= 1 << anchor
            down &= ~H_EDGES[anchor]
            up &= ~(H_EDGES[anchor] << 9)
        else:
            v_walls |= 1 << anchor
            right &= ~V_EDGES[anchor]
            left &= ~(V_EDGES[anchor] << 1)
    edges = up, down, left, right
    starts = tuple(1 << (row * 9 + col) for row, col in state.pawns)
    goals = TOP, BOTTOM
    paths = tuple(_goal_path(start, goal, edges) for start, goal in zip(starts, goals))
    if any(path is None for path in paths):
        return mask
    forbidden_h = h_walls | ((h_walls & ~ANCHOR_LEFT) >> 1) | ((h_walls & ~ANCHOR_RIGHT) << 1) | v_walls
    forbidden_v = v_walls | (v_walls >> 8) | ((v_walls << 8) & ANCHORS) | h_walls
    for kind, forbidden, footprints, action_base in (("h", forbidden_h, H_EDGES, 81), ("v", forbidden_v, V_EDGES, 145)):
        candidates = ANCHORS & ~forbidden
        while candidates:
            bit = candidates & -candidates
            candidates ^= bit
            anchor = bit.bit_length() - 1
            removed = footprints[anchor]
            component = 0 if kind == "h" else 1
            affected = tuple(bool(path[component] & removed) for path in paths)
            if not any(affected):
                mask[action_base + anchor] = 1
                continue
            candidate = ((up & ~(removed << 9), down & ~removed, left, right) if kind == "h" else
                         (up, down, left & ~(removed << 1), right & ~removed))
            if all(not changed or _reachable(start, goal, candidate)
                   for changed, start, goal in zip(affected, starts, goals)):
                mask[action_base + anchor] = 1
    return mask


def training_observation(state, max_plies=256):
    return {"board": state.board_planes(max_plies), "action_mask": fast_action_mask(state)}
