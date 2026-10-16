"""
Network input planes for the Tafl games (Brandubh, Tablut, Hnefatafl).

A FeatureSpec says which planes a network gets. The plane order is fixed:

    [attackers, defenders, king] for the current position,
    [attackers, defenders, king] for the previous position, ... (history_length positions in all;
                                                                missing ones are zero planes)
    [current player]                                           (0 = attackers, 1 = defenders)
    [ones, corners, throne]                                    if constant_planes
    [reach_attackers, reach_defenders, reach_king]             if mobility_planes
    [king_region]                                              if region_plane

With history_length = 1 and nothing else, this is the classic 4-plane input of the older networks.
Every plane holds only 0 and 1, so the replay buffer can store states as uint8.

The extra planes are computed from the board by the game engine, which is cheap compared with a
network call, and hand the network facts that a small convolutional network finds hard to compute
on its own:
- constant planes tell it where the board ends and where the special squares are (zero padding
  cannot tell an off-board square from an empty one);
- reach planes mark the squares each side can move to this turn (pseudo-legal moves, i.e. without
  the repetition rule), which is the first step of every capture, block and escape threat;
- the king-region plane marks the squares connected to the king without crossing an attacker
  (the Hnefatafl encirclement rule asks whether this region touches the edge).
A browser implementation must build the same planes in the same order; see plane_names().
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

# Piece codes, the same in all game variants
EMPTY, ATTACKER, DEFENDER, KING = 0, 1, 2, 3

PIECE_PLANE_NAMES = ("attackers", "defenders", "king")


@dataclass
class FeatureSpec:
    """Which input planes a network gets (see the module docstring for the order)."""
    history_length: int = 1
    constant_planes: bool = False
    mobility_planes: bool = False
    region_plane: bool = False

    def __post_init__(self):
        if self.history_length < 1:
            raise ValueError("history_length must be at least 1")

    def plane_names(self) -> List[str]:
        """Names of the planes, in input order."""
        names = []
        for t in range(self.history_length):
            names += [f"{piece}_t{t}" for piece in PIECE_PLANE_NAMES]
        names.append("player")
        if self.constant_planes:
            names += ["ones", "corners", "throne"]
        if self.mobility_planes:
            names += ["reach_attackers", "reach_defenders", "reach_king"]
        if self.region_plane:
            names.append("king_region")
        return names

    @property
    def num_planes(self) -> int:
        return len(self.plane_names())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSpec":
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown feature settings: {sorted(unknown)}")
        return cls(**data)


# ---------------------------------------------------------------------------------------------
# Plane computations
# ---------------------------------------------------------------------------------------------

def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """out[r + dr, c + dc] = a[r, c]; squares shifted in from outside the board are 0."""
    n = a.shape[0]
    out = np.zeros_like(a)
    out[max(dr, 0):n + min(dr, 0), max(dc, 0):n + min(dc, 0)] = \
        a[max(-dr, 0):n + min(-dr, 0), max(-dc, 0):n + min(-dc, 0)]
    return out


DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _reachable(sources: np.ndarray, landable: np.ndarray, passable: np.ndarray) -> np.ndarray:
    """
    Squares that some piece in `sources` can move to along a straight line: it may land on
    `landable` squares and continue past `passable` squares (both exclude occupied squares).
    """
    n = sources.shape[0]
    reach = np.zeros_like(sources)
    for dr, dc in DIRECTIONS:
        # moving[x] = a source piece stands k steps behind x and every square between is passable
        moving = sources
        for _ in range(n - 1):
            moving = _shift(moving, dr, dc)
            reach |= moving & landable
            moving = moving & passable
            if not moving.any():
                break
    return reach


def special_squares(game):
    """Boolean (corner, throne) masks of a game; the throne mask is empty when the throne is disabled."""
    n = game.board.shape[0]
    corners = np.zeros((n, n), dtype=bool)
    for r, c in game.corners:
        corners[r, c] = True
    throne = np.zeros((n, n), dtype=bool)
    if game.throne_enabled:
        throne[game.throne] = True
    return corners, throne


def constant_planes(game) -> np.ndarray:
    """[ones, corners, throne] as float planes."""
    corners, throne = special_squares(game)
    return np.stack([np.ones_like(corners), corners, throne]).astype(np.float32)


def reach_planes(game) -> np.ndarray:
    """
    [reach_attackers, reach_defenders, reach_king]: the squares each side can move to this turn,
    ignoring whose turn it is and the repetition rule. reach_defenders includes the king's moves.
    Pieces other than the king may pass over the empty throne but not land on it, and may not land
    on a corner; the king may land anywhere empty. This equals the game classes' _get_piece_moves.
    """
    board = game.board
    empty = board == EMPTY
    corners, throne = special_squares(game)
    piece_landable = empty & ~corners & ~throne
    king_reach = _reachable(board == KING, empty, empty)
    attacker_reach = _reachable(board == ATTACKER, piece_landable, empty)
    defender_reach = _reachable(board == DEFENDER, piece_landable, empty) | king_reach
    return np.stack([attacker_reach, defender_reach, king_reach]).astype(np.float32)


def king_region_plane(game) -> np.ndarray:
    """
    Squares connected to the king through squares that hold no attacker (the king's own square
    included). All zero when the king is off the board. Encirclement means this region does not
    reach the edge.
    """
    board = game.board
    region = board == KING
    if not region.any():
        return region.astype(np.float32)
    open_squares = board != ATTACKER
    while True:
        grown = region.copy()
        for dr, dc in DIRECTIONS:
            grown |= _shift(region, dr, dc)
        grown &= open_squares
        if np.array_equal(grown, region):
            return region.astype(np.float32)
        region = grown


def build_features(game, spec: FeatureSpec, history=None) -> np.ndarray:
    """
    The network input of a position, shape (spec.num_planes, N, N), float32.

    Args:
        game: the position (any Tafl game class)
        spec: which planes to build
        history: a StateHistory whose newest entry is this position, for history_length > 1. If
                 None, the history is rebuilt from game.position_history.
    """
    if spec.history_length > 1:
        if history is None:
            from network import StateHistory  # local import: network.py needs torch
            history = StateHistory.from_game(game, spec.history_length)
        planes = [history.get_state_with_history(game.current_player)]
    else:
        planes = [game.get_state()]
    if spec.constant_planes:
        planes.append(constant_planes(game))
    if spec.mobility_planes:
        planes.append(reach_planes(game))
    if spec.region_plane:
        planes.append(king_region_plane(game)[None])
    return np.concatenate(planes, axis=0) if len(planes) > 1 else planes[0]
