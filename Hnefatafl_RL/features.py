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
import scipy.ndimage

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

def _nearest_piece_to_the_left(boards: np.ndarray) -> np.ndarray:
    """
    For every square of each board in `boards` (shape (..., N, N)): the piece code of the nearest
    occupied square to its left in the same row (EMPTY if there is none). That piece is the only
    one that can slide onto the square from the left, because every square between them is empty.
    """
    n = boards.shape[-1]
    occupied_column = np.where(boards != EMPTY, np.arange(n), -1)
    # Column of the last occupied square at or left of each square, then shifted one step right
    # to give the last occupied square strictly left of it
    last = np.maximum.accumulate(occupied_column, axis=-1)
    left = np.concatenate([np.full(boards.shape[:-1] + (1,), -1), last[..., :-1]], axis=-1)
    pieces = np.take_along_axis(boards, np.maximum(left, 0), axis=-1)
    return np.where(left >= 0, pieces, EMPTY)


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
    # arriving[k] = the piece that can slide onto each square from one of the four sides: the
    # left-to-right scan runs on the board seen from each side (one numpy pass for all four), and
    # each result is turned back to the board's orientation
    views = np.stack([board, board[:, ::-1], board.T, board.T[:, ::-1]])
    nearest = _nearest_piece_to_the_left(views)
    arriving = np.stack([nearest[0], nearest[1][:, ::-1], nearest[2].T, nearest[3][:, ::-1].T])
    # The empty throne can be passed but only the king may land on it (or on a corner)
    piece_landable = empty & ~corners & ~throne
    king_reach = empty & (arriving == KING).any(axis=0)
    attacker_reach = piece_landable & (arriving == ATTACKER).any(axis=0)
    defender_reach = (piece_landable & (arriving == DEFENDER).any(axis=0)) | king_reach
    return np.stack([attacker_reach, defender_reach, king_reach]).astype(np.float32)


def king_region_plane(game) -> np.ndarray:
    """
    Squares connected to the king through squares that hold no attacker (the king's own square
    included). All zero when the king is off the board. Encirclement means this region does not
    reach the edge.
    """
    board = game.board
    kings = np.argwhere(board == KING)
    if len(kings) == 0:
        return np.zeros(board.shape, dtype=np.float32)
    # Number the connected areas of non-attacker squares (neighbours share a side), then keep the
    # area that holds the king
    areas, _ = scipy.ndimage.label(board != ATTACKER)
    return (areas == areas[tuple(kings[0])]).astype(np.float32)


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
