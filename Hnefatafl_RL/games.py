"""
Registry of the game variants: the game class, move encoder, board size and the legacy network
class of each. Training and the tools look variants up here by name or by board size, so no file
needs its own if/elif over the three games.
"""

from dataclasses import dataclass

from brandubh import Brandubh
from tablut import Tablut
from hnefatafl import Hnefatafl


@dataclass(frozen=True)
class GameSpec:
    name: str
    game_class: type
    board_size: int
    move_encoder_module: str  # the move encoders live in the torch-dependent network modules, so
    move_encoder_name: str    # they are imported on demand
    legacy_network_name: str  # the network class of checkpoints written before the config system

    @property
    def move_encoder_class(self):
        import importlib
        return getattr(importlib.import_module(self.move_encoder_module), self.move_encoder_name)

    @property
    def legacy_network_class(self):
        import importlib
        return getattr(importlib.import_module(self.move_encoder_module), self.legacy_network_name)

    @property
    def policy_size(self) -> int:
        return self.board_size * self.board_size * 4 * (self.board_size - 1)


GAMES = {
    "brandubh": GameSpec("brandubh", Brandubh, 7, "network", "MoveEncoder", "BrandubhNet"),
    "tablut": GameSpec("tablut", Tablut, 9, "network_tablut", "TablutMoveEncoder", "TablutNet"),
    "hnefatafl": GameSpec("hnefatafl", Hnefatafl, 11, "network_hnefatafl", "HnefataflMoveEncoder",
                          "HnefataflNet"),
}


def game_spec(name: str) -> GameSpec:
    try:
        return GAMES[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown game '{name}'; choose one of {sorted(GAMES)}") from None


def game_spec_for_board(board_size: int) -> GameSpec:
    for spec in GAMES.values():
        if spec.board_size == board_size:
            return spec
    raise ValueError(f"No game variant has a {board_size}x{board_size} board")
