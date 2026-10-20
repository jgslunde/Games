"""
Tafl games on 7x7, 9x9 and 11x11 boards, played under named rule sets.

A rule set (Rules) fixes everything that differs between the common variants: the starting
position, who moves first, where the king escapes to, how he is captured, the throne, and the
extra rules of some sets (encirclement, shieldwall, exit fort, what an endless repetition means).
Training runs use only the named sets in RULESETS: the nine most-played variants, as defined in the
implementation reference of the tafl overview (October 2026). RULESETS.md lists them, their
sources and how balanced they are in human play.

Board: an NxN int8 array with EMPTY 0, ATTACKER 1, DEFENDER 2, KING 3. Players: 0 attackers,
1 defenders. A move is (from_row, from_col, to_row, to_col).

Rules shared by all sets:
- every piece moves like a rook in chess: any number of empty squares along a row or column;
- a piece other than the king is captured when the opponent's move closes a sandwich on two
  opposite sides, by two enemy pieces or by an enemy piece and a hostile square; moving into a
  sandwich is safe;
- only the king may stop on the throne (the centre square, in sets with a throne) and on the
  corners (in sets where the king escapes to a corner); the corners are hostile to all pieces;
- a player who cannot move loses. Overlong games are ended as draws by the code that plays them.

After every move the checks run in this order (reference A7): captures, king on a goal square,
king captured, exit fort (after a defenders' move), encirclement (after an attackers' move),
repetition, no legal move for the side to move.
"""

import copy
import functools
from dataclasses import asdict, dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
import scipy.ndimage

EMPTY, ATTACKER, DEFENDER, KING = 0, 1, 2, 3
ATTACKER_PLAYER, DEFENDER_PLAYER = 0, 1
PIECE_CODES = {".": EMPTY, "A": ATTACKER, "D": DEFENDER, "K": KING}
# Directions in policy order: up, down, left, right
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

KING_CAPTURE_RULES = {
    "weak": "between two attackers like any other piece, everywhere",
    "weak_throne": "like any other piece, but on the throne only from all four sides",
    "weak_near_throne": "like any other piece, but on the throne from all four sides and next to "
                        "it from the three other sides",
    "strong": "from all four sides (next to the throne from the three other sides); never on the "
              "board edge",
}
# What the repetition_count-th occurrence of a position means, whichever side made the move
REPETITION_RULES = {
    "defenders_lose": "a win for the attackers",
    "draw": "a draw",
}


@dataclass(frozen=True)
class Rules:
    """
    One complete rule set.

    Attributes:
        name: key in RULESETS (also the directory name of its configs and checkpoints)
        description: one line naming the published rules this follows
        setup: the starting position, one string per row: A attacker, D defender, K king, . empty
        first_player: "attackers" or "defenders"
        king_goal: "corner" (the king wins on a corner square) or "edge" (on any edge square)
        king_capture: how the king is captured, a key of KING_CAPTURE_RULES
        king_armed: the king can capture, as a moving piece and as the second piece of a sandwich
        throne: the centre square is a throne (only the king may stop on it)
        throne_hostile: the throne closes sandwiches: against attackers always, against defenders
                        when it is empty (the king's own capture rule says how it counts for him)
        encirclement: the attackers win when no defender can reach the board edge any more
        shieldwall: a row of two or more pieces on the edge is captured together when the moving
                    piece brackets one end, a piece of its own side or a corner the other end, and
                    every piece of the row has an enemy directly in front of it (the king of such a
                    row survives)
        exit_fort: the defenders win when the king is on the edge, can move, and sits inside a fort
                   of defenders that the attackers can never break
        repetition: what the repetition_count-th occurrence of a position means, a key of
                    REPETITION_RULES
        throne_passable: every piece may pass over the empty throne (else only the king may)
        repetition_count: the occurrence of a position that ends the game (3 in most sets)
    The last two have defaults so that checkpoints written before they existed still load.
    """
    name: str
    description: str
    setup: Tuple[str, ...]
    first_player: str
    king_goal: str
    king_capture: str
    king_armed: bool
    throne: bool
    throne_hostile: bool
    encirclement: bool
    shieldwall: bool
    exit_fort: bool
    repetition: str
    throne_passable: bool = True
    repetition_count: int = 3

    def __post_init__(self):
        object.__setattr__(self, "setup", tuple(self.setup))  # JSON gives a list
        n = len(self.setup)
        if n % 2 == 0 or any(len(row) != n for row in self.setup):
            raise ValueError(f"{self.name}: the setup must be a square with an odd side")
        if self.setup[n // 2][n // 2] != "K":
            raise ValueError(f"{self.name}: the king must start on the centre square")
        if self.first_player not in ("attackers", "defenders") or self.king_goal not in ("corner", "edge"):
            raise ValueError(f"{self.name}: invalid first_player or king_goal")
        if self.king_capture not in KING_CAPTURE_RULES or self.repetition not in REPETITION_RULES:
            raise ValueError(f"{self.name}: invalid king_capture or repetition")

    @property
    def board_size(self) -> int:
        return len(self.setup)

    def start_board(self) -> np.ndarray:
        return np.array([[PIECE_CODES[ch] for ch in row] for row in self.setup], dtype=np.int8)

    def describe(self) -> str:
        """The rules as a few lines of text (for logs)."""
        extras = [name for name in ("encirclement", "shieldwall", "exit_fort") if getattr(self, name)]
        throne = "none" if not self.throne else ("hostile" if self.throne_hostile else "not hostile")
        if self.throne and not self.throne_passable:
            throne += ", only the king may pass"
        return (f"Rule set: {self.name} - {self.description}\n"
                f"  {self.board_size}x{self.board_size}, {self.first_player} move first, the king "
                f"wins on {'a corner' if self.king_goal == 'corner' else 'the edge'} and is "
                f"{'armed' if self.king_armed else 'unarmed'}\n"
                f"  King captured: {KING_CAPTURE_RULES[self.king_capture]}\n"
                f"  Throne: {throne}; extra rules: {', '.join(extras) or 'none'}\n"
                f"  Repetition: occurrence {self.repetition_count} of a position is "
                f"{REPETITION_RULES[self.repetition]}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Rules":
        return cls(**data)


BRANDUBH_SETUP = ("...A...",
                  "...A...",
                  "...D...",
                  "AADKDAA",
                  "...D...",
                  "...A...",
                  "...A...")
TABLUT_SETUP = ("...AAA...",
                "....A....",
                "....D....",
                "A...D...A",
                "AADDKDDAA",
                "A...D...A",
                "....D....",
                "....A....",
                "...AAA...")
COPENHAGEN_SETUP = ("...AAAAA...",
                    ".....A.....",
                    "...........",
                    "A....D....A",
                    "A...DDD...A",
                    "AA.DDKDD.AA",
                    "A...DDD...A",
                    "A....D....A",
                    "...........",
                    ".....A.....",
                    "...AAAAA...")
SEA_BATTLE_9X9_SETUP = ("....A....",
                        ".A..A..A.",
                        "..A...A..",
                        "...DDD...",
                        "AA.DKD.AA",
                        "...DDD...",
                        "..A...A..",
                        ".A..A..A.",
                        "....A....")
LEWIS_CROSS_SETUP = ("....AAA....",
                     "....AAA....",
                     ".....D.....",
                     ".....D.....",
                     "AA...D...AA",
                     "AADDDKDDDAA",
                     "AA...D...AA",
                     ".....D.....",
                     ".....D.....",
                     "....AAA....",
                     "....AAA....")
BELL_SETUP = ("....AAA....",
              "....A.A....",
              ".....A.....",
              ".....D.....",
              "AA..DDD..AA",
              "A.ADDKDDA.A",
              "AA..DDD..AA",
              ".....D.....",
              ".....A.....",
              "....A.A....",
              "....AAA....")

# The variant cards C1-C9 of the reference
BRANDUBH = Rules(
    "brandubh", "Brandubh 2 (World Tafl Federation, 2020), the most played 7x7 rules",
    BRANDUBH_SETUP, first_player="attackers", king_goal="corner", king_capture="weak_throne",
    king_armed=True, throne=True, throne_hostile=True, encirclement=True, shieldwall=False,
    exit_fort=False, repetition="defenders_lose")
SAMI_TABLUT = Rules(
    "sami_tablut", "Sami Tablut 9x9 as in the WTF championship (Saami Tablut-w, defenders first)",
    TABLUT_SETUP, first_player="defenders", king_goal="edge", king_capture="weak_near_throne",
    king_armed=True, throne=True, throne_hostile=True, encirclement=True, shieldwall=False,
    exit_fort=False, repetition="defenders_lose")
COPENHAGEN = Rules(
    "copenhagen", "Copenhagen Hnefatafl 11x11 (World Tafl Federation, 2012), the most played 11x11 rules",
    COPENHAGEN_SETUP, first_player="attackers", king_goal="corner", king_capture="strong",
    king_armed=True, throne=True, throne_hostile=True, encirclement=True, shieldwall=True,
    exit_fort=True, repetition="defenders_lose")

RULESETS = {rules.name: rules for rules in [
    BRANDUBH,
    replace(BRANDUBH, name="walker_brandub", king_capture="weak", throne_hostile=False,
            description="Walker's Brandub 7x7 (2011), as played on aagenielsen.dk"),
    replace(BRANDUBH, name="brandubh_2_5w", first_player="defenders", throne_passable=False,
            repetition_count=5, description="Brandubh 2.5W 7x7 (draft of February 2026; start "
                                            "position assumed to be that of Brandubh 2)"),
    SAMI_TABLUT,
    Rules("sea_battle_9x9", "Sea Battle 9x9, Nielsen setup", SEA_BATTLE_9X9_SETUP,
          first_player="attackers", king_goal="edge", king_capture="strong", king_armed=False,
          throne=False, throne_hostile=False, encirclement=True, shieldwall=False, exit_fort=False,
          repetition="defenders_lose"),
    Rules("market_hnefatafl", "Market Hnefatafl 9x9, the beginners' version of Sami Tablut",
          TABLUT_SETUP, first_player="defenders", king_goal="edge", king_capture="weak",
          king_armed=True, throne=False, throne_hostile=False, encirclement=False, shieldwall=False,
          exit_fort=False, repetition="defenders_lose"),
    COPENHAGEN,
    replace(SAMI_TABLUT, name="historical_lewis", setup=LEWIS_CROSS_SETUP,
            description="Historical 11x11 (Tawlbwrdd), Lewis cross setup, as in the WTF championship"),
    replace(SAMI_TABLUT, name="historical_bell", setup=BELL_SETUP,
            description="Historical 11x11 (Tawlbwrdd), Bell setup, defenders first"),
    replace(SAMI_TABLUT, name="simple_tafl", setup=LEWIS_CROSS_SETUP, throne=False,
            throne_hostile=False, description="Simple Tafl 11x11: historical rules, Lewis cross "
                                              "setup, no throne, defenders first"),
    replace(COPENHAGEN, name="fetlar", shieldwall=False, exit_fort=False, repetition="draw",
            description="Fetlar Hnefatafl 11x11 (2007)"),
]}


def get_rules(rules) -> Rules:
    """A Rules object from a Rules object, a rule set name, or the dict form stored in checkpoints."""
    if isinstance(rules, Rules):
        return rules
    if isinstance(rules, dict):
        return Rules.from_dict(rules)
    if rules not in RULESETS:
        raise ValueError(f"Unknown rule set '{rules}'; choose one of {sorted(RULESETS)}")
    return RULESETS[rules]


# ---------------------------------------------------------------------------------------------
# Moves and policy indices
# ---------------------------------------------------------------------------------------------

def policy_size(board_size: int) -> int:
    """Every from-square x 4 directions x (N-1) distances."""
    return board_size * board_size * 4 * (board_size - 1)


def move_index(move: Tuple[int, int, int, int], board_size: int) -> int:
    """
    Policy index of a move: from_square * 4(N-1) + direction * (N-1) + (distance - 1), with
    from_square = row * N + col and directions 0 up, 1 down, 2 left, 3 right.
    """
    from_r, from_c, to_r, to_c = move
    dr, dc = to_r - from_r, to_c - from_c
    direction = (0 if dr < 0 else 1) if dr != 0 else (2 if dc < 0 else 3)
    distance = abs(dr) + abs(dc)
    return (from_r * board_size + from_c) * 4 * (board_size - 1) + direction * (board_size - 1) + distance - 1


@functools.lru_cache(maxsize=None)
def _board_geometry(n: int):
    """
    For an NxN board: coords[square] = (row, col) of each flat square index, and rays[square] = the
    four lists of squares met when sliding from the square up, down, left and right.
    """
    coords = [(r, c) for r in range(n) for c in range(n)]
    rays = []
    for r, c in coords:
        square_rays = []
        for dr, dc in DIRECTIONS:
            ray, rr, cc = [], r + dr, c + dc
            while 0 <= rr < n and 0 <= cc < n:
                ray.append(rr * n + cc)
                rr, cc = rr + dr, cc + dc
            square_rays.append(ray)
        rays.append(square_rays)
    return coords, rays


# ---------------------------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------------------------

class Tafl:
    """A game in progress under one rule set (a Rules object or the name of one in RULESETS)."""

    def __init__(self, rules="copenhagen"):
        self.rules = get_rules(rules)
        n = self.size = self.rules.board_size
        self.board = self.rules.start_board()
        self.current_player = ATTACKER_PLAYER if self.rules.first_player == "attackers" else DEFENDER_PLAYER
        self.game_over = False
        self.winner = None  # stays None for a draw
        self.move_count = 0
        self.throne = (n // 2, n // 2)  # the centre square, also when it is not a throne
        self.corners = ([(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
                        if self.rules.king_goal == "corner" else [])
        # (board bytes, player to move) of every position so far, the current one last
        self.position_history = [self._position_key()]
        # Cache of get_legal_moves: the key (board, player to move) and the moves
        self._legal_moves_key = None
        self._legal_moves = []

    def _position_key(self):
        return (self.board.tobytes(), self.current_player)

    def clone(self) -> "Tafl":
        # The rules and the cached move list are never changed in place, so they can be shared
        new_game = copy.copy(self)
        new_game.board = self.board.copy()
        new_game.position_history = self.position_history.copy()
        return new_game

    def get_piece_planes(self) -> np.ndarray:
        """Piece planes [attackers, defenders, king] (used for history planes)."""
        return np.stack([self.board == piece for piece in (ATTACKER, DEFENDER, KING)]).astype(np.float32)

    def get_state(self) -> np.ndarray:
        """The 4 basic network planes: [attackers, defenders, king, player to move]."""
        player = np.full((1, self.size, self.size), self.current_player, dtype=np.float32)
        return np.concatenate([self.get_piece_planes(), player])

    # --- moves -----------------------------------------------------------------------------

    def get_legal_moves(self, stop_at_first: bool = False) -> List[Tuple[int, int, int, int]]:
        """
        All legal moves of the player to move, as (from_row, from_col, to_row, to_col).

        The list is cached until the board or the player to move changes, so the move check in
        make_move and the search's expansion of a position do not generate the moves again.

        Args:
            stop_at_first: if the moves are not cached, return only the first legal move found
                           (enough to know whether there is any)
        """
        key = (self.board.tobytes(), self.current_player)
        if key == self._legal_moves_key:
            return list(self._legal_moves)

        coords, rays = _board_geometry(self.size)
        cells = self.board.ravel().tolist()  # plain list: much faster to index than numpy
        own = (ATTACKER,) if self.current_player == ATTACKER_PLAYER else (DEFENDER, KING)
        n = self.size
        throne = (n // 2) * n + n // 2 if self.rules.throne else -1
        corners = {r * n + c for r, c in self.corners}

        moves = []
        for square, piece in enumerate(cells):
            if piece not in own:
                continue
            for ray in rays[square]:
                for target in ray:
                    if cells[target] != EMPTY:
                        break
                    if piece != KING:
                        if target in corners:  # corners end a ray, so nothing lies beyond
                            break
                        if target == throne:  # never stop on the throne; pass over it if allowed
                            if self.rules.throne_passable:
                                continue
                            break
                    moves.append(coords[square] + coords[target])
                    if stop_at_first:
                        return moves

        self._legal_moves_key, self._legal_moves = key, moves
        return list(moves)

    def make_move(self, move: Tuple[int, int, int, int]) -> bool:
        """Play a move and decide whether the game is over. Returns False for an illegal move."""
        if self.game_over or move not in self.get_legal_moves():
            return False
        fr, fc, tr, tc = move
        piece = self.board[fr, fc]
        self.board[tr, tc], self.board[fr, fc] = piece, EMPTY
        self.move_count += 1
        captured, king_captured = self._find_captures(self.board, tr, tc)
        for square in captured:
            self.board[square] = EMPTY
        mover = self.current_player
        self.current_player = 1 - mover
        key = self._position_key()
        self.position_history.append(key)

        if piece == KING and self._is_goal(tr, tc):
            self._end(DEFENDER_PLAYER)
        elif king_captured:
            self.board[self.board == KING] = EMPTY
            self._end(ATTACKER_PLAYER)
        elif self.rules.exit_fort and mover == DEFENDER_PLAYER and self._king_in_exit_fort():
            self._end(DEFENDER_PLAYER)
        elif self.rules.encirclement and mover == ATTACKER_PLAYER and self._defenders_encircled():
            self._end(ATTACKER_PLAYER)
        elif self.position_history.count(key) >= self.rules.repetition_count:
            self._end(ATTACKER_PLAYER if self.rules.repetition == "defenders_lose" else None)
        elif not self.get_legal_moves(stop_at_first=True):
            self._end(mover)
        return True

    def _end(self, winner: Optional[int]):
        self.game_over, self.winner = True, winner

    def _is_goal(self, r: int, c: int) -> bool:
        if self.rules.king_goal == "corner":
            return (r, c) in self.corners
        return r in (0, self.size - 1) or c in (0, self.size - 1)

    # --- captures --------------------------------------------------------------------------

    def _inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def _closes_sandwich(self, board: np.ndarray, r: int, c: int, target: int) -> bool:
        """Whether square (r, c) closes a sandwich against a `target` piece (not the king)."""
        piece = board[r, c]
        if target == ATTACKER and (piece == DEFENDER or (piece == KING and self.rules.king_armed)):
            return True
        if target == DEFENDER and piece == ATTACKER:
            return True
        if (r, c) in self.corners:
            return True
        if self.rules.throne and self.rules.throne_hostile and (r, c) == self.throne:
            return target == ATTACKER or piece == EMPTY
        return False

    def _find_captures(self, board: np.ndarray, r: int, c: int):
        """
        The captures made by the piece that just moved to (r, c), without changing the board.

        Returns:
            (list of captured (row, col) squares other than the king, whether the king is captured)
        """
        piece = board[r, c]
        if piece == KING and not self.rules.king_armed:
            return [], False
        attacking = piece == ATTACKER
        captured, king_captured = [], False
        for dr, dc in DIRECTIONS:
            r1, c1, r2, c2 = r + dr, c + dc, r + 2 * dr, c + 2 * dc
            if not self._inside(r1, c1):
                continue
            target = board[r1, c1]
            if target == EMPTY or (target == ATTACKER) == attacking:
                continue
            if target == KING:
                king_captured = king_captured or self._king_captured(board, r1, c1, dr, dc)
            elif self._inside(r2, c2) and self._closes_sandwich(board, r2, c2, target):
                captured.append((r1, c1))
        if self.rules.shieldwall and (r in (0, self.size - 1) or c in (0, self.size - 1)):
            captured += self._shieldwall_captures(board, r, c)
        return captured, king_captured

    def _king_captured(self, board: np.ndarray, r: int, c: int, dr: int, dc: int) -> bool:
        """Whether the king on (r, c) is captured by the attacker that just moved next to him,
        coming from direction (-dr, -dc)."""
        rule, n = self.rules.king_capture, self.size
        t = n // 2
        on_throne = self.rules.throne and (r, c) == self.throne
        next_to_throne = self.rules.throne and abs(r - t) + abs(c - t) == 1
        four_sides = (rule == "strong" or (rule == "weak_throne" and on_throne)
                      or (rule == "weak_near_throne" and (on_throne or next_to_throne)))
        if not four_sides:
            # Two opposite sides: the attacker that moved, and an attacker or a corner beyond
            r2, c2 = r + dr, c + dc
            return self._inside(r2, c2) and (board[r2, c2] == ATTACKER or (r2, c2) in self.corners)
        if r in (0, n - 1) or c in (0, n - 1):  # the edge never counts as a side
            return False
        # Attackers on all four sides; the empty throne next to the king counts as one of them
        return all(board[r + ar, c + ac] == ATTACKER
                   or (self.rules.throne and (r + ar, c + ac) == self.throne) for ar, ac in DIRECTIONS)

    def _shieldwall_captures(self, board: np.ndarray, r: int, c: int) -> List[Tuple[int, int]]:
        """Pieces captured by a shieldwall that the piece on edge square (r, c) closes."""
        n = self.size
        attacking = board[r, c] == ATTACKER
        enemies = (DEFENDER, KING) if attacking else (ATTACKER,)
        own = (ATTACKER,) if attacking else ((DEFENDER, KING) if self.rules.king_armed else (DEFENDER,))
        captured = []
        # (direction into the board, the two directions along the edge) for each edge of (r, c)
        edges = [((1, 0), r == 0), ((-1, 0), r == n - 1), ((0, 1), c == 0), ((0, -1), c == n - 1)]
        for (ir, ic), on_edge in edges:
            if not on_edge:
                continue
            for ar, ac in ((ic, ir), (-ic, -ir)):
                row, rr, cc = [], r + ar, c + ac
                while self._inside(rr, cc) and board[rr, cc] in enemies:
                    row.append((rr, cc))
                    rr, cc = rr + ar, cc + ac
                closed = self._inside(rr, cc) and (board[rr, cc] in own or (rr, cc) in self.corners)
                faced = all(board[sr + ir, sc + ic] in own for sr, sc in row)
                if len(row) >= 2 and closed and faced:
                    captured += [(sr, sc) for sr, sc in row if board[sr, sc] != KING]
        return captured

    # --- encirclement and exit fort --------------------------------------------------------

    def _defenders_encircled(self) -> bool:
        """True if no defender (king included) can reach the edge through squares without attackers."""
        # Number the connected areas of squares without attackers (neighbours share a side)
        areas, _ = scipy.ndimage.label(self.board != ATTACKER)
        defender_areas = set(areas[self.board >= DEFENDER].tolist())
        edge_areas = set(np.concatenate([areas[0], areas[-1], areas[:, 0], areas[:, -1]]).tolist())
        return not (defender_areas & edge_areas)

    def _king_in_exit_fort(self) -> bool:
        """
        The exit fort test (reference B2): the king stands on the edge, can move, and is walled in
        by defending soldiers that can never be captured, whatever the attackers do.

        The fort's inside is the king's square and the empty squares he can walk to. No attacker may
        stand next to it, so all its neighbours are defending soldiers: the wall. A square is "open"
        if it is on the board, outside the fort and not held by a soldier of the safe set, so that an
        attacker or a hostile square might one day use it. The safe set starts as all defending
        soldiers. Soldiers that could be captured leave it until nothing changes: a soldier with open
        squares on both sides along one line, and (with the shieldwall rule) a run of two or more
        soldiers along an edge whose front squares and both end squares are open. The fort holds if
        every wall soldier stays safe. This may reject a fort that holds only through good play, but
        never accepts one that can be broken.
        """
        n, board = self.size, self.board
        kings = np.argwhere(board == KING)
        if len(kings) == 0:
            return False
        kr, kc = kings[0]
        if not (kr in (0, n - 1) or kc in (0, n - 1)):
            return False
        walkable = board == EMPTY
        walkable[kr, kc] = True
        areas, _ = scipy.ndimage.label(walkable)
        inside = areas == areas[kr, kc]
        if inside.sum() < 2:  # the king cannot move
            return False
        wall = scipy.ndimage.binary_dilation(inside) & ~inside
        if (board[wall] == ATTACKER).any():
            return False

        safe = board == DEFENDER
        # The four edge lines, each with the direction into the board (for the shieldwall runs)
        edge_lines = [([(0, c) for c in range(n)], (1, 0)), ([(n - 1, c) for c in range(n)], (-1, 0)),
                      ([(r, 0) for r in range(n)], (0, 1)), ([(r, n - 1) for r in range(n)], (0, -1))]
        changed = True
        while changed:
            changed = False
            is_open = ~inside & ~safe
            for r, c in np.argwhere(safe):
                for dr, dc in ((1, 0), (0, 1)):
                    if (self._inside(r - dr, c - dc) and self._inside(r + dr, c + dc)
                            and is_open[r - dr, c - dc] and is_open[r + dr, c + dc]):
                        safe[r, c], changed = False, True
            if not self.rules.shieldwall:
                continue
            is_open = ~inside & ~safe
            for line, (ir, ic) in edge_lines:
                i = 0
                while i < n:
                    if not safe[line[i]]:
                        i += 1
                        continue
                    j = i
                    while j < n and safe[line[j]]:
                        j += 1
                    run = line[i:j]  # an unbroken run of safe soldiers, line[i] to line[j - 1]
                    ends_open = i > 0 and j < n and is_open[line[i - 1]] and is_open[line[j]]
                    if len(run) >= 2 and ends_open and all(is_open[r + ir, c + ic] for r, c in run):
                        for square in run:
                            safe[square] = False
                        changed = True
                    i = j
        return bool(safe[wall].all())

    def __str__(self) -> str:
        symbols = {EMPTY: '·', ATTACKER: '○', DEFENDER: '●', KING: '♔'}
        lines = ["   " + " ".join(f"{c:x}" for c in range(self.size))]
        for r in range(self.size):
            row = []
            for c in range(self.size):
                piece = self.board[r, c]
                if piece == EMPTY and (r, c) in self.corners:
                    row.append("X")
                elif piece == EMPTY and self.rules.throne and (r, c) == self.throne:
                    row.append("⊕")
                else:
                    row.append(symbols[piece])
            lines.append(f"{r:x} │" + " ".join(row))
        lines.append(f"\n{self.rules.name}; to move: "
                     f"{'attackers (○)' if self.current_player == ATTACKER_PLAYER else 'defenders (●/♔)'}")
        if self.game_over:
            lines.append("Game over: " + {None: "draw", 0: "attackers win", 1: "defenders win"}[self.winner])
        return "\n".join(lines)
