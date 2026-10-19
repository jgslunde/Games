"""
Tafl games on 7x7, 9x9 and 11x11 boards, played under named rule sets.

A rule set (Rules) fixes everything that differs between the common variants: the starting
position, who moves first, where the king escapes to, how he is captured, the throne, and the
extra rules of some sets (encirclement, shieldwall, exit fort, what an endless repetition means).
Training runs use only the named sets in RULESETS, which follow published rules; RULESETS.md
lists their sources and how balanced they are in human play.

Board: an NxN int8 array with EMPTY 0, ATTACKER 1, DEFENDER 2, KING 3. Players: 0 attackers,
1 defenders. A move is (from_row, from_col, to_row, to_col).

Rules shared by all sets:
- every piece moves like a rook in chess: any number of empty squares along a row or column;
- a piece other than the king is captured when the opponent's move closes a sandwich on two
  opposite sides, by two enemy pieces or by an enemy piece and a hostile square; moving into a
  sandwich is safe;
- only the king may stop on the throne (the centre square, in sets with a throne) and on the
  corners (in sets where the king escapes to a corner); every piece may pass over the empty
  throne; the corners are hostile to all pieces;
- a player who cannot move loses. Overlong games are ended as draws by the code that plays them.
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
    "strong_edge": "from all four sides, where the board edge, a corner and the empty throne each "
                   "count as one side",
}
REPETITION_RULES = {
    "forbidden": "a move that gives a position for the third time is not allowed (a player with "
                 "no other move loses)",
    "defenders_lose": "the third occurrence of a position is a win for the attackers",
    "draw": "the third occurrence of a position is a draw",
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
        repetition: what a threefold repetition means, a key of REPETITION_RULES
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
        return (f"Rule set: {self.name} - {self.description}\n"
                f"  {self.board_size}x{self.board_size}, {self.first_player} move first, the king "
                f"wins on {'a corner' if self.king_goal == 'corner' else 'the edge'} and is "
                f"{'armed' if self.king_armed else 'unarmed'}\n"
                f"  King captured: {KING_CAPTURE_RULES[self.king_capture]}\n"
                f"  Throne: {throne}; extra rules: {', '.join(extras) or 'none'}\n"
                f"  Repetition: {REPETITION_RULES[self.repetition]}")

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

SAMI_TABLUT = Rules(
    "sami_tablut", "Sami Tablut 9x9 (historical rules after Linnaeus, aagenielsen.dk), attackers first",
    TABLUT_SETUP, first_player="attackers", king_goal="edge", king_capture="weak_near_throne",
    king_armed=True, throne=True, throne_hostile=True, encirclement=True, shieldwall=False,
    exit_fort=False, repetition="defenders_lose")
FETLAR = Rules(
    "fetlar", "Fetlar Hnefatafl 11x11 (2007)", COPENHAGEN_SETUP, first_player="attackers",
    king_goal="corner", king_capture="strong", king_armed=True, throne=True, throne_hostile=True,
    encirclement=True, shieldwall=False, exit_fort=False, repetition="draw")

RULESETS = {rules.name: rules for rules in [
    Rules("brandubh", "Brandubh 2 (World Tafl Federation, 2020), the most played 7x7 rules",
          BRANDUBH_SETUP, first_player="attackers", king_goal="corner", king_capture="weak_throne",
          king_armed=True, throne=True, throne_hostile=True, encirclement=True, shieldwall=False,
          exit_fort=False, repetition="forbidden"),
    SAMI_TABLUT,
    replace(SAMI_TABLUT, name="sami_tablut_defenders_first", first_player="defenders",
            description="Sami Tablut 9x9 with the defenders moving first (WTF championship rules)"),
    Rules("copenhagen", "Copenhagen Hnefatafl 11x11 (World Tafl Federation, 2012), the most played "
          "11x11 rules", COPENHAGEN_SETUP, first_player="attackers", king_goal="corner",
          king_capture="strong", king_armed=True, throne=True, throne_hostile=True,
          encirclement=True, shieldwall=True, exit_fort=True, repetition="defenders_lose"),
    FETLAR,
    replace(FETLAR, name="old_hnefatafl", king_capture="strong_edge", repetition="forbidden",
            description="Old Hnefatafl 11x11: Fetlar, but the king can be captured on the edge "
                        "and repetitions are not allowed"),
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
        # Cache of get_legal_moves: the key (board, player to move, history length) and the moves
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

        The list is cached until the board, the player to move or the length of the position
        history changes (the history only grows through make_move), so the move check in make_move
        and the search's expansion of a position do not generate the moves again.

        Args:
            stop_at_first: if the moves are not cached, return only the first legal move found
                           (enough to know whether there is any)
        """
        key = (self.board.tobytes(), self.current_player, len(self.position_history))
        if key == self._legal_moves_key:
            return list(self._legal_moves)

        coords, rays = _board_geometry(self.size)
        cells = self.board.ravel().tolist()  # plain list: much faster to index than numpy
        own = (ATTACKER,) if self.current_player == ATTACKER_PLAYER else (DEFENDER, KING)
        n = self.size
        throne = (n // 2) * n + n // 2 if self.rules.throne else -1
        corners = {r * n + c for r, c in self.corners}
        # A move can only give a third occurrence of a position if some position occurred twice
        history = self.position_history
        check_repetition = self.rules.repetition == "forbidden" and len(set(history)) < len(history)

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
                        if target == throne:  # pass over the throne, but never stop on it
                            continue
                    move = coords[square] + coords[target]
                    if check_repetition and self._repeats_position(move):
                        continue
                    moves.append(move)
                    if stop_at_first:
                        return moves

        self._legal_moves_key, self._legal_moves = key, moves
        return list(moves)

    def _repeats_position(self, move) -> bool:
        """Whether `move` gives a position that has already occurred twice."""
        fr, fc, tr, tc = move
        board = self.board.copy()
        board[tr, tc], board[fr, fc] = board[fr, fc], EMPTY
        if self.position_history.count((board.tobytes(), 1 - self.current_player)) < 2:
            return False
        # A capture changes the position: pieces are never added, so a position with fewer
        # pieces has not occurred before (the count above was made without the captures)
        captured, king_captured = self._find_captures(board, tr, tc)
        return not captured and not king_captured

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

        if king_captured:
            self.board[self.board == KING] = EMPTY
            self._end(ATTACKER_PLAYER)
        elif piece == KING and self._is_goal(tr, tc):
            self._end(DEFENDER_PLAYER)
        elif self.rules.encirclement and mover == ATTACKER_PLAYER and self._defenders_encircled():
            self._end(ATTACKER_PLAYER)
        elif self.rules.exit_fort and self._king_in_exit_fort():
            self._end(DEFENDER_PLAYER)
        elif self.rules.repetition != "forbidden" and self.position_history.count(key) >= 3:
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
        four_sides = (rule in ("strong", "strong_edge") or (rule == "weak_throne" and on_throne)
                      or (rule == "weak_near_throne" and (on_throne or next_to_throne)))
        if not four_sides:
            # Two opposite sides: the attacker that moved, and an attacker or a corner beyond
            r2, c2 = r + dr, c + dc
            return self._inside(r2, c2) and (board[r2, c2] == ATTACKER or (r2, c2) in self.corners)
        if rule == "strong" and (r in (0, n - 1) or c in (0, n - 1)):
            return False
        for ar, ac in DIRECTIONS:
            nr, nc = r + ar, c + ac
            if not self._inside(nr, nc):
                if rule == "strong_edge":
                    continue
                return False
            # The empty throne next to the king counts as one side
            if board[nr, nc] == ATTACKER or (self.rules.throne and (nr, nc) == self.throne):
                continue
            if rule == "strong_edge" and (nr, nc) in self.corners:
                continue
            return False
        return True

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
        True if the king is on the edge, can move, and is walled in by defenders that can never be
        captured. The fort's inside is the area of empty squares the king can walk to. No attacker
        may touch it, and every wall piece (a defender next to the inside) must, along both lines
        through it, have a neighbour that an attacker can never occupy: off the board, inside the
        fort or another wall piece. With the shieldwall rule a wall piece on the edge must also have
        such a square in front of it. This misses forts that are only safe thanks to tactics, but
        never calls a breakable fort safe.
        """
        n, board = self.size, self.board
        kings = np.argwhere(board == KING)
        if len(kings) == 0:
            return False
        kr, kc = kings[0]
        if not (kr in (0, n - 1) or kc in (0, n - 1)):
            return False
        open_squares = board == EMPTY
        open_squares[kr, kc] = True
        areas, _ = scipy.ndimage.label(open_squares)
        inside = areas == areas[kr, kc]
        if inside.sum() < 2:  # the king cannot move
            return False
        walls = scipy.ndimage.binary_dilation(inside) & ~inside
        if (board[walls] == ATTACKER).any():
            return False
        safe = lambda rr, cc: not self._inside(rr, cc) or inside[rr, cc] or walls[rr, cc]
        for r, c in np.argwhere(walls):
            for dr, dc in ((1, 0), (0, 1)):
                if not (safe(r - dr, c - dc) or safe(r + dr, c + dc)):
                    return False
            if self.rules.shieldwall:
                for (ir, ic), on_edge in [((1, 0), r == 0), ((-1, 0), r == n - 1),
                                          ((0, 1), c == 0), ((0, -1), c == n - 1)]:
                    if on_edge and not (inside[r + ir, c + ic] or walls[r + ir, c + ic]):
                        return False
        return True

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
