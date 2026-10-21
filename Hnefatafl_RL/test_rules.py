"""
Tests of the rule sets and the game engine (tafl.py): one hand-made position per rule, checked
against the published rules of each named rule set.

Run with: python test_rules.py (exit code 1 if a test fails)
"""

import sys
from dataclasses import replace

import numpy as np

from tafl import ATTACKER, DEFENDER, EMPTY, ENDINGS, KING, RULESETS, Tafl

A, D, K = ATTACKER, DEFENDER, KING


def position(rules, pieces: dict, player: int) -> Tafl:
    """A game under `rules` with only `pieces` ({(row, col): piece}) on the board, `player` to move."""
    game = Tafl(rules)
    game.board[:] = EMPTY
    for (r, c), piece in pieces.items():
        game.board[r, c] = piece
    game.current_player = player
    game.position_history = [game._position_key()]
    return game


def after(rules, pieces: dict, player: int, move) -> Tafl:
    """The game after `move` from the given position."""
    game = position(rules, pieces, player)
    assert game.make_move(move), f"{rules if isinstance(rules, str) else rules.name}: {move} is illegal"
    return game


def test_setups():
    """Every setup has the published piece counts and all 8 board symmetries (the training data are
    augmented with them), and the first player is as named."""
    counts = {7: (8, 4), 9: (16, 8), 11: (24, 12)}
    for rules in RULESETS.values():
        board = rules.start_board()
        assert ((board == A).sum(), (board == D).sum(), (board == K).sum()) == (*counts[rules.board_size], 1)
        for k in range(4):
            assert np.array_equal(np.rot90(board, k), board) and np.array_equal(np.rot90(board, k).T, board)
        assert Tafl(rules).current_player == (0 if rules.first_player == "attackers" else 1)
        assert rules.describe().startswith(f"Rule set: {rules.name}")
    print("  setups: piece counts, symmetries and first player")
    return True


def test_movement():
    """Only the king may stop on the throne and the corners; every piece passes over the empty throne."""
    game = position("brandubh", {(3, 0): A, (2, 3): K, (0, 4): A}, 0)
    moves = game.get_legal_moves()
    assert (3, 0, 3, 4) in moves and (3, 0, 3, 3) not in moves, "the throne must be passable, not a stop"
    assert (0, 4, 0, 6) not in moves and (0, 4, 0, 5) in moves, "only the king may stop on a corner"
    game.current_player = 1
    assert (2, 3, 3, 3) in game.get_legal_moves(), "the king may return to the throne"
    # Without corner squares (edge goal) a corner is an ordinary square
    assert (0, 4, 0, 8) in position("sami_tablut", {(0, 4): A, (4, 4): K}, 0).get_legal_moves()
    # Brandubh 2.5W: only the king may cross the throne
    game = position("brandubh_2_5w", {(3, 0): A, (2, 3): K}, 0)
    assert (3, 0, 3, 2) in game.get_legal_moves() and (3, 0, 3, 4) not in game.get_legal_moves()
    game = position("brandubh_2_5w", {(3, 1): K, (0, 3): A}, 1)
    assert (3, 1, 3, 5) in game.get_legal_moves(), "the king may cross the throne"
    # No throne (Sea Battle): any piece may stop on the centre square
    assert (4, 0, 4, 4) in position("sea_battle_9x9", {(4, 0): A, (1, 1): K}, 0).get_legal_moves()
    print("  movement: throne and corners restricted to the king, throne passable")
    return True


def test_king_goal():
    """The king wins on a corner in corner-goal sets (not on the edge), on any edge square in
    edge-goal sets."""
    assert after("brandubh", {(1, 1): K, (5, 5): A}, 1, (1, 1, 0, 1)).game_over is False
    assert after("brandubh", {(0, 1): K, (5, 5): A}, 1, (0, 1, 0, 0)).winner == 1
    assert after("sami_tablut", {(2, 2): K, (6, 6): A}, 1, (2, 2, 0, 2)).winner == 1
    print("  king goal: corner or edge")
    return True


def test_king_capture():
    """Each king capture rule, at the squares where the rule sets differ."""
    cases = [
        # Brandubh 2: two sides, but four on the throne; the throne never counts against the king,
        # a corner does
        ("brandubh", {(3, 3): K, (3, 2): A, (6, 4): A}, (6, 4, 3, 4), False),
        ("brandubh", {(3, 3): K, (3, 2): A, (2, 3): A, (4, 3): A, (6, 4): A}, (6, 4, 3, 4), True),
        ("brandubh", {(2, 3): K, (2, 2): A, (6, 4): A}, (6, 4, 2, 4), True),
        ("brandubh", {(2, 3): K, (1, 0): A}, (1, 0, 1, 3), False),
        ("brandubh", {(0, 1): K, (4, 2): A}, (4, 2, 0, 2), True),
        # Walker's Brandub: two sides also on the throne
        ("walker_brandub", {(3, 3): K, (3, 2): A, (6, 4): A}, (6, 4, 3, 4), True),
        # Sami Tablut: next to the throne the three other sides, elsewhere two
        ("sami_tablut", {(3, 4): K, (3, 3): A, (6, 5): A}, (6, 5, 3, 5), False),
        ("sami_tablut", {(3, 4): K, (3, 3): A, (2, 4): A, (6, 5): A}, (6, 5, 3, 5), True),
        ("sami_tablut", {(2, 2): K, (2, 1): A, (6, 3): A}, (6, 3, 2, 3), True),
        ("historical_lewis", {(4, 5): K, (4, 4): A, (3, 5): A, (8, 6): A}, (8, 6, 4, 6), True),
        # Without a throne the centre square is ordinary: Market Hnefatafl and Simple Tafl capture
        # the king there between two attackers, Sea Battle needs four
        ("market_hnefatafl", {(4, 4): K, (4, 3): A, (8, 5): A}, (8, 5, 4, 5), True),
        ("simple_tafl", {(5, 5): K, (5, 4): A, (9, 6): A}, (9, 6, 5, 6), True),
        ("sea_battle_9x9", {(4, 4): K, (4, 3): A, (3, 4): A, (8, 5): A}, (8, 5, 4, 5), False),
        ("sea_battle_9x9", {(4, 4): K, (4, 3): A, (3, 4): A, (5, 4): A, (8, 5): A}, (8, 5, 4, 5), True),
        # Copenhagen: four sides, three next to the throne, never on the edge
        ("copenhagen", {(2, 2): K, (1, 2): A, (3, 2): A, (8, 3): A}, (8, 3, 2, 3), False),
        ("copenhagen", {(2, 2): K, (1, 2): A, (3, 2): A, (2, 1): A, (8, 3): A}, (8, 3, 2, 3), True),
        ("copenhagen", {(4, 5): K, (3, 5): A, (4, 4): A, (8, 6): A}, (8, 6, 4, 6), True),
        ("copenhagen", {(0, 5): K, (0, 4): A, (1, 5): A, (3, 6): A}, (3, 6, 0, 6), False),
    ]
    for rules, pieces, move, captured in cases:
        game = after(rules, {**pieces, (6, 0): D}, 0, move)  # a spare defender, so the game goes on
        name = rules if isinstance(rules, str) else rules.name
        assert (game.winner == 0) == captured, f"{name}: king at {list(pieces)[0]} after {move}"
        assert ((game.board == K).sum() == 0) == captured
    print("  king capture: weak, weak on throne, weak near throne, strong; with and without throne")
    return True


def test_hostile_squares():
    """The empty throne is hostile to defenders and attackers, the occupied throne not to defenders;
    the corners are hostile to all; without throne hostility nothing is captured against it."""
    defender_vs_throne = ({(2, 3): D, (1, 0): A, (0, 1): K}, 0, (1, 0, 1, 3))
    game = after("brandubh", *defender_vs_throne)
    assert game.board[2, 3] == EMPTY, "the empty throne must capture a defender"
    game = after("brandubh", {(2, 3): D, (1, 0): A, (3, 3): K}, 0, (1, 0, 1, 3))
    assert game.board[2, 3] == D, "a throne with the king on it is not hostile to defenders"
    game = after("brandubh", {(3, 2): A, (6, 1): D, (0, 1): K, (5, 5): A}, 1, (6, 1, 3, 1))
    assert game.board[3, 2] == EMPTY, "the empty throne must capture an attacker"
    game = after("walker_brandub", *defender_vs_throne)
    assert game.board[2, 3] == D, "a throne that is not hostile captures nothing"
    game = after("copenhagen", {(0, 1): D, (3, 2): A, (5, 5): K, (8, 8): A}, 0, (3, 2, 0, 2))
    assert game.board[0, 1] == EMPTY, "a corner must capture a defender"
    print("  hostile squares: throne (empty and occupied) and corners")
    return True


def test_shieldwall():
    """A row on the edge is captured when bracketed and faced by enemies; a corner can bracket; the
    king survives in the row; a row with a gap in front survives; Fetlar has no shieldwall."""
    wall = {(10, 4): D, (10, 5): D, (9, 4): A, (9, 5): A, (10, 6): A, (8, 3): A, (2, 2): K, (1, 8): D}
    move = (8, 3, 10, 3)
    game = after("copenhagen", wall, 0, move)
    assert game.board[10, 4] == EMPTY and game.board[10, 5] == EMPTY, "shieldwall not captured"
    game = after("copenhagen", {**wall, (10, 5): K, (2, 2): D}, 0, move)
    assert game.board[10, 4] == EMPTY and game.board[10, 5] == K, "the king must survive in the wall"
    game = after("copenhagen", {k: v for k, v in wall.items() if k != (9, 5)}, 0, move)
    assert game.board[10, 4] == D and game.board[10, 5] == D, "a row with a gap in front survives"
    corner = {(10, 1): D, (10, 2): D, (9, 1): A, (9, 2): A, (8, 3): A, (2, 2): K, (1, 8): D}
    game = after("copenhagen", corner, 0, move)
    assert game.board[10, 1] == EMPTY and game.board[10, 2] == EMPTY, "the corner must bracket"
    game = after("fetlar", wall, 0, move)
    assert game.board[10, 4] == D, "Fetlar has no shieldwall"
    print("  shieldwall: brackets, corner, king in the row, gap in front, not in Fetlar")
    return True


def test_exit_fort():
    """The defenders win when the king sits on the edge in a fort whose walls cannot be captured,
    also when soldiers outside the wall protect it; not when a wall piece can be captured, not in
    Fetlar, and only after a defenders' move."""
    far = {(2, 2): A, (3, 8): A}
    # Inside: (10,5) king, (10,6), (9,5), (9,6). The last wall piece arrives at (10,7).
    fort = {(10, 5): K, (8, 5): D, (8, 6): D, (9, 4): D, (9, 7): D, (10, 4): D, (10, 9): D, **far}
    assert after("copenhagen", fort, 1, (10, 9, 10, 7)).winner == 1
    assert not after("fetlar", fort, 1, (10, 9, 10, 7)).game_over
    # Inside (10,5) king and (9,5): the top piece (8,5) can be captured from (8,4) and (8,6)
    weak_fort = {(10, 5): K, (8, 5): D, (9, 4): D, (9, 6): D, (10, 4): D, (10, 9): D, **far}
    assert not after("copenhagen", weak_fort, 1, (10, 9, 10, 6)).game_over
    # The same fort with (8,4) and (8,6) added: they are not wall pieces, but they protect (8,5)
    assert after("copenhagen", {**weak_fort, (8, 4): D, (8, 6): D}, 1, (10, 9, 10, 6)).winner == 1
    # A finished fort is not tested after an attackers' move, but after the next defenders' move
    finished = {**fort, (10, 7): D, (10, 9): EMPTY, (5, 0): D}
    game = after("copenhagen", finished, 0, (2, 2, 2, 1))
    assert not game.game_over
    game.make_move((5, 0, 6, 0))
    assert game.winner == 1
    print("  exit fort: unbreakable forts win (also with outside support), breakable ones and Fetlar "
          "do not; tested after defenders' moves")
    return True


def test_encirclement():
    """Closing a ring around all defenders wins; a defender outside the ring prevents it."""
    ring = {(5, 5): K, (3, 5): A, (7, 5): A, (5, 3): A, (5, 7): A, (4, 4): A, (4, 6): A, (6, 4): A,
            (6, 9): A}
    # Copenhagen, and the same rule with a weak king (as in Brandubh 2 and Sami Tablut)
    for rules in (RULESETS["copenhagen"], replace(RULESETS["copenhagen"], king_capture="weak")):
        assert after(rules, ring, 0, (6, 9, 6, 6)).winner == 0
        assert not after(rules, {**ring, (1, 1): D}, 0, (6, 9, 6, 6)).game_over
    assert not after(replace(RULESETS["copenhagen"], encirclement=False), ring, 0, (6, 9, 6, 6)).game_over
    print("  encirclement: a closed ring wins, unless a defender is outside")
    return True


def test_repetition():
    """The third occurrence of a position (the start counts) ends the game: an attacker win in
    Copenhagen, also when the attackers made the move, a draw in Fetlar; in Brandubh 2.5W only the
    fifth occurrence ends it."""
    pieces = {(2, 2): K, (7, 7): D, (4, 8): A, (8, 3): A}
    cycle = [(4, 8, 4, 9), (7, 7, 7, 6), (4, 9, 4, 8), (7, 6, 7, 7)]  # back to the start
    for rules, expected in [("copenhagen", 0), ("fetlar", None)]:
        game = position(rules, pieces, 0)
        for move in cycle + cycle[:3]:
            assert game.make_move(move) and not game.game_over
        game.make_move(cycle[3])
        assert game.game_over and game.winner == expected, f"{rules}: wrong repetition result"
    # The attackers repeat: their move gives the third occurrence of a position and they win
    game = position("copenhagen", pieces, 0)
    for move in cycle * 2:
        game.make_move(move)
    game.make_move(cycle[0])
    assert game.winner == 0
    # Brandubh 2.5W (defenders first): the start position occurs for the fifth time after 16 plies
    game = position("brandubh_2_5w", {(2, 2): K, (5, 5): D, (1, 5): A, (5, 1): A}, 1)
    cycle = [(5, 5, 5, 4), (1, 5, 1, 4), (5, 4, 5, 5), (1, 4, 1, 5)]
    for ply, move in enumerate(cycle * 4):
        assert game.make_move(move) and game.game_over == (ply == 15)
    assert game.winner == 0
    print("  repetition: attackers win or draw at the third occurrence; fifth in Brandubh 2.5W")
    return True


def test_end_reasons():
    """Every way a game ends is recorded in end_reason (one of ENDINGS), and Rules.endings() leaves
    out the endings a rule set does not have."""
    far = {(2, 2): A, (3, 8): A}
    fort = {(10, 5): K, (8, 5): D, (8, 6): D, (9, 4): D, (9, 7): D, (10, 4): D, (10, 9): D, **far}
    ring = {(5, 5): K, (3, 5): A, (7, 5): A, (5, 3): A, (5, 7): A, (4, 4): A, (4, 6): A, (6, 4): A,
            (6, 9): A}
    cases = [
        (after("brandubh", {(0, 1): K, (5, 5): A}, 1, (0, 1, 0, 0)), 1, "king_escaped"),
        (after("brandubh", {(2, 2): K, (2, 1): A, (2, 5): A}, 0, (2, 5, 2, 3)), 0, "king_captured"),
        (after("copenhagen", fort, 1, (10, 9, 10, 7)), 1, "exit_fort"),
        (after("copenhagen", ring, 0, (6, 9, 6, 6)), 0, "encircled"),
        # The king alone on the edge, walled in: no capture on the edge, but no legal move either
        (after("copenhagen", {(0, 5): K, (0, 4): A, (0, 6): A, (1, 9): A}, 0, (1, 9, 1, 5)), 0,
         "defenders_blocked"),
    ]
    for game, winner, reason in cases:
        assert game.game_over and game.winner == winner and game.end_reason == reason, \
            f"expected {reason}, got {game.end_reason} (winner {game.winner})"
    game = position("copenhagen", {(2, 2): K, (7, 7): D, (4, 8): A, (8, 3): A}, 0)
    for move in [(4, 8, 4, 9), (7, 7, 7, 6), (4, 9, 4, 8), (7, 6, 7, 7)] * 2:
        game.make_move(move)
    assert game.end_reason == "repetition" and Tafl("copenhagen").end_reason is None
    assert RULESETS["copenhagen"].endings() == ENDINGS
    assert set(ENDINGS) - set(RULESETS["market_hnefatafl"].endings()) == {"exit_fort", "encircled"}
    print("  end reasons: escape, capture, fort, encirclement, blocked, repetition; endings per "
          "rule set")
    return True


if __name__ == "__main__":
    print("Testing the rule sets...")
    tests = [test_setups, test_movement, test_king_goal, test_king_capture, test_hostile_squares,
             test_shieldwall, test_exit_fort, test_encirclement, test_repetition, test_end_reasons]
    all_passed = True
    for test in tests:
        try:
            all_passed &= test()
        except Exception as e:  # any error is a failure, not only a failed assert
            print(f"  FAILED {test.__name__}: {type(e).__name__}: {e}")
            all_passed = False
    print("All tests PASSED!" if all_passed else "Some tests FAILED!")
    sys.exit(0 if all_passed else 1)
