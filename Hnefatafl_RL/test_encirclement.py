"""
Test the encirclement rule in Hnefatafl.
"""

import numpy as np
from hnefatafl import Hnefatafl, EMPTY, ATTACKER, DEFENDER, KING

def test_no_encirclement_start():
    """Test that the starting position is not encircled."""
    game = Hnefatafl()
    print("Test 1: Starting position should NOT be encircled")
    print(game)
    is_encircled = game._is_king_encircled()
    print(f"King encircled: {is_encircled}")
    assert not is_encircled, "Starting position should not be encircled"
    print("✓ PASSED\n")

def test_simple_encirclement():
    """Test a simple encirclement scenario."""
    game = Hnefatafl()
    print("Test 2: Simple box encirclement")
    
    # Clear the board
    game.board = np.zeros((11, 11), dtype=np.int8)
    
    # Put king in center
    game.board[5, 5] = KING
    
    # Create a box of attackers around the king (3x3 box)
    for r in range(4, 7):
        for c in range(4, 7):
            if (r, c) != (5, 5):  # Don't overwrite the king
                game.board[r, c] = ATTACKER if r in [4, 6] or c in [4, 6] else EMPTY
    
    print(game)
    is_encircled = game._is_king_encircled()
    print(f"King encircled: {is_encircled}")
    assert is_encircled, "King should be encircled in a box"
    print("✓ PASSED\n")

def test_encirclement_with_defenders():
    """Test encirclement with some defenders inside the ring."""
    game = Hnefatafl()
    print("Test 3: Encirclement with defenders inside")
    
    # Clear the board
    game.board = np.zeros((11, 11), dtype=np.int8)
    
    # Put king and some defenders in center area
    game.board[5, 5] = KING
    game.board[5, 4] = DEFENDER
    game.board[4, 5] = DEFENDER
    
    # Create a larger box of attackers (5x5 box)
    for r in range(3, 8):
        for c in range(3, 8):
            if r in [3, 7] or c in [3, 7]:
                game.board[r, c] = ATTACKER
    
    print(game)
    is_encircled = game._is_king_encircled()
    print(f"King encircled: {is_encircled}")
    assert is_encircled, "King and defenders should be encircled together"
    print("✓ PASSED\n")

def test_breach_in_wall():
    """Test that a breach in the attacker wall means no encirclement."""
    game = Hnefatafl()
    print("Test 4: Breach in attacker wall")
    
    # Clear the board
    game.board = np.zeros((11, 11), dtype=np.int8)
    
    # Put king in center
    game.board[5, 5] = KING
    
    # Create a box with one gap
    for r in range(3, 8):
        for c in range(3, 8):
            if r in [3, 7] or c in [3, 7]:
                game.board[r, c] = ATTACKER
    
    # Create a breach
    game.board[3, 5] = EMPTY
    
    print(game)
    is_encircled = game._is_king_encircled()
    print(f"King encircled: {is_encircled}")
    assert not is_encircled, "King should NOT be encircled with a breach"
    print("✓ PASSED\n")

def test_path_to_edge():
    """Test that king can reach edge through empty squares."""
    game = Hnefatafl()
    print("Test 5: Path to edge through empty squares")
    
    # Clear the board
    game.board = np.zeros((11, 11), dtype=np.int8)
    
    # Put king off-center with a clear path to edge
    game.board[5, 5] = KING
    
    # Put attackers that don't form complete encirclement
    game.board[4, 4] = ATTACKER
    game.board[4, 6] = ATTACKER
    game.board[6, 4] = ATTACKER
    game.board[6, 6] = ATTACKER
    
    print(game)
    is_encircled = game._is_king_encircled()
    print(f"King encircled: {is_encircled}")
    assert not is_encircled, "King should reach edge through gaps"
    print("✓ PASSED\n")

def test_complex_shape():
    """Test a more complex irregular encirclement."""
    game = Hnefatafl()
    print("Test 6: Complex irregular encirclement")
    
    # Clear the board
    game.board = np.zeros((11, 11), dtype=np.int8)
    
    # Put king in center
    game.board[5, 5] = KING
    
    # Create an irregular but complete encirclement
    # Using a diamond-ish shape
    attacker_positions = [
        (2, 5), (3, 4), (3, 6), (4, 3), (4, 7),
        (5, 2), (5, 8), (6, 3), (6, 7), (7, 4), (7, 6), (8, 5)
    ]
    for r, c in attacker_positions:
        game.board[r, c] = ATTACKER
    
    print(game)
    is_encircled = game._is_king_encircled()
    print(f"King encircled: {is_encircled}")
    assert is_encircled, "King should be encircled by irregular shape"
    print("✓ PASSED\n")

def test_encirclement_triggers_game_over():
    """Test that encirclement actually ends the game."""
    game = Hnefatafl()
    print("Test 7: Encirclement triggers game over")
    
    # Clear the board
    game.board = np.zeros((11, 11), dtype=np.int8)
    
    # Put king in center
    game.board[5, 5] = KING
    
    # Create encirclement
    for r in range(4, 7):
        for c in range(4, 7):
            if r in [4, 6] or c in [4, 6]:
                if (r, c) != (5, 5):
                    game.board[r, c] = ATTACKER
    
    print(game)
    
    # Check game over
    game._check_game_over()
    
    print(f"Game over: {game.game_over}")
    print(f"Winner: {'Attackers' if game.winner == 0 else 'Defenders'}")
    assert game.game_over, "Game should be over"
    assert game.winner == 0, "Attackers should win by encirclement"
    print("✓ PASSED\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hnefatafl Encirclement Rule")
    print("=" * 60)
    print()
    
    test_no_encirclement_start()
    test_simple_encirclement()
    test_encirclement_with_defenders()
    test_breach_in_wall()
    test_path_to_edge()
    test_complex_shape()
    test_encirclement_triggers_game_over()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
