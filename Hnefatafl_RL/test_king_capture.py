"""
Test script to verify king capture between two attackers works correctly.
"""

from brandubh import Brandubh, ATTACKER, DEFENDER, KING, EMPTY

def test_basic_king_capture():
    """Test that king can be captured between two attackers."""
    game = Brandubh()
    
    # Clear the board and set up a simple capture scenario
    game.board[:, :] = EMPTY
    
    # Place king at (3, 3)
    game.board[3, 3] = KING
    
    # Place attacker at (3, 2) - left of king
    game.board[3, 2] = ATTACKER
    
    # Place attacker at (3, 4) - right of king
    game.board[3, 4] = ATTACKER
    
    # Place another attacker at (2, 3) that will move to capture
    game.board[2, 3] = ATTACKER
    
    game.current_player = 0  # Attacker's turn
    game.game_over = False
    game.winner = None
    
    print("Initial board:")
    print(game)
    print(f"\nKing is at (3, 3)")
    print(f"Attacker pieces at: (3, 2), (3, 4), (2, 3)")
    print(f"\nAttempting to move attacker from (2, 3) to (3, 1)...")
    print("This should NOT capture the king (no sandwich).\n")
    
    # Try moving the attacker to (3, 1) - this should NOT capture
    game.make_move((2, 3, 3, 1))
    print("After move:")
    print(game)
    print(f"Game over: {game.game_over}, Winner: {game.winner}")
    print(f"King still on board: {KING in game.board}\n")
    
    # Reset for actual capture test
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING
    game.board[3, 4] = ATTACKER  # Already on right side
    game.board[2, 2] = ATTACKER  # Will move to sandwich
    game.current_player = 0
    game.game_over = False
    game.winner = None
    
    print("\n" + "="*50)
    print("SECOND TEST - Actual capture")
    print("="*50)
    print("\nInitial board:")
    print(game)
    print(f"\nKing is at (3, 3)")
    print(f"Attacker at (3, 4) - right of king")
    print(f"Attacker at (2, 2) - will move to (3, 2) to sandwich king")
    print(f"\nAttempting to move attacker from (2, 2) to (3, 2)...")
    print("This SHOULD capture the king!\n")
    
    # Move attacker to create sandwich
    result = game.make_move((2, 2, 3, 2))
    
    print(f"Move result: {result}")
    print("After move:")
    print(game)
    print(f"\nGame over: {game.game_over}")
    print(f"Winner: {'Attacker' if game.winner == 0 else 'Defender' if game.winner == 1 else None}")
    print(f"King still on board: {KING in game.board}")
    
    if game.game_over and game.winner == 0 and KING not in game.board:
        print("\n✓ SUCCESS: King was captured between two attackers!")
    else:
        print("\n✗ FAILURE: King capture did not work as expected")
        print(f"  Expected: game_over=True, winner=0, king removed")
        print(f"  Got: game_over={game.game_over}, winner={game.winner}, king present={KING in game.board}")

if __name__ == "__main__":
    test_basic_king_capture()
