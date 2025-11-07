"""
Test king capture in a scenario similar to what would happen in GUI.
"""

from brandubh import Brandubh, ATTACKER, KING, EMPTY, ATTACKER_PLAYER

def test_gui_like_scenario():
    """Test king capture in a fresh game with realistic setup."""
    game = Brandubh()
    
    print("Starting from initial position")
    print(game)
    
    # Simulate a game where attackers try to capture the king
    # Let's create a simple scenario where king is in danger
    
    print("\n" + "="*60)
    print("Setting up a capture scenario...")
    print("="*60)
    
    # Clear and set up simple scenario
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING  # King in center
    game.board[3, 4] = ATTACKER  # Attacker on the right
    game.board[1, 2] = ATTACKER  # Attacker that will move to sandwich
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    game.winner = None
    game.king_has_left_throne = False
    
    print("\nSetup:")
    print(game)
    print(f"\nKing at (3, 3) (throne)")
    print(f"Attacker at (3, 4) - right of king")
    print(f"Attacker at (1, 2) - will move to (3, 2) to sandwich")
    
    legal_moves = game.get_legal_moves()
    target_move = (1, 2, 3, 2)
    
    print(f"\nLegal moves from (1, 2): {[m for m in legal_moves if m[0] == 1 and m[1] == 2]}")
    print(f"\nIs target move {target_move} legal? {target_move in legal_moves}")
    
    # Make the capture move
    print(f"\nMaking move: {target_move}")
    success = game.make_move(target_move)
    
    print(f"Move success: {success}")
    print("\nBoard after move:")
    print(game)
    print(f"\nGame over: {game.game_over}")
    print(f"Winner: {game.winner} (0=Attacker, 1=Defender)")
    print(f"King on board: {KING in game.board}")
    
    # Verify
    if game.game_over and game.winner == ATTACKER_PLAYER and KING not in game.board:
        print("\n✓ SUCCESS: King captured correctly!")
        return True
    else:
        print("\n✗ FAILURE: King not captured")
        print(f"  game_over={game.game_over}, winner={game.winner}, king_present={KING in game.board}")
        return False

def test_vertical_capture():
    """Test vertical king capture."""
    game = Brandubh()
    
    print("\n" + "="*60)
    print("Testing VERTICAL capture (top-bottom sandwich)")
    print("="*60)
    
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING  # King in center
    game.board[2, 3] = ATTACKER  # Attacker above king
    game.board[5, 3] = ATTACKER  # Attacker that will move below king
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    game.winner = None
    
    print("\nSetup:")
    print(game)
    print("King at (3, 3), Attacker at (2, 3) above")
    print("Moving attacker from (5, 3) to (4, 3) to sandwich vertically")
    
    move = (5, 3, 4, 3)
    success = game.make_move(move)
    
    print(f"\nMove success: {success}")
    print("After move:")
    print(game)
    
    if game.game_over and game.winner == ATTACKER_PLAYER and KING not in game.board:
        print("\n✓ SUCCESS: Vertical capture works!")
        return True
    else:
        print("\n✗ FAILURE: Vertical capture failed")
        return False

if __name__ == "__main__":
    result1 = test_gui_like_scenario()
    result2 = test_vertical_capture()
    
    if result1 and result2:
        print("\n" + "="*60)
        print("ALL TESTS PASSED - King capture works correctly!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED - Check implementation")
        print("="*60)
