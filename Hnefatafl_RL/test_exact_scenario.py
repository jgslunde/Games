"""
Test the EXACT scenario from the user's image.
King (yellow with crown) at center, two black attackers on left and right.
"""

from brandubh import Brandubh, ATTACKER, KING, EMPTY, ATTACKER_PLAYER

def test_exact_image_scenario():
    """
    Recreate the exact scenario from the image:
    - King at (3, 3) - throne position
    - Attacker at (3, 2) - left of king
    - Attacker at (3, 4) - right of king
    
    One of these attackers was just moved this turn to complete the sandwich.
    """
    game = Brandubh()
    
    # Set up the exact scenario
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING  # King at center/throne
    game.board[3, 2] = ATTACKER  # Left attacker (already there)
    game.board[2, 4] = ATTACKER  # Right attacker (will move to sandwich)
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    game.winner = None
    
    print("="*70)
    print("EXACT SCENARIO FROM IMAGE")
    print("="*70)
    print("\nBEFORE the capturing move:")
    print(game)
    print(f"\nKing is at (3, 3) - the throne")
    print(f"Attacker already at (3, 2) - LEFT of king")
    print(f"Attacker at (2, 4) will move to (3, 4) - RIGHT of king")
    print(f"\nThis should create: ATTACKER - KING - ATTACKER")
    
    # Check capture logic step by step
    print("\n" + "-"*70)
    print("MAKING THE MOVE (2, 4) -> (3, 4)")
    print("-"*70)
    
    move = (2, 4, 3, 4)
    
    # Verify move is legal
    legal_moves = game.get_legal_moves()
    print(f"\nIs move legal? {move in legal_moves}")
    
    if move not in legal_moves:
        print("ERROR: Move is not in legal moves list!")
        print(f"Legal moves from (2, 4): {[m for m in legal_moves if m[0] == 2 and m[1] == 4]}")
        return False
    
    # Make the move
    print(f"\nMaking move: {move}")
    success = game.make_move(move)
    print(f"Move returned: {success}")
    
    print("\nAFTER the move:")
    print(game)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"Game over: {game.game_over}")
    print(f"Winner: {game.winner} (0=Attacker, 1=Defender, None=ongoing)")
    print(f"King still on board: {KING in game.board}")
    print(f"Board at (3,2): {game.board[3, 2]} (should be ATTACKER={ATTACKER})")
    print(f"Board at (3,3): {game.board[3, 3]} (should be EMPTY={EMPTY} if captured)")
    print(f"Board at (3,4): {game.board[3, 4]} (should be ATTACKER={ATTACKER})")
    
    # Manual check of what should happen
    print("\n" + "="*70)
    print("MANUAL VERIFICATION:")
    print("="*70)
    print(f"After move to (3,4), checking captures in all 4 directions from (3,4):")
    
    # The move was to (3, 4), so we check from there
    # Direction: LEFT (dr=0, dc=-1)
    print(f"\n  Checking LEFT from (3,4):")
    print(f"    Position (3,3) contains: {game.board[3, 3]} (KING={KING}, EMPTY={EMPTY})")
    print(f"    Position (3,2) contains: {game.board[3, 2]} (ATTACKER={ATTACKER})")
    print(f"    This SHOULD trigger capture: ATTACKER(3,4) - KING(3,3) - ATTACKER(3,2)")
    
    if game.game_over and game.winner == ATTACKER_PLAYER and KING not in game.board:
        print("\n✓✓✓ SUCCESS - King was captured correctly! ✓✓✓")
        return True
    else:
        print("\n✗✗✗ FAILURE - King was NOT captured! ✗✗✗")
        print("\nThis is the exact bug the user reported.")
        return False

def test_reverse_scenario():
    """Test moving the left attacker instead of the right one."""
    game = Brandubh()
    
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING  # King at center
    game.board[3, 4] = ATTACKER  # Right attacker (already there)
    game.board[2, 2] = ATTACKER  # Left attacker (will move to sandwich)
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    game.winner = None
    
    print("\n\n" + "="*70)
    print("REVERSE SCENARIO - Move left attacker instead")
    print("="*70)
    print("\nBEFORE:")
    print(game)
    
    move = (2, 2, 3, 2)
    print(f"\nMoving attacker from (2,2) to (3,2) to sandwich king")
    game.make_move(move)
    
    print("\nAFTER:")
    print(game)
    
    if game.game_over and game.winner == ATTACKER_PLAYER:
        print("\n✓ Reverse scenario works")
        return True
    else:
        print("\n✗ Reverse scenario also fails!")
        return False

if __name__ == "__main__":
    result1 = test_exact_image_scenario()
    result2 = test_reverse_scenario()
    
    if not result1 or not result2:
        print("\n" + "!"*70)
        print("BUG CONFIRMED - King capture is not working as expected!")
        print("!"*70)
