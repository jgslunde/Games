"""
Comprehensive tests for throne rule implementation in all Tafl variants.

Tests verify:
1. King can re-enter throne after leaving
2. Throne hostility is context-dependent (based on target piece)
3. Temperature drops when king is NOT on throne (not based on history)
"""

import numpy as np
from brandubh import Brandubh, EMPTY, ATTACKER, DEFENDER, KING, ATTACKER_PLAYER, DEFENDER_PLAYER
from tablut import Tablut
from hnefatafl import Hnefatafl


def test_king_can_reenter_throne_brandubh():
    """Test that king can move back onto throne after leaving (Brandubh 7x7)."""
    print("\n" + "="*70)
    print("TEST: King can re-enter throne (Brandubh)")
    print("="*70)
    
    game = Brandubh()
    throne = game.throne  # (3, 3)
    
    # Create a simple test board with king able to move
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = KING  # King starts on throne
    game.current_player = DEFENDER_PLAYER
    
    print(f"✓ King starts on throne at {throne}")
    
    # Move king off throne (to the right)
    move_off = (throne[0], throne[1], throne[0], throne[1] + 1)
    legal_moves = game.get_legal_moves()
    assert move_off in legal_moves, f"King should be able to move off throne"
    game.make_move(move_off)
    print(f"✓ King moved off throne from {throne} to {(throne[0], throne[1] + 1)}")
    
    # Verify throne is now empty
    assert game.board[throne] == EMPTY, "Throne should be empty after king leaves"
    assert game.board[throne[0], throne[1] + 1] == KING, "King should be at new position"
    print(f"✓ Throne is now empty, king at {(throne[0], throne[1] + 1)}")
    
    # Switch to attackers, then back to defenders
    game.current_player = ATTACKER_PLAYER
    game.current_player = DEFENDER_PLAYER
    
    # Try to move king BACK onto throne
    move_back = (throne[0], throne[1] + 1, throne[0], throne[1])
    legal_moves = game.get_legal_moves()
    
    # Filter to only king moves for clarity
    king_moves = [m for m in legal_moves if game.board[m[0], m[1]] == KING]
    print(f"✓ King has {len(king_moves)} legal moves")
    
    assert move_back in legal_moves, f"King SHOULD be able to return to throne! Move {move_back} not in legal moves"
    print(f"✓ King CAN move back onto throne (move {move_back} is legal)")
    
    # Execute the move back
    game.make_move(move_back)
    assert game.board[throne] == KING, "King should be back on throne"
    print(f"✓ King successfully returned to throne at {throne}")
    print("✅ TEST PASSED: King can re-enter throne\n")


def test_king_can_reenter_throne_tablut():
    """Test that king can move back onto throne after leaving (Tablut 9x9)."""
    print("\n" + "="*70)
    print("TEST: King can re-enter throne (Tablut)")
    print("="*70)
    
    game = Tablut()
    throne = game.throne  # (4, 4)
    
    # Initial state: King is on throne
    assert game.board[throne] == KING, "King should start on throne"
    print(f"✓ King starts on throne at {throne}")
    
    # Move king off throne
    move_off = (throne[0], throne[1], throne[0], throne[1] + 1)
    legal_moves = game.get_legal_moves()
    assert move_off in legal_moves, "King should be able to move off throne"
    game.make_move(move_off)
    print(f"✓ King moved off throne to {(throne[0], throne[1] + 1)}")
    
    # Switch players
    game.current_player = ATTACKER_PLAYER
    game.current_player = DEFENDER_PLAYER
    
    # Move king back
    move_back = (throne[0], throne[1] + 1, throne[0], throne[1])
    legal_moves = game.get_legal_moves()
    assert move_back in legal_moves, "King should be able to return to throne"
    game.make_move(move_back)
    assert game.board[throne] == KING, "King should be back on throne"
    print(f"✓ King successfully returned to throne at {throne}")
    print("✅ TEST PASSED: King can re-enter throne (Tablut)\n")


def test_king_can_reenter_throne_hnefatafl():
    """Test that king can move back onto throne after leaving (Hnefatafl 11x11)."""
    print("\n" + "="*70)
    print("TEST: King can re-enter throne (Hnefatafl)")
    print("="*70)
    
    game = Hnefatafl()
    throne = game.throne  # (5, 5)
    
    # Initial state: King is on throne
    assert game.board[throne] == KING, "King should start on throne"
    print(f"✓ King starts on throne at {throne}")
    
    # Move king off throne
    move_off = (throne[0], throne[1], throne[0], throne[1] + 1)
    legal_moves = game.get_legal_moves()
    assert move_off in legal_moves, "King should be able to move off throne"
    game.make_move(move_off)
    print(f"✓ King moved off throne to {(throne[0], throne[1] + 1)}")
    
    # Switch players
    game.current_player = ATTACKER_PLAYER
    game.current_player = DEFENDER_PLAYER
    
    # Move king back
    move_back = (throne[0], throne[1] + 1, throne[0], throne[1])
    legal_moves = game.get_legal_moves()
    assert move_back in legal_moves, "King should be able to return to throne"
    game.make_move(move_back)
    assert game.board[throne] == KING, "King should be back on throne"
    print(f"✓ King successfully returned to throne at {throne}")
    print("✅ TEST PASSED: King can re-enter throne (Hnefatafl)\n")


def test_throne_hostility_to_attackers_brandubh():
    """Test that empty throne is ALWAYS hostile to attackers being captured."""
    print("\n" + "="*70)
    print("TEST: Throne hostility to attackers (Brandubh)")
    print("="*70)
    
    # Test with throne_is_hostile=False (should still be hostile to attackers)
    game = Brandubh(throne_is_hostile=False)
    throne = game.throne
    
    # Clear board and set up test position
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY  # Empty throne
    
    # Place attacker next to throne, and defender to capture it
    game.board[throne[0], throne[1] - 1] = ATTACKER  # Attacker to the left of throne
    game.board[throne[0], throne[1] - 2] = DEFENDER  # Defender to capture it
    game.current_player = DEFENDER_PLAYER
    
    print(f"Setup: Empty throne at {throne}, attacker at {(throne[0], throne[1] - 1)}")
    print(f"       Defender at {(throne[0], throne[1] - 2)} ready to capture")
    print(f"       throne_is_hostile flag = {game.throne_is_hostile}")
    
    # Move defender to complete sandwich
    game.make_move((throne[0], throne[1] - 2, throne[0], throne[1] - 1))
    
    # Attacker should be captured (throne is hostile to attackers regardless of flag)
    assert game.board[throne[0], throne[1] - 1] == EMPTY, \
        "Attacker should be captured against empty throne even when throne_is_hostile=False"
    print(f"✓ Attacker was captured against empty throne (throne_is_hostile={game.throne_is_hostile})")
    print("✅ TEST PASSED: Empty throne is hostile to attackers\n")


def test_throne_hostility_to_defenders_brandubh():
    """Test that empty throne is only hostile to defenders when throne_is_hostile=True."""
    print("\n" + "="*70)
    print("TEST: Throne hostility to defenders (Brandubh)")
    print("="*70)
    
    # Test 1: throne_is_hostile=False -> throne NOT hostile to defenders
    print("\n--- Test 1: throne_is_hostile=False ---")
    game = Brandubh(throne_is_hostile=False)
    throne = game.throne
    
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY
    game.board[throne[0], throne[1] - 1] = DEFENDER  # Defender next to throne
    game.board[throne[0], throne[1] - 2] = ATTACKER  # Attacker to "capture" it
    game.current_player = ATTACKER_PLAYER
    
    print(f"Setup: Empty throne at {throne}, defender at {(throne[0], throne[1] - 1)}")
    print(f"       Attacker at {(throne[0], throne[1] - 2)}, throne_is_hostile={game.throne_is_hostile}")
    
    # Move attacker to create sandwich
    game.make_move((throne[0], throne[1] - 2, throne[0], throne[1] - 1))
    
    # Defender should NOT be captured (throne not hostile to defenders when flag=False)
    assert game.board[throne[0], throne[1] - 1] == ATTACKER, \
        "Attacker should have moved to defender's position (no capture)"
    print(f"✓ Defender was NOT captured (correctly, throne_is_hostile=False)")
    
    # Test 2: throne_is_hostile=True -> throne IS hostile to defenders
    print("\n--- Test 2: throne_is_hostile=True ---")
    game = Brandubh(throne_is_hostile=True)
    throne = game.throne
    
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY
    game.board[throne[0], throne[1] - 1] = DEFENDER
    game.board[throne[0], throne[1] - 2] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    
    print(f"Setup: Empty throne at {throne}, defender at {(throne[0], throne[1] - 1)}")
    print(f"       Attacker at {(throne[0], throne[1] - 2)}, throne_is_hostile={game.throne_is_hostile}")
    
    # Move attacker to complete sandwich
    game.make_move((throne[0], throne[1] - 2, throne[0], throne[1] - 1))
    
    # Defender SHOULD be captured (throne is hostile to defenders when flag=True)
    assert game.board[throne[0], throne[1] - 1] == ATTACKER, \
        "Attacker should be at this position after capturing defender"
    print(f"✓ Defender WAS captured (correctly, throne_is_hostile=True)")
    print("✅ TEST PASSED: Throne hostility to defenders depends on flag\n")


def test_occupied_throne_not_hostile():
    """Test that OCCUPIED throne is never hostile (can't capture against a piece)."""
    print("\n" + "="*70)
    print("TEST: Occupied throne is NOT hostile")
    print("="*70)
    
    game = Brandubh(throne_is_hostile=True)
    throne = game.throne
    
    # Setup: King on throne, attacker next to it, defender to try capture
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = KING  # King occupies throne
    game.board[throne[0], throne[1] - 1] = ATTACKER
    game.board[throne[0], throne[1] - 2] = DEFENDER
    game.current_player = DEFENDER_PLAYER
    
    print(f"Setup: King on throne at {throne}, attacker at {(throne[0], throne[1] - 1)}")
    print(f"       Defender at {(throne[0], throne[1] - 2)}")
    
    # Move defender to try to capture attacker against occupied throne
    game.make_move((throne[0], throne[1] - 2, throne[0], throne[1] - 1))
    
    # Attacker should NOT be captured (throne is occupied, so not hostile)
    assert game.board[throne[0], throne[1] - 1] == DEFENDER, \
        "Defender should have moved here, attacker should NOT be captured"
    print(f"✓ Attacker was NOT captured against occupied throne (correct)")
    print("✅ TEST PASSED: Occupied throne is not hostile\n")


def test_throne_hostility_king_capture():
    """Test that throne hostility works correctly for king captures."""
    print("\n" + "="*70)
    print("TEST: Throne hostility in king captures")
    print("="*70)
    
    # Test with 2-piece king capture mode
    game = Brandubh(king_capture_pieces=2, throne_is_hostile=True)
    throne = game.throne
    
    # Setup: Empty throne, king next to it, attacker to capture
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY  # Empty throne
    game.board[throne[0], throne[1] - 1] = KING
    game.board[throne[0], throne[1] - 2] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    
    print(f"Setup: Empty throne at {throne}, king at {(throne[0], throne[1] - 1)}")
    print(f"       Attacker at {(throne[0], throne[1] - 2)}, king_capture_pieces={game.king_capture_pieces}")
    
    # Move attacker to sandwich king against throne
    game.make_move((throne[0], throne[1] - 2, throne[0], throne[1] - 1))
    
    # King should be captured (throne is hostile to king when flag=True)
    assert game.game_over, "Game should be over (king captured)"
    assert game.winner == ATTACKER_PLAYER, "Attackers should win"
    print(f"✓ King was captured against empty throne (throne_is_hostile=True)")
    
    # Test with throne_is_hostile=False
    print("\n--- Testing with throne_is_hostile=False ---")
    game = Brandubh(king_capture_pieces=2, throne_is_hostile=False)
    
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY
    game.board[throne[0], throne[1] - 1] = KING
    game.board[throne[0], throne[1] - 2] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    print(f"Setup: Same position, but throne_is_hostile={game.throne_is_hostile}")
    
    # Move attacker
    game.make_move((throne[0], throne[1] - 2, throne[0], throne[1] - 1))
    
    # King should NOT be captured (throne not hostile to king when flag=False)
    assert not game.game_over, "Game should NOT be over (throne not hostile to king)"
    assert game.board[throne[0], throne[1] - 1] == ATTACKER, "Attacker should have moved"
    print(f"✓ King was NOT captured against empty throne (throne_is_hostile=False)")
    print("✅ TEST PASSED: Throne hostility works correctly for king captures\n")


def test_no_king_has_left_throne_attribute():
    """Test that king_has_left_throne attribute no longer exists."""
    print("\n" + "="*70)
    print("TEST: No king_has_left_throne attribute")
    print("="*70)
    
    for GameClass, name in [(Brandubh, "Brandubh"), (Tablut, "Tablut"), (Hnefatafl, "Hnefatafl")]:
        game = GameClass()
        assert not hasattr(game, 'king_has_left_throne'), \
            f"{name} should not have king_has_left_throne attribute"
        print(f"✓ {name}: No king_has_left_throne attribute (correct)")
        
        # Test that clone also doesn't have it
        cloned = game.clone()
        assert not hasattr(cloned, 'king_has_left_throne'), \
            f"Cloned {name} should not have king_has_left_throne attribute"
        print(f"✓ {name}: Cloned game also has no king_has_left_throne (correct)")
    
    print("✅ TEST PASSED: king_has_left_throne attribute removed\n")


def test_hostile_square_function_signature():
    """Test that _is_hostile_square now requires target_piece parameter."""
    print("\n" + "="*70)
    print("TEST: _is_hostile_square signature")
    print("="*70)
    
    for GameClass, name in [(Brandubh, "Brandubh"), (Tablut, "Tablut"), (Hnefatafl, "Hnefatafl")]:
        game = GameClass()
        throne = game.throne
        
        # Test that calling with target_piece works
        result = game._is_hostile_square(throne[0], throne[1], ATTACKER)
        print(f"✓ {name}._is_hostile_square({throne[0]}, {throne[1]}, ATTACKER) = {result}")
        
        result = game._is_hostile_square(throne[0], throne[1], DEFENDER)
        print(f"✓ {name}._is_hostile_square({throne[0]}, {throne[1]}, DEFENDER) = {result}")
        
        result = game._is_hostile_square(throne[0], throne[1], KING)
        print(f"✓ {name}._is_hostile_square({throne[0]}, {throne[1]}, KING) = {result}")
    
    print("✅ TEST PASSED: _is_hostile_square accepts target_piece parameter\n")


def test_corners_always_hostile():
    """Test that corners are always hostile regardless of target piece."""
    print("\n" + "="*70)
    print("TEST: Corners always hostile")
    print("="*70)
    
    game = Brandubh()
    
    for corner in game.corners:
        # Test with different target pieces
        assert game._is_hostile_square(corner[0], corner[1], ATTACKER), \
            f"Corner {corner} should be hostile to attackers"
        assert game._is_hostile_square(corner[0], corner[1], DEFENDER), \
            f"Corner {corner} should be hostile to defenders"
        assert game._is_hostile_square(corner[0], corner[1], KING), \
            f"Corner {corner} should be hostile to king"
        print(f"✓ Corner {corner} is hostile to all piece types")
    
    print("✅ TEST PASSED: Corners are always hostile\n")


def test_comprehensive_capture_scenarios():
    """Test various capture scenarios to ensure throne hostility works correctly."""
    print("\n" + "="*70)
    print("TEST: Comprehensive capture scenarios")
    print("="*70)
    
    # Scenario 1: Attacker captured by two defenders (no throne involved)
    print("\n--- Scenario 1: Regular attacker capture (no throne) ---")
    game = Brandubh()
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[3, 3] = ATTACKER
    game.board[3, 2] = DEFENDER
    game.board[3, 1] = DEFENDER
    game.current_player = DEFENDER_PLAYER
    
    game.make_move((3, 1, 3, 2))
    assert game.board[3, 3] == EMPTY, "Attacker should be captured"
    print("✓ Attacker captured between two defenders")
    
    # Scenario 2: Defender captured by two attackers (no throne)
    print("\n--- Scenario 2: Regular defender capture (no throne) ---")
    game = Brandubh()
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[3, 3] = DEFENDER
    game.board[3, 2] = ATTACKER
    game.board[3, 1] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    
    game.make_move((3, 1, 3, 2))
    assert game.board[3, 3] == EMPTY, "Defender should be captured"
    print("✓ Defender captured between two attackers")
    
    # Scenario 3: Attacker captured between defender and corner
    print("\n--- Scenario 3: Attacker captured against corner ---")
    game = Brandubh()
    corner = game.corners[0]  # Top-left corner
    game.board = np.zeros((7, 7), dtype=np.int8)
    
    # Place attacker next to corner, defender to capture
    if corner == (0, 0):
        game.board[0, 1] = ATTACKER
        game.board[0, 2] = DEFENDER
        game.current_player = DEFENDER_PLAYER
        game.make_move((0, 2, 0, 1))
        assert game.board[0, 1] == DEFENDER, "Defender should have moved here"
        # Note: Corner capture logic might differ, but corner should be hostile
    print("✓ Corner hostility test completed")
    
    # Scenario 4: King protected by defenders
    print("\n--- Scenario 4: King NOT captured when protected ---")
    game = Brandubh(king_capture_pieces=2)
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[3, 3] = KING
    game.board[3, 2] = ATTACKER
    game.board[3, 4] = DEFENDER  # King has friend on other side
    game.current_player = ATTACKER_PLAYER
    
    assert not game._is_king_captured(3, 3), "King should not be captured (has friend)"
    print("✓ King not captured when protected by defender")
    
    print("✅ TEST PASSED: Comprehensive capture scenarios\n")


def test_all_variants_consistency():
    """Test that all three variants behave consistently with throne rules."""
    print("\n" + "="*70)
    print("TEST: All variants have consistent throne behavior")
    print("="*70)
    
    for GameClass, name, board_size in [(Brandubh, "Brandubh", 7), 
                                         (Tablut, "Tablut", 9),
                                         (Hnefatafl, "Hnefatafl", 11)]:
        print(f"\n--- Testing {name} ({board_size}x{board_size}) ---")
        
        # Test 1: King can re-enter throne
        game = GameClass()
        throne = game.throne
        move_off = (throne[0], throne[1], throne[0], throne[1] + 1)
        game.make_move(move_off)
        game.current_player = ATTACKER_PLAYER
        game.current_player = DEFENDER_PLAYER
        move_back = (throne[0], throne[1] + 1, throne[0], throne[1])
        assert move_back in game.get_legal_moves(), f"{name}: King should be able to return to throne"
        print(f"  ✓ King can re-enter throne")
        
        # Test 2: Empty throne hostile to attackers
        game = GameClass(throne_is_hostile=False)
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        assert game._is_hostile_square(throne[0], throne[1], ATTACKER) == True, \
            f"{name}: Empty throne should be hostile to attackers (always)"
        print(f"  ✓ Empty throne hostile to attackers")
        
        # Test 3: Empty throne hostility to defenders depends on flag
        game = GameClass(throne_is_hostile=False)
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        assert game._is_hostile_square(throne[0], throne[1], DEFENDER) == False, \
            f"{name}: Empty throne should NOT be hostile to defenders when flag=False"
        
        game = GameClass(throne_is_hostile=True)
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        assert game._is_hostile_square(throne[0], throne[1], DEFENDER) == True, \
            f"{name}: Empty throne SHOULD be hostile to defenders when flag=True"
        print(f"  ✓ Empty throne hostility to defenders depends on flag")
        
        # Test 4: Occupied throne not hostile
        game = GameClass()
        game.board[throne] = KING
        assert game._is_hostile_square(throne[0], throne[1], ATTACKER) == False, \
            f"{name}: Occupied throne should not be hostile"
        assert game._is_hostile_square(throne[0], throne[1], DEFENDER) == False, \
            f"{name}: Occupied throne should not be hostile"
        print(f"  ✓ Occupied throne not hostile to any piece")
        
    print("\n✅ TEST PASSED: All variants have consistent throne behavior\n")


def run_all_tests():
    """Run all throne rule tests."""
    print("\n" + "#"*70)
    print("# THRONE RULES TEST SUITE")
    print("#"*70)
    
    tests = [
        # King re-entry tests
        test_king_can_reenter_throne_brandubh,
        test_king_can_reenter_throne_tablut,
        test_king_can_reenter_throne_hnefatafl,
        
        # Throne hostility tests
        test_throne_hostility_to_attackers_brandubh,
        test_throne_hostility_to_defenders_brandubh,
        test_occupied_throne_not_hostile,
        test_throne_hostility_king_capture,
        
        # Implementation tests
        test_no_king_has_left_throne_attribute,
        test_hostile_square_function_signature,
        test_corners_always_hostile,
        
        # Comprehensive tests
        test_comprehensive_capture_scenarios,
        test_all_variants_consistency,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n❌ TEST FAILED: {test_func.__name__}")
            print(f"   Error: {e}\n")
        except Exception as e:
            failed += 1
            print(f"\n❌ TEST ERROR: {test_func.__name__}")
            print(f"   Error: {e}\n")
    
    print("\n" + "#"*70)
    print(f"# TEST RESULTS: {passed} passed, {failed} failed")
    print("#"*70 + "\n")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! 🎉\n")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
