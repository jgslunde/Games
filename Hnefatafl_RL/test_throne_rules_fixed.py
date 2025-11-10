"""
Comprehensive tests for throne rule implementation in all Tafl variants.

Tests verify:
1. King can re-enter throne after leaving
2. Throne hostility is context-dependent (based on target piece)
3. Occupied throne is never hostile
"""

import numpy as np
from brandubh import Brandubh, EMPTY, ATTACKER, DEFENDER, KING, ATTACKER_PLAYER, DEFENDER_PLAYER
from tablut import Tablut
from hnefatafl import Hnefatafl


def test_king_can_reenter_throne_all_variants():
    """Test that king can move back onto throne after leaving (all variants)."""
    print("\n" + "="*70)
    print("TEST: King can re-enter throne (all variants)")
    print("="*70)
    
    for GameClass, name, board_size in [(Brandubh, "Brandubh", 7), 
                                         (Tablut, "Tablut", 9),
                                         (Hnefatafl, "Hnefatafl", 11)]:
        print(f"\n--- Testing {name} ---")
        game = GameClass()
        throne = game.throne
        
        # Create simple test board with just the king
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        game.board[throne] = KING
        game.current_player = DEFENDER_PLAYER
        
        print(f"  King starts on throne at {throne}")
        
        # Move king off throne
        move_off = (throne[0], throne[1], throne[0], throne[1] + 1)
        assert move_off in game.get_legal_moves(), f"{name}: King should be able to move off throne"
        game.make_move(move_off)
        print(f"  ✓ King moved off throne")
        
        # Verify throne is empty
        assert game.board[throne] == EMPTY, f"{name}: Throne should be empty"
        assert game.board[throne[0], throne[1] + 1] == KING, f"{name}: King should be at new position"
        
        # Switch players twice to get back to defenders
        game.current_player = ATTACKER_PLAYER
        game.current_player = DEFENDER_PLAYER
        
        # Move king BACK onto throne
        move_back = (throne[0], throne[1] + 1, throne[0], throne[1])
        legal_moves = game.get_legal_moves()
        assert move_back in legal_moves, f"{name}: King SHOULD be able to return to throne"
        game.make_move(move_back)
        assert game.board[throne] == KING, f"{name}: King should be back on throne"
        print(f"  ✓ King successfully returned to throne")
    
    print("\n✅ TEST PASSED: King can re-enter throne\n")


def test_throne_hostility_to_attackers():
    """Test that empty throne is ALWAYS hostile to attackers being captured."""
    print("\n" + "="*70)
    print("TEST: Throne hostility to attackers")
    print("="*70)
    
    # Test with throne_is_hostile=False (should still be hostile to attackers)
    game = Brandubh(throne_is_hostile=False)
    throne = game.throne
    
    # Setup: Defender - Attacker - Empty Throne
    # Defender will move to sandwich attacker
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY
    game.board[throne[0], throne[1] - 1] = ATTACKER  # Attacker next to throne
    game.board[throne[0], throne[1] - 3] = DEFENDER  # Defender two spaces away
    game.current_player = DEFENDER_PLAYER
    
    print(f"Setup: Defender - space - Attacker - Throne")
    print(f"       throne_is_hostile flag = {game.throne_is_hostile}")
    
    # Move defender to sandwich attacker against throne
    move = (throne[0], throne[1] - 3, throne[0], throne[1] - 2)
    game.make_move(move)
    
    # Attacker should be captured (throne is hostile to attackers regardless of flag)
    assert game.board[throne[0], throne[1] - 1] == EMPTY, \
        "Attacker should be captured against empty throne even when throne_is_hostile=False"
    print("✓ Attacker was captured against empty throne (throne_is_hostile=False)")
    print("✅ TEST PASSED: Empty throne is always hostile to attackers\n")


def test_throne_hostility_to_defenders():
    """Test that empty throne is only hostile to defenders when throne_is_hostile=True."""
    print("\n" + "="*70)
    print("TEST: Throne hostility to defenders")
    print("="*70)
    
    # Test 1: throne_is_hostile=False -> throne NOT hostile to defenders
    print("\n--- Test 1: throne_is_hostile=False ---")
    game = Brandubh(throne_is_hostile=False)
    throne = game.throne
    
    # Setup: Attacker - Defender - Empty Throne
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY
    game.board[throne[0], throne[1] - 1] = DEFENDER
    game.board[throne[0], throne[1] - 3] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    
    print(f"Setup: Attacker - space - Defender - Throne, throne_is_hostile={game.throne_is_hostile}")
    
    # Move attacker to try to sandwich defender
    move = (throne[0], throne[1] - 3, throne[0], throne[1] - 2)
    game.make_move(move)
    
    # Defender should NOT be captured (throne not hostile to defenders when flag=False)
    assert game.board[throne[0], throne[1] - 1] == DEFENDER, \
        "Defender should NOT be captured (throne not hostile when flag=False)"
    print("✓ Defender was NOT captured (correct, throne_is_hostile=False)")
    
    # Test 2: throne_is_hostile=True -> throne IS hostile to defenders
    print("\n--- Test 2: throne_is_hostile=True ---")
    game = Brandubh(throne_is_hostile=True)
    
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY
    game.board[throne[0], throne[1] - 1] = DEFENDER
    game.board[throne[0], throne[1] - 3] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    
    print(f"Setup: Attacker - space - Defender - Throne, throne_is_hostile={game.throne_is_hostile}")
    
    # Move attacker to sandwich defender against throne
    move = (throne[0], throne[1] - 3, throne[0], throne[1] - 2)
    game.make_move(move)
    
    # Defender SHOULD be captured (throne is hostile to defenders when flag=True)
    assert game.board[throne[0], throne[1] - 1] == EMPTY, \
        "Defender SHOULD be captured (throne IS hostile when flag=True)"
    print("✓ Defender WAS captured (correct, throne_is_hostile=True)")
    print("✅ TEST PASSED: Throne hostility to defenders depends on flag\n")


def test_occupied_throne_not_hostile():
    """Test that OCCUPIED throne is never hostile (can't capture against a piece)."""
    print("\n" + "="*70)
    print("TEST: Occupied throne is NOT hostile")
    print("="*70)
    
    game = Brandubh(throne_is_hostile=True)
    throne = game.throne
    
    # Setup: Defender - Attacker - King on throne
    # Try to capture attacker against occupied throne (should fail)
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = KING  # King occupies throne
    game.board[throne[0], throne[1] - 1] = ATTACKER
    game.board[throne[0], throne[1] - 3] = DEFENDER
    game.current_player = DEFENDER_PLAYER
    
    print(f"Setup: Defender - space - Attacker - King on throne")
    
    # Move defender to try to capture attacker
    move = (throne[0], throne[1] - 3, throne[0], throne[1] - 2)
    game.make_move(move)
    
    # Attacker should NOT be captured (throne is occupied, so not hostile)
    assert game.board[throne[0], throne[1] - 1] == ATTACKER, \
        "Attacker should NOT be captured against occupied throne"
    print("✓ Attacker was NOT captured against occupied throne (correct)")
    print("✅ TEST PASSED: Occupied throne is not hostile\n")


def test_throne_hostility_king_capture():
    """Test that throne hostility works correctly for king captures."""
    print("\n" + "="*70)
    print("TEST: Throne hostility in king captures")
    print("="*70)
    
    # Test with 2-piece king capture mode and throne_is_hostile=True
    game = Brandubh(king_capture_pieces=2, throne_is_hostile=True)
    throne = game.throne
    
    # Setup: Attacker - King - Empty throne
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY
    game.board[throne[0], throne[1] - 1] = KING
    game.board[throne[0], throne[1] - 3] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    
    print(f"Setup: Attacker - space - King - Throne, throne_is_hostile={game.throne_is_hostile}")
    
    # Move attacker to sandwich king against throne
    move = (throne[0], throne[1] - 3, throne[0], throne[1] - 2)
    game.make_move(move)
    
    # King should be captured (throne is hostile to king when flag=True)
    assert game.game_over, "Game should be over (king captured)"
    assert game.winner == ATTACKER_PLAYER, "Attackers should win"
    print("✓ King was captured against empty throne (throne_is_hostile=True)")
    
    # Test with throne_is_hostile=False
    print("\n--- Testing with throne_is_hostile=False ---")
    game = Brandubh(king_capture_pieces=2, throne_is_hostile=False)
    
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY
    game.board[throne[0], throne[1] - 1] = KING
    game.board[throne[0], throne[1] - 3] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    print(f"Setup: Same position, but throne_is_hostile={game.throne_is_hostile}")
    
    # Move attacker
    move = (throne[0], throne[1] - 3, throne[0], throne[1] - 2)
    game.make_move(move)
    
    # King should NOT be captured (throne not hostile to king when flag=False)
    assert not game.game_over, "Game should NOT be over (throne not hostile to king)"
    assert game.board[throne[0], throne[1] - 1] == KING, "King should still be there"
    print("✓ King was NOT captured against empty throne (throne_is_hostile=False)")
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
        print(f"✓ {name}: No king_has_left_throne attribute")
        
        # Test that clone also doesn't have it
        cloned = game.clone()
        assert not hasattr(cloned, 'king_has_left_throne'), \
            f"Cloned {name} should not have king_has_left_throne attribute"
        print(f"✓ {name}: Cloned game also has no king_has_left_throne")
    
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
        print(f"✓ {name}._is_hostile_square(throne, ATTACKER) = {result}")
        
        result = game._is_hostile_square(throne[0], throne[1], DEFENDER)
        print(f"✓ {name}._is_hostile_square(throne, DEFENDER) = {result}")
        
        result = game._is_hostile_square(throne[0], throne[1], KING)
        print(f"✓ {name}._is_hostile_square(throne, KING) = {result}")
    
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
    game.board[3, 1] = DEFENDER
    game.current_player = DEFENDER_PLAYER
    
    # Defender moves to sandwich attacker at (3,3) between (3,2) and (3,4)
    # But we need another defender at (3,4)
    game.board[3, 5] = DEFENDER
    move = (3, 5, 3, 4)
    game.make_move(move)
    # Now we have D at (3,4) but need D at (3,2) too
    
    # Let me redo this more carefully
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[3, 3] = ATTACKER  # Target
    game.board[3, 2] = DEFENDER  # One side
    game.board[3, 5] = DEFENDER  # Will move to other side
    game.current_player = DEFENDER_PLAYER
    
    move = (3, 5, 3, 4)
    game.make_move(move)
    
    assert game.board[3, 3] == EMPTY, "Attacker should be captured between two defenders"
    print("✓ Attacker captured between two defenders")
    
    # Scenario 2: Defender captured by two attackers (no throne)
    print("\n--- Scenario 2: Regular defender capture (no throne) ---")
    game = Brandubh()
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[3, 3] = DEFENDER  # Target
    game.board[3, 2] = ATTACKER  # One side
    game.board[3, 5] = ATTACKER  # Will move to other side
    game.current_player = ATTACKER_PLAYER
    
    move = (3, 5, 3, 4)
    game.make_move(move)
    
    assert game.board[3, 3] == EMPTY, "Defender should be captured between two attackers"
    print("✓ Defender captured between two attackers")
    
    # Scenario 3: King protected by defenders
    print("\n--- Scenario 3: King NOT captured when protected ---")
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
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        game.board[throne] = KING
        game.current_player = DEFENDER_PLAYER
        
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
        assert game._is_hostile_square(throne[0], throne[1], ATTACKER), \
            f"{name}: Empty throne should be hostile to attackers (always)"
        print(f"  ✓ Empty throne hostile to attackers")
        
        # Test 3: Empty throne hostility to defenders depends on flag
        game = GameClass(throne_is_hostile=False)
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        assert not game._is_hostile_square(throne[0], throne[1], DEFENDER), \
            f"{name}: Empty throne should NOT be hostile to defenders when flag=False"
        
        game = GameClass(throne_is_hostile=True)
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        assert game._is_hostile_square(throne[0], throne[1], DEFENDER), \
            f"{name}: Empty throne SHOULD be hostile to defenders when flag=True"
        print(f"  ✓ Empty throne hostility to defenders depends on flag")
        
        # Test 4: Occupied throne not hostile
        game = GameClass()
        game.board[throne] = KING
        assert not game._is_hostile_square(throne[0], throne[1], ATTACKER), \
            f"{name}: Occupied throne should not be hostile"
        assert not game._is_hostile_square(throne[0], throne[1], DEFENDER), \
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
        test_king_can_reenter_throne_all_variants,
        
        # Throne hostility tests
        test_throne_hostility_to_attackers,
        test_throne_hostility_to_defenders,
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
            import traceback
            traceback.print_exc()
    
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
