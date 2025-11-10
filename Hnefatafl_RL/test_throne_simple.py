#!/usr/bin/env python3
"""
Simple, clear tests demonstrating throne rules are correctly implemented.
Run with: python test_throne_simple.py
"""

import numpy as np
from collections import deque
from brandubh import Brandubh, EMPTY, ATTACKER, DEFENDER, KING, ATTACKER_PLAYER, DEFENDER_PLAYER
from tablut import Tablut
from hnefatafl import Hnefatafl


def test_king_reentry():
    """Test 1: King can move back onto throne after leaving."""
    print("\n" + "="*60)
    print("TEST 1: King can re-enter throne")
    print("="*60)
    
    for GameClass, name in [(Brandubh, "Brandubh"), (Tablut, "Tablut"), (Hnefatafl, "Hnefatafl")]:
        game = GameClass()
        throne = game.throne
        board_size = game.board.shape[0]
        
        # Simple board: just the king
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        game.board[throne] = KING
        game.current_player = DEFENDER_PLAYER
        game.position_history = deque(maxlen=100)
        game._record_position()
        
        # Move king off throne
        move_off = (throne[0], throne[1], throne[0], throne[1] + 1)
        assert move_off in game.get_legal_moves(), f"{name}: King can't move off throne"
        
        # Check that king can move back
        game.make_move(move_off)
        game.current_player = ATTACKER_PLAYER
        game.current_player = DEFENDER_PLAYER
        
        move_back = (throne[0], throne[1] + 1, throne[0], throne[1])
        assert move_back in game.get_legal_moves(), f"{name}: King can't move back to throne!"
        
        print(f"✓ {name}: King can re-enter throne")
    
    print("✅ PASS: King can re-enter throne in all variants\n")


def test_throne_hostility_attackers():
    """Test 2: Empty throne is always hostile to attackers."""
    print("="*60)
    print("TEST 2: Throne hostile to attackers")
    print("="*60)
    
    game = Brandubh(throne_is_hostile=False)  # Even with flag=False!
    throne = game.throne
    
    # Setup: Defender will capture attacker against empty throne
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY
    game.board[throne[0], throne[1] - 1] = ATTACKER  # Next to throne
    game.board[throne[0], throne[1] - 3] = DEFENDER  # Two spaces away
    game.current_player = DEFENDER_PLAYER
    
    # Capture attacker
    move = (throne[0], throne[1] - 3, throne[0], throne[1] - 2)
    game.make_move(move)
    
    assert game.board[throne[0], throne[1] - 1] == EMPTY, "Attacker should be captured"
    print(f"✓ Attacker captured against empty throne (throne_is_hostile={game.throne_is_hostile})")
    print("✅ PASS: Empty throne always hostile to attackers\n")


def test_throne_hostility_defenders_flag():
    """Test 3: Empty throne only hostile to defenders when flag=True."""
    print("="*60)
    print("TEST 3: Throne hostility to defenders (flag-dependent)")
    print("="*60)
    
    # Test with flag=False
    game = Brandubh(throne_is_hostile=False)
    throne = game.throne
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY
    
    assert not game._is_hostile_square(throne[0], throne[1], DEFENDER), \
        "Throne should NOT be hostile to defenders when flag=False"
    print(f"✓ Throne NOT hostile to defenders (throne_is_hostile=False)")
    
    # Test with flag=True
    game = Brandubh(throne_is_hostile=True)
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = EMPTY
    
    assert game._is_hostile_square(throne[0], throne[1], DEFENDER), \
        "Throne SHOULD be hostile to defenders when flag=True"
    print(f"✓ Throne IS hostile to defenders (throne_is_hostile=True)")
    print("✅ PASS: Throne hostility to defenders depends on flag\n")


def test_occupied_throne():
    """Test 4: Occupied throne is never hostile."""
    print("="*60)
    print("TEST 4: Occupied throne not hostile")
    print("="*60)
    
    game = Brandubh(throne_is_hostile=True)
    throne = game.throne
    
    # Put king on throne
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne] = KING
    
    # Check hostility
    assert not game._is_hostile_square(throne[0], throne[1], ATTACKER), \
        "Occupied throne should not be hostile to attackers"
    assert not game._is_hostile_square(throne[0], throne[1], DEFENDER), \
        "Occupied throne should not be hostile to defenders"
    assert not game._is_hostile_square(throne[0], throne[1], KING), \
        "Occupied throne should not be hostile to king"
    
    print("✓ Occupied throne not hostile to any piece")
    print("✅ PASS: Occupied throne never hostile\n")


def test_no_legacy_attribute():
    """Test 5: king_has_left_throne attribute removed."""
    print("="*60)
    print("TEST 5: No legacy king_has_left_throne attribute")
    print("="*60)
    
    for GameClass, name in [(Brandubh, "Brandubh"), (Tablut, "Tablut"), (Hnefatafl, "Hnefatafl")]:
        game = GameClass()
        assert not hasattr(game, 'king_has_left_throne'), \
            f"{name} still has king_has_left_throne attribute!"
        print(f"✓ {name}: No king_has_left_throne attribute")
    
    print("✅ PASS: Legacy attribute removed from all variants\n")


def test_function_signature():
    """Test 6: _is_hostile_square requires target_piece parameter."""
    print("="*60)
    print("TEST 6: _is_hostile_square signature updated")
    print("="*60)
    
    game = Brandubh()
    throne = game.throne
    
    # These should all work without error
    game._is_hostile_square(throne[0], throne[1], ATTACKER)
    game._is_hostile_square(throne[0], throne[1], DEFENDER)
    game._is_hostile_square(throne[0], throne[1], KING)
    
    print("✓ _is_hostile_square accepts target_piece parameter")
    print("✅ PASS: Function signature correctly updated\n")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# THRONE RULES - COMPREHENSIVE TEST SUITE")
    print("#"*60)
    
    tests = [
        test_king_reentry,
        test_throne_hostility_attackers,
        test_throne_hostility_defenders_flag,
        test_occupied_throne,
        test_no_legacy_attribute,
        test_function_signature,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n❌ FAILED: {test_func.__name__}")
            print(f"   Error: {e}\n")
        except Exception as e:
            failed += 1
            print(f"\n❌ ERROR: {test_func.__name__}")
            print(f"   Error: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("#"*60)
    print(f"# RESULTS: {passed}/{len(tests)} tests passed")
    print("#"*60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe throne rules are correctly implemented:")
        print("  1. ✅ King can re-enter throne")
        print("  2. ✅ Empty throne always hostile to attackers")
        print("  3. ✅ Empty throne hostile to defenders only if flag=True")
        print("  4. ✅ Occupied throne never hostile")
        print("  5. ✅ Legacy tracking removed")
        print("  6. ✅ Function signatures updated")
        print()
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
