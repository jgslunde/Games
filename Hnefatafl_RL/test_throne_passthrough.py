#!/usr/bin/env python3
"""
Test that pieces can move THROUGH the throne square (but not land on it).

This is a critical rule: non-king pieces cannot LAND on the throne,
but they CAN move THROUGH it to reach squares on the other side.

Run with: python test_throne_passthrough.py
"""

import numpy as np
from brandubh import Brandubh, EMPTY, ATTACKER, DEFENDER, KING, ATTACKER_PLAYER, DEFENDER_PLAYER
from tablut import Tablut
from hnefatafl import Hnefatafl


def test_pieces_can_move_through_empty_throne():
    """Test that all pieces can move through an empty throne."""
    print("\n" + "="*70)
    print("TEST: Pieces can move THROUGH empty throne")
    print("="*70)
    
    for GameClass, name, board_size in [(Brandubh, "Brandubh", 7),
                                         (Tablut, "Tablut", 9),
                                         (Hnefatafl, "Hnefatafl", 11)]:
        print(f"\n--- Testing {name} ({board_size}x{board_size}) ---")
        game = GameClass()
        throne = game.throne
        
        # Test 1: Defender moves horizontally through throne
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        game.board[throne[0], throne[1] - 2] = DEFENDER  # Two spaces left
        game.current_player = DEFENDER_PLAYER
        
        moves = game.get_legal_moves()
        
        # Should be able to reach squares on both sides of throne
        can_move_before = (throne[0], throne[1] - 2, throne[0], throne[1] - 1) in moves
        can_move_after = (throne[0], throne[1] - 2, throne[0], throne[1] + 1) in moves
        cannot_land_on = (throne[0], throne[1] - 2, throne[0], throne[1]) in moves
        
        assert can_move_before, f"{name}: Should reach square before throne"
        assert can_move_after, f"{name}: Should reach square AFTER throne (move through)"
        assert not cannot_land_on, f"{name}: Should NOT land ON throne"
        
        print(f"  ✓ Defender can move through throne horizontally")
        
        # Test 2: Attacker moves vertically through throne
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        game.board[throne[0] - 2, throne[1]] = ATTACKER  # Two spaces above
        game.current_player = ATTACKER_PLAYER
        
        moves = game.get_legal_moves()
        can_move_after = (throne[0] - 2, throne[1], throne[0] + 1, throne[1]) in moves
        
        assert can_move_after, f"{name}: Attacker should move through throne vertically"
        print(f"  ✓ Attacker can move through throne vertically")
    
    print("\n✅ PASS: All pieces can move through empty throne\n")


def test_cannot_move_through_occupied_throne():
    """Test that pieces CANNOT move through an occupied throne."""
    print("="*70)
    print("TEST: Cannot move through OCCUPIED throne")
    print("="*70)
    
    for GameClass, name, board_size in [(Brandubh, "Brandubh", 7),
                                         (Tablut, "Tablut", 9),
                                         (Hnefatafl, "Hnefatafl", 11)]:
        print(f"\n--- Testing {name} ({board_size}x{board_size}) ---")
        game = GameClass()
        throne = game.throne
        
        # King occupies throne
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        game.board[throne] = KING  # King on throne
        game.board[throne[0], throne[1] - 2] = DEFENDER
        game.current_player = DEFENDER_PLAYER
        
        moves = game.get_legal_moves()
        
        # Can move up to throne but not through it
        can_move_before = (throne[0], throne[1] - 2, throne[0], throne[1] - 1) in moves
        cannot_move_through = (throne[0], throne[1] - 2, throne[0], throne[1] + 1) in moves
        
        assert can_move_before, f"{name}: Should reach square before occupied throne"
        assert not cannot_move_through, f"{name}: Should NOT move through occupied throne"
        
        print(f"  ✓ Cannot move through occupied throne")
    
    print("\n✅ PASS: Pieces blocked by occupied throne\n")


def test_king_can_land_on_throne():
    """Test that king can still land on throne."""
    print("="*70)
    print("TEST: King can land on throne")
    print("="*70)
    
    for GameClass, name, board_size in [(Brandubh, "Brandubh", 7),
                                         (Tablut, "Tablut", 9),
                                         (Hnefatafl, "Hnefatafl", 11)]:
        print(f"\n--- Testing {name} ({board_size}x{board_size}) ---")
        game = GameClass()
        throne = game.throne
        
        game.board = np.zeros((board_size, board_size), dtype=np.int8)
        game.board[throne[0], throne[1] - 1] = KING
        game.current_player = DEFENDER_PLAYER
        
        moves = game.get_legal_moves()
        can_land = (throne[0], throne[1] - 1, throne[0], throne[1]) in moves
        
        assert can_land, f"{name}: King should be able to land on throne"
        print(f"  ✓ King can land on throne")
    
    print("\n✅ PASS: King can land on throne\n")


def test_long_distance_moves_through_throne():
    """Test that pieces can move multiple squares through throne."""
    print("="*70)
    print("TEST: Long-distance moves through throne")
    print("="*70)
    
    game = Brandubh()
    throne = game.throne
    
    # Place defender far from throne
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne[0], 0] = DEFENDER  # Far left edge
    game.current_player = DEFENDER_PLAYER
    
    moves = game.get_legal_moves()
    
    # Check all valid destination squares to the right
    destinations = [(m[2], m[3]) for m in moves if m[0] == throne[0] and m[1] == 0]
    destinations_right = [d for d in destinations if d[1] > 0]
    
    print(f"Defender at {(throne[0], 0)}")
    print(f"Throne at {throne}")
    print(f"Can reach rightward: {sorted(destinations_right)}")
    
    # Should be able to reach squares on both sides of throne
    # (1,2) before throne, skip (3,3) throne itself, then (3,4), (3,5), (3,6) after
    assert (throne[0], throne[1] - 1) in destinations_right, "Should reach square before throne"
    assert (throne[0], throne[1]) not in destinations_right, "Should NOT land on throne"
    assert (throne[0], throne[1] + 1) in destinations_right, "Should reach square after throne"
    assert (throne[0], 6) in destinations_right, "Should reach far right edge"
    
    print("✓ Can make long moves through throne")
    print("✅ PASS: Long-distance moves work correctly\n")


def test_multiple_pieces_movement():
    """Test that multiple pieces can independently move through throne."""
    print("="*70)
    print("TEST: Multiple pieces can move through throne")
    print("="*70)
    
    game = Brandubh()
    throne = game.throne
    
    # Place multiple defenders around throne
    game.board = np.zeros((7, 7), dtype=np.int8)
    game.board[throne[0], 0] = DEFENDER  # Left
    game.board[throne[0], 6] = DEFENDER  # Right
    game.board[0, throne[1]] = DEFENDER  # Top
    game.board[6, throne[1]] = DEFENDER  # Bottom
    game.current_player = DEFENDER_PLAYER
    
    moves = game.get_legal_moves()
    
    # All should be able to move through throne
    left_can_cross = (throne[0], 0, throne[0], throne[1] + 1) in moves
    right_can_cross = (throne[0], 6, throne[0], throne[1] - 1) in moves
    top_can_cross = (0, throne[1], throne[0] + 1, throne[1]) in moves
    bottom_can_cross = (6, throne[1], throne[0] - 1, throne[1]) in moves
    
    assert left_can_cross, "Left defender should cross throne"
    assert right_can_cross, "Right defender should cross throne"
    assert top_can_cross, "Top defender should cross throne"
    assert bottom_can_cross, "Bottom defender should cross throne"
    
    print("✓ All defenders can independently move through throne")
    print("✅ PASS: Multiple pieces movement works\n")


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# THRONE PASS-THROUGH TEST SUITE")
    print("#"*70)
    
    tests = [
        test_pieces_can_move_through_empty_throne,
        test_cannot_move_through_occupied_throne,
        test_king_can_land_on_throne,
        test_long_distance_moves_through_throne,
        test_multiple_pieces_movement,
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
    
    print("#"*70)
    print(f"# RESULTS: {passed}/{len(tests)} tests passed")
    print("#"*70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThrone pass-through rules are correctly implemented:")
        print("  1. ✅ Non-king pieces can move THROUGH empty throne")
        print("  2. ✅ Non-king pieces cannot LAND ON throne")
        print("  3. ✅ Occupied throne blocks movement")
        print("  4. ✅ King can land on throne")
        print("  5. ✅ Long-distance moves through throne work")
        print()
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
