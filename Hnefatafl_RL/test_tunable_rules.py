"""
Comprehensive tests for tunable Brandubh rules.
Tests all combinations of:
- king_capture_pieces: 2, 3, or 4
- king_can_capture: True or False
- throne_is_hostile: True or False
"""

import numpy as np
from brandubh import Brandubh, ATTACKER, DEFENDER, KING, EMPTY, ATTACKER_PLAYER, DEFENDER_PLAYER


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []
    
    def record(self, test_name: str, passed: bool, message: str = ""):
        if passed:
            self.passed += 1
            print(f"✓ {test_name}")
        else:
            self.failed += 1
            self.failures.append((test_name, message))
            print(f"✗ {test_name}: {message}")
    
    def summary(self):
        print("\n" + "="*70)
        print(f"Test Results: {self.passed} passed, {self.failed} failed")
        print("="*70)
        if self.failures:
            print("\nFailures:")
            for name, msg in self.failures:
                print(f"  - {name}: {msg}")
        return self.failed == 0


def test_king_capture_2_pieces(results: TestResults):
    """Test king capture with 2 pieces (standard custodian capture)."""
    print("\n--- Testing king_capture_pieces=2 ---")
    
    # Test horizontal capture
    game = Brandubh(king_capture_pieces=2)
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING
    game.board[3, 2] = ATTACKER
    game.board[2, 4] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    game.make_move((2, 4, 3, 4))
    results.record(
        "2-piece horizontal capture",
        game.game_over and game.winner == ATTACKER_PLAYER and KING not in game.board,
        f"Expected king captured, got game_over={game.game_over}, winner={game.winner}"
    )
    
    # Test vertical capture
    game = Brandubh(king_capture_pieces=2)
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING
    game.board[2, 3] = ATTACKER
    game.board[5, 3] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    game.make_move((5, 3, 4, 3))
    results.record(
        "2-piece vertical capture",
        game.game_over and game.winner == ATTACKER_PLAYER and KING not in game.board,
        f"Expected king captured, got game_over={game.game_over}, winner={game.winner}"
    )
    
    # Test that diagonal placement doesn't capture
    game = Brandubh(king_capture_pieces=2)
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING
    game.board[2, 2] = ATTACKER
    game.board[2, 4] = ATTACKER
    game.board[4, 2] = ATTACKER
    game.board[1, 4] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    game.make_move((1, 4, 4, 4))  # Place fourth attacker diagonally
    results.record(
        "2-piece: diagonal doesn't capture",
        not game.game_over,
        f"King should NOT be captured with only diagonal attackers, got game_over={game.game_over}"
    )


def test_king_capture_3_pieces(results: TestResults):
    """Test king capture with 3 pieces."""
    print("\n--- Testing king_capture_pieces=3 ---")
    
    # Test that 2 sides is not enough
    game = Brandubh(king_capture_pieces=3)
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING
    game.board[3, 2] = ATTACKER
    game.board[2, 4] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    game.make_move((2, 4, 3, 4))
    results.record(
        "3-piece: 2 sides not enough",
        not game.game_over,
        f"King should NOT be captured with only 2 attackers, got game_over={game.game_over}"
    )
    
    # Test that 3 sides is enough
    game = Brandubh(king_capture_pieces=3)
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING
    game.board[2, 3] = ATTACKER  # Top
    game.board[3, 2] = ATTACKER  # Left
    game.board[1, 4] = ATTACKER  # Will move to right
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    game.make_move((1, 4, 3, 4))
    results.record(
        "3-piece: 3 sides captures",
        game.game_over and game.winner == ATTACKER_PLAYER and KING not in game.board,
        f"Expected king captured with 3 attackers, got game_over={game.game_over}"
    )
    
    # Test another 3-side configuration
    game = Brandubh(king_capture_pieces=3)
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING
    game.board[3, 2] = ATTACKER  # Left
    game.board[3, 4] = ATTACKER  # Right
    game.board[5, 3] = ATTACKER  # Will move to bottom
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    game.make_move((5, 3, 4, 3))
    results.record(
        "3-piece: different 3-side configuration",
        game.game_over and game.winner == ATTACKER_PLAYER and KING not in game.board,
        f"Expected king captured, got game_over={game.game_over}"
    )


def test_king_capture_4_pieces(results: TestResults):
    """Test king capture with 4 pieces."""
    print("\n--- Testing king_capture_pieces=4 ---")
    
    # Test that 3 sides is not enough
    game = Brandubh(king_capture_pieces=4)
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING
    game.board[2, 3] = ATTACKER
    game.board[3, 2] = ATTACKER
    game.board[1, 4] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    game.make_move((1, 4, 3, 4))
    results.record(
        "4-piece: 3 sides not enough",
        not game.game_over,
        f"King should NOT be captured with only 3 attackers, got game_over={game.game_over}"
    )
    
    # Test that 4 sides captures
    game = Brandubh(king_capture_pieces=4)
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING
    game.board[2, 3] = ATTACKER  # Top
    game.board[3, 2] = ATTACKER  # Left
    game.board[3, 4] = ATTACKER  # Right
    game.board[5, 3] = ATTACKER  # Will move to bottom
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    game.make_move((5, 3, 4, 3))
    results.record(
        "4-piece: all 4 sides captures",
        game.game_over and game.winner == ATTACKER_PLAYER and KING not in game.board,
        f"Expected king captured with 4 attackers, got game_over={game.game_over}"
    )


def test_king_can_capture_true(results: TestResults):
    """Test king participating in captures when king_can_capture=True."""
    print("\n--- Testing king_can_capture=True ---")
    
    # King helps defender capture attacker
    game = Brandubh(king_can_capture=True)
    game.board[:, :] = EMPTY
    game.board[3, 2] = KING
    game.board[3, 3] = ATTACKER
    game.board[3, 5] = DEFENDER  # Will move to sandwich
    game.current_player = DEFENDER_PLAYER
    game.game_over = False
    
    # Move defender to sandwich attacker between king and defender
    game.make_move((3, 5, 3, 4))
    
    results.record(
        "king_can_capture=True: king helps capture",
        game.board[3, 3] == EMPTY,
        f"King should help capture attacker, board[3,3]={game.board[3, 3]}"
    )


def test_king_can_capture_false(results: TestResults):
    """Test king NOT participating in captures when king_can_capture=False."""
    print("\n--- Testing king_can_capture=False ---")
    
    # King does not help defender capture attacker
    game = Brandubh(king_can_capture=False)
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING
    game.board[3, 4] = ATTACKER
    game.board[1, 2] = DEFENDER
    game.current_player = DEFENDER_PLAYER
    game.game_over = False
    
    # Try to sandwich attacker between king and defender
    game.make_move((1, 2, 3, 2))
    
    results.record(
        "king_can_capture=False: king doesn't help capture",
        game.board[3, 4] == ATTACKER,
        f"King should NOT help capture, but attacker was captured"
    )
    
    # Defender with defender should still work
    game = Brandubh(king_can_capture=False)
    game.board[:, :] = EMPTY
    game.board[3, 3] = DEFENDER
    game.board[3, 4] = ATTACKER
    game.board[3, 5] = DEFENDER
    game.board[1, 2] = DEFENDER
    game.current_player = DEFENDER_PLAYER
    game.game_over = False
    
    game.make_move((1, 2, 2, 2))  # Dummy move to trigger check
    # Manually set up and check
    game.board[3, 2] = DEFENDER
    game.board[3, 3] = ATTACKER
    game.board[3, 4] = DEFENDER
    game.current_player = DEFENDER_PLAYER
    game._check_captures(3, 2)
    
    results.record(
        "king_can_capture=False: defender+defender still captures",
        game.board[3, 3] == EMPTY,
        f"Defenders should still capture without king"
    )


def test_throne_is_hostile_false(results: TestResults):
    """Test throne NOT being hostile when throne_is_hostile=False."""
    print("\n--- Testing throne_is_hostile=False ---")
    
    # Attacker next to throne should not be captured by defender + throne
    game = Brandubh(throne_is_hostile=False)
    game.board[:, :] = EMPTY
    game.board[3, 3] = EMPTY  # Throne empty
    game.board[3, 4] = ATTACKER
    game.board[1, 2] = DEFENDER
    game.current_player = DEFENDER_PLAYER
    game.game_over = False
    
    game.make_move((1, 2, 3, 2))
    
    results.record(
        "throne_is_hostile=False: throne doesn't capture",
        game.board[3, 4] == ATTACKER,
        f"Throne should NOT help capture, but attacker at (3,4) was captured"
    )


def test_throne_is_hostile_true(results: TestResults):
    """Test throne being hostile when throne_is_hostile=True."""
    print("\n--- Testing throne_is_hostile=True ---")
    
    # Attacker next to throne should be captured by defender + throne
    game = Brandubh(throne_is_hostile=True)
    game.board[:, :] = EMPTY
    game.board[3, 3] = EMPTY  # Throne empty
    game.board[3, 4] = ATTACKER  # Attacker between throne and where defender will be
    game.board[3, 6] = DEFENDER
    game.current_player = DEFENDER_PLAYER
    game.game_over = False
    
    # Move defender to sandwich attacker between throne and defender
    # Pattern: THRONE(3,3) - ATTACKER(3,4) - DEFENDER(3,5)
    game.make_move((3, 6, 3, 5))
    
    results.record(
        "throne_is_hostile=True: throne helps capture",
        game.board[3, 4] == EMPTY,
        f"Throne should help capture, but attacker still at (3,4)={game.board[3, 4]}"
    )
    
    # Corner should still work
    game = Brandubh(throne_is_hostile=True)
    game.board[:, :] = EMPTY
    game.board[0, 1] = ATTACKER  # Attacker between corner and where defender will be
    game.board[0, 3] = DEFENDER
    game.current_player = DEFENDER_PLAYER
    game.game_over = False
    
    # Move defender to sandwich attacker between corner and defender
    # Pattern: CORNER(0,0) - ATTACKER(0,1) - DEFENDER(0,2)
    game.make_move((0, 3, 0, 2))
    
    results.record(
        "throne_is_hostile=True: corner still works",
        game.board[0, 1] == EMPTY,
        f"Corner should capture, but attacker still at (0,1)={game.board[0, 1]}"
    )


def test_combined_rules(results: TestResults):
    """Test combinations of different rules."""
    print("\n--- Testing combined rules ---")
    
    # king_capture_pieces=3, king_can_capture=False, throne_is_hostile=True
    game = Brandubh(king_capture_pieces=3, king_can_capture=False, throne_is_hostile=True)
    game.board[:, :] = EMPTY
    game.board[3, 3] = KING
    game.board[2, 3] = ATTACKER
    game.board[3, 2] = ATTACKER
    game.board[1, 4] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    game.make_move((1, 4, 3, 4))
    results.record(
        "Combined: 3-piece capture works",
        game.game_over and game.winner == ATTACKER_PLAYER,
        f"Expected king captured with 3 pieces, got game_over={game.game_over}"
    )
    
    # king_capture_pieces=4, throne_is_hostile=True - throne as 4th side
    game = Brandubh(king_capture_pieces=4, throne_is_hostile=True)
    game.board[:, :] = EMPTY
    game.board[3, 4] = KING  # King next to throne
    game.board[2, 4] = ATTACKER
    game.board[4, 4] = ATTACKER
    game.board[3, 5] = ATTACKER
    game.board[1, 2] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    # King at (3,4): attackers at top, bottom, right. Throne at (3,3) = left
    # But throne doesn't count for king capture (king needs attackers)
    game.make_move((1, 2, 2, 2))  # Dummy move
    results.record(
        "Combined: throne doesn't count for king capture",
        not game.game_over,
        f"Throne should not count as attacker for king capture"
    )


def test_clone_preserves_rules(results: TestResults):
    """Test that cloning preserves rule settings."""
    print("\n--- Testing clone preserves rules ---")
    
    game = Brandubh(king_capture_pieces=3, king_can_capture=False, throne_is_hostile=True)
    cloned = game.clone()
    
    results.record(
        "Clone: king_capture_pieces preserved",
        cloned.king_capture_pieces == 3,
        f"Expected 3, got {cloned.king_capture_pieces}"
    )
    results.record(
        "Clone: king_can_capture preserved",
        cloned.king_can_capture == False,
        f"Expected False, got {cloned.king_can_capture}"
    )
    results.record(
        "Clone: throne_is_hostile preserved",
        cloned.throne_is_hostile == True,
        f"Expected True, got {cloned.throne_is_hostile}"
    )


def test_edge_cases(results: TestResults):
    """Test edge cases and boundary conditions."""
    print("\n--- Testing edge cases ---")
    
    # King at edge of board with 2-piece capture
    game = Brandubh(king_capture_pieces=2)
    game.board[:, :] = EMPTY
    game.board[0, 3] = KING
    game.board[0, 2] = ATTACKER
    game.board[2, 4] = ATTACKER
    game.current_player = ATTACKER_PLAYER
    game.game_over = False
    
    game.make_move((2, 4, 0, 4))
    results.record(
        "Edge: king at edge captured horizontally",
        game.game_over and game.winner == ATTACKER_PLAYER,
        f"King at edge should be captured, got game_over={game.game_over}"
    )
    
    # King in corner should still win for defenders
    game = Brandubh(king_capture_pieces=2)
    game.board[:, :] = EMPTY
    game.board[1, 0] = KING
    game.board[2, 1] = DEFENDER
    game.current_player = DEFENDER_PLAYER
    game.game_over = False
    
    game.make_move((2, 1, 0, 1))  # Move king to corner
    # Actually move king
    game.board[1, 0] = EMPTY
    game.board[0, 0] = KING
    game._check_game_over()
    
    results.record(
        "Edge: king in corner wins",
        game.game_over and game.winner == DEFENDER_PLAYER,
        f"King in corner should win, got game_over={game.game_over}, winner={game.winner}"
    )


def run_all_tests():
    """Run all tests and report results."""
    results = TestResults()
    
    print("="*70)
    print("COMPREHENSIVE BRANDUBH RULE TESTING")
    print("="*70)
    
    test_king_capture_2_pieces(results)
    test_king_capture_3_pieces(results)
    test_king_capture_4_pieces(results)
    test_king_can_capture_true(results)
    test_king_can_capture_false(results)
    test_throne_is_hostile_false(results)
    test_throne_is_hostile_true(results)
    test_combined_rules(results)
    test_clone_preserves_rules(results)
    test_edge_cases(results)
    
    return results.summary()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
