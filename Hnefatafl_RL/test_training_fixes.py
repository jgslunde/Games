"""
Tests for the training-pipeline fixes:
1. Symmetry augmentation: every augmented policy target matches its augmented board.
2. StateHistory.from_game rebuilds the same history planes as tracking them move by move.
3. MCTS._evaluate (softmax over legal moves) equals the old masked softmax over all moves.
4. play_game creates the game with the given rules.

Run with: python test_training_fixes.py
"""

import numpy as np
import torch

from brandubh import Brandubh
from tablut import Tablut
from network import BrandubhNet, MoveEncoder, StateHistory
from network_tablut import TablutMoveEncoder
from mcts import MCTS
from agent import RandomAgent, play_game
from train import augment_sample


def random_position(game_class, rng, max_plies=14):
    """Play a random number of random moves (no rules changed) and return a non-finished game."""
    while True:
        game = game_class(throne_enabled=False)
        for _ in range(rng.integers(1, max_plies)):
            moves = game.get_legal_moves()
            game.make_move(moves[rng.integers(len(moves))])
            if game.game_over:
                break
        if not game.game_over:
            return game


def test_augmentation_consistent():
    """Each of the 8 symmetries must put policy mass only on moves that are legal-looking on its board."""
    rng = np.random.default_rng(0)
    for game_class, encoder, size in [(Brandubh, MoveEncoder, 7), (Tablut, TablutMoveEncoder, 9)]:
        for _ in range(20):
            game = random_position(game_class, rng)
            state, player, legal = game.get_state(), game.current_player, game.get_legal_moves()
            policy = np.zeros(size * size * 4 * (size - 1), dtype=np.float32)
            for move in legal:
                policy[encoder.encode_move(move)] = 1.0 / len(legal)
            augmented = augment_sample(state, policy, 0.5, board_size=size)
            assert len(augmented) == 8
            for aug_state, aug_policy, aug_value in augmented:
                assert aug_value == 0.5 and np.isclose(aug_policy.sum(), 1.0)
                for idx in np.nonzero(aug_policy)[0]:
                    fr, fc, tr, tc = encoder.decode_move(idx)
                    # Plane 0 = attackers; planes 1 + 2 = defenders and king
                    own_piece = aug_state[0, fr, fc] if player == 0 else aug_state[1, fr, fc] + aug_state[2, fr, fc]
                    assert own_piece > 0, f"{game_class.__name__}: move {idx} does not start on own piece"
                    assert aug_state[0:3, tr, tc].sum() == 0, f"{game_class.__name__}: move {idx} ends on a piece"
    print("  augmentation: all 8 symmetries consistent for Brandubh and Tablut")
    return True


def test_from_game_matches_tracked_history():
    """from_game must give the same planes as pushing piece planes after every move (as in self-play)."""
    rng = np.random.default_rng(1)
    for history_length in (1, 4):
        game = Brandubh(throne_enabled=False)
        tracked = StateHistory(history_length=history_length, board_size=7)
        tracked.push(game.get_piece_planes())
        for _ in range(30):
            rebuilt = StateHistory.from_game(game, history_length)
            expected = tracked.get_state_with_history(game.current_player)
            assert np.array_equal(rebuilt.get_state_with_history(game.current_player), expected)
            moves = game.get_legal_moves()
            game.make_move(moves[rng.integers(len(moves))])
            if game.game_over:
                break
            tracked.push(game.get_piece_planes())
    print("  from_game: rebuilt history equals tracked history")
    return True


def test_evaluate_matches_masked_softmax():
    """Softmax over legal moves must equal the old mask-all-moves softmax restricted to legal moves."""
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    network = BrandubhNet(num_res_blocks=2, num_channels=16).eval()
    mcts = MCTS(network, num_simulations=10, move_encoder_class=MoveEncoder)
    for _ in range(10):
        game = random_position(Brandubh, rng)
        legal = game.get_legal_moves()
        probs, value = mcts._evaluate(game, legal)
        with torch.inference_mode():
            logits, _ = network(torch.from_numpy(game.get_state()).unsqueeze(0))
        logits = logits.numpy()[0]
        mask = MoveEncoder.get_legal_move_mask(game)
        old = MCTS._softmax(logits * mask + (1 - mask) * (-1e8))[[MoveEncoder.encode_move(m) for m in legal]]
        assert np.allclose(probs, old, atol=1e-6) and np.isclose(probs.sum(), 1.0)
    print("  _evaluate: legal-move softmax equals masked softmax")
    return True


def test_play_game_uses_rules():
    """play_game must pass the rules to the game class (evaluation used to run with default rules)."""
    created = []

    class RecordingBrandubh(Brandubh):
        def __init__(self, **kwargs):
            created.append(kwargs)
            super().__init__(**kwargs)

    rules = dict(king_capture_pieces=2, king_can_capture=True, throne_is_hostile=False, throne_enabled=False)
    play_game(RandomAgent(), RandomAgent(), RecordingBrandubh, display=False, rules=rules, max_moves=20)
    assert created == [rules]
    print("  play_game: game created with the given rules")
    return True


if __name__ == "__main__":
    print("Testing training-pipeline fixes...")
    tests = [test_augmentation_consistent, test_from_game_matches_tracked_history,
             test_evaluate_matches_masked_softmax, test_play_game_uses_rules]
    all_passed = True
    for test in tests:
        try:
            all_passed &= test()
        except AssertionError as e:
            print(f"  FAILED {test.__name__}: {e}")
            all_passed = False
    print("All tests PASSED!" if all_passed else "Some tests FAILED!")
