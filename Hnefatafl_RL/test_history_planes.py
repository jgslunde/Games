"""
Test script for history planes implementation.
Verifies that:
1. BrandubhNet works with different history_length values
2. StateHistory correctly maintains state history
3. MCTS integrates properly with history tracking
4. Backward compatibility with history_length=1 (default)
"""

import numpy as np
import torch
from brandubh import Brandubh
from network import BrandubhNet, StateHistory, MoveEncoder
from mcts import MCTS


def test_state_history():
    """Test StateHistory class functionality."""
    print("Testing StateHistory...")
    
    # Create a game and history tracker
    game = Brandubh()
    history = StateHistory(history_length=3, board_size=7)
    
    # Push initial state
    history.push(game.get_piece_planes())
    assert len(history.history) == 1, f"Expected 1 history entry, got {len(history.history)}"
    
    # Get state with history (should have 3*3 + 1 = 10 planes)
    state = history.get_state_with_history(game.current_player)
    assert state.shape == (10, 7, 7), f"Expected shape (10, 7, 7), got {state.shape}"
    
    # First 3 planes should have the current piece positions
    assert np.sum(state[0]) > 0, "First plane (attackers) should have pieces"
    assert np.sum(state[1]) > 0, "Second plane (defenders) should have pieces"
    assert np.sum(state[2]) > 0, "Third plane (king) should have the king"
    
    # Planes 3-5 and 6-8 should be zeros (no history yet)
    assert np.sum(state[3:9]) == 0, "History planes should be zero for missing history"
    
    # Last plane should be current player
    assert np.all(state[-1] == game.current_player), "Last plane should be current player"
    
    # Make a move and update history
    legal_moves = game.get_legal_moves()
    game.make_move(legal_moves[0])
    history.push(game.get_piece_planes())
    
    assert len(history.history) == 2, f"Expected 2 history entries, got {len(history.history)}"
    
    # Now planes 3-5 should have the previous state
    state = history.get_state_with_history(game.current_player)
    # Previous state planes should now be non-zero (they were the initial position)
    assert np.sum(state[3:6]) > 0, "Previous state planes should have pieces now"
    
    print("  StateHistory tests passed!")
    return True


def test_network_history_length():
    """Test BrandubhNet with different history lengths."""
    print("Testing BrandubhNet with different history lengths...")
    
    # Test history_length=1 (backward compatible)
    net1 = BrandubhNet(num_res_blocks=2, num_channels=32, history_length=1)
    assert net1.input_channels == 4, f"Expected 4 input channels, got {net1.input_channels}"
    
    # Create input with 4 channels
    x1 = torch.randn(1, 4, 7, 7)
    policy1, value1 = net1(x1)
    assert policy1.shape == (1, 1176), f"Expected policy shape (1, 1176), got {policy1.shape}"
    assert value1.shape == (1, 1), f"Expected value shape (1, 1), got {value1.shape}"
    
    print("  history_length=1: OK")
    
    # Test history_length=4 (4 timesteps)
    net4 = BrandubhNet(num_res_blocks=2, num_channels=32, history_length=4)
    assert net4.input_channels == 13, f"Expected 13 input channels (4*3 + 1), got {net4.input_channels}"
    
    # Create input with 13 channels
    x4 = torch.randn(1, 13, 7, 7)
    policy4, value4 = net4(x4)
    assert policy4.shape == (1, 1176), f"Expected policy shape (1, 1176), got {policy4.shape}"
    assert value4.shape == (1, 1), f"Expected value shape (1, 1), got {value4.shape}"
    
    print("  history_length=4: OK")
    
    print("  BrandubhNet tests passed!")
    return True


def test_load_state_dict_compatible():
    """Test loading old checkpoint into network with more history planes."""
    print("Testing checkpoint backward compatibility...")
    
    # Create and save a network with history_length=1
    old_net = BrandubhNet(num_res_blocks=2, num_channels=32, history_length=1)
    old_state = old_net.state_dict()
    
    # Save original input weights for comparison
    old_input_weight = old_state['conv_input.weight'].clone()
    
    # Create a network with history_length=4
    new_net = BrandubhNet(num_res_blocks=2, num_channels=32, history_length=4)
    
    # Load old state into new network using compatible method
    new_net.load_state_dict_compatible(old_state)
    
    # Verify the network works
    x = torch.randn(1, 13, 7, 7)
    policy, value = new_net(x)
    assert policy.shape == (1, 1176), f"Expected policy shape (1, 1176), got {policy.shape}"
    
    # Verify input conv weights were properly adapted
    new_input_weight = new_net.state_dict()['conv_input.weight']
    
    # Check piece planes (channels 0-2) - should match original piece planes
    piece_planes_match = torch.allclose(old_input_weight[:, 0:3, :, :], new_input_weight[:, 0:3, :, :])
    assert piece_planes_match, "Piece plane weights should be copied to first 3 channels"
    
    # Check player plane - old channel 3 should now be at new channel -1 (12)
    player_plane_match = torch.allclose(old_input_weight[:, 3, :, :], new_input_weight[:, -1, :, :])
    assert player_plane_match, f"Player plane weight should be copied to last channel (shape comparison: {old_input_weight[:, 3, :, :].shape} vs {new_input_weight[:, -1, :, :].shape})"
    
    # Check that middle history planes are zero (channels 3 to 11)
    history_planes_zero = torch.all(new_input_weight[:, 3:-1, :, :] == 0)
    assert history_planes_zero, "History planes (channels 3-11) should be zero"
    
    print("  Checkpoint compatibility tests passed!")
    return True


def test_mcts_with_history():
    """Test MCTS search with history planes."""
    print("Testing MCTS with history planes...")
    
    # Create game and network
    game = Brandubh()
    net = BrandubhNet(num_res_blocks=2, num_channels=32, history_length=3)
    net.eval()
    
    # Create MCTS with history
    mcts = MCTS(net, num_simulations=10, history_length=3)
    
    # Create initial history
    history = StateHistory(history_length=3, board_size=7)
    history.push(game.get_piece_planes())
    
    # Run search with history
    visit_probs = mcts.search(game, state_history=history)
    
    # Should get valid move probabilities
    assert len(visit_probs) > 0, "Should get at least one move"
    assert abs(sum(visit_probs.values()) - 1.0) < 1e-5, "Visit probs should sum to 1"
    
    print("  MCTS with history tests passed!")
    return True


def test_mcts_backward_compatible():
    """Test MCTS without history (backward compatible)."""
    print("Testing MCTS backward compatibility (no history)...")
    
    # Create game and network with history_length=1
    game = Brandubh()
    net = BrandubhNet(num_res_blocks=2, num_channels=32, history_length=1)
    net.eval()
    
    # Create MCTS with default history_length=1
    mcts = MCTS(net, num_simulations=10, history_length=1)
    
    # Run search without passing state_history
    visit_probs = mcts.search(game)
    
    # Should get valid move probabilities
    assert len(visit_probs) > 0, "Should get at least one move"
    assert abs(sum(visit_probs.values()) - 1.0) < 1e-5, "Visit probs should sum to 1"
    
    print("  MCTS backward compatibility tests passed!")
    return True


def test_full_game_with_history():
    """Test playing a few moves with history tracking."""
    print("Testing full game flow with history...")
    
    game = Brandubh()
    net = BrandubhNet(num_res_blocks=2, num_channels=32, history_length=4)
    net.eval()
    
    mcts = MCTS(net, num_simulations=10, history_length=4)
    history = StateHistory(history_length=4, board_size=7)
    history.push(game.get_piece_planes())
    
    # Play 5 moves
    for i in range(5):
        if game.game_over:
            break
        
        # Run MCTS with history
        visit_probs = mcts.search(game, state_history=history)
        
        # Select best move
        move = max(visit_probs, key=visit_probs.get)
        game.make_move(move)
        
        # Update history
        history.push(game.get_piece_planes())
        
        print(f"    Move {i+1}: History length = {len(history.history)}")
    
    # History should have min(5+1, 4) = 4 entries (capped by history_length)
    assert len(history.history) <= 4, f"History should be capped at 4, got {len(history.history)}"
    
    print("  Full game flow tests passed!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("History Planes Implementation Tests")
    print("="*60)
    print()
    
    all_passed = True
    
    try:
        all_passed &= test_state_history()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        all_passed &= test_network_history_length()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        all_passed &= test_load_state_dict_compatible()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        all_passed &= test_mcts_with_history()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        all_passed &= test_mcts_backward_compatible()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        all_passed &= test_full_game_with_history()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    print("="*60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("="*60)
