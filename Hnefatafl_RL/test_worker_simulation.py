"""
Simulate the actual training worker process to verify the fix works.
This mimics what happens in _play_self_play_game_worker in train.py.
"""

import importlib
import torch

def simulate_worker_setup():
    """
    Simulate the setup that happens in _play_self_play_game_worker.
    This is what was failing before the fix.
    """
    print("=" * 70)
    print("Simulating Training Worker Setup for Hnefatafl")
    print("=" * 70)
    print()
    
    # These are the parameters passed to the worker
    network_module = 'network_hnefatafl'
    network_class_name = 'HnefataflNet'
    num_res_blocks = 4
    num_channels = 64
    
    print(f"Network module: {network_module}")
    print(f"Network class: {network_class_name}")
    print()
    
    # Import network module and class (this is what train.py does)
    print("Step 1: Importing network module...")
    network_mod = importlib.import_module(network_module)
    NetworkClass = getattr(network_mod, network_class_name)
    print(f"  ✓ Network class loaded: {NetworkClass}")
    print()
    
    # Import MoveEncoder from the same network module (THE FIX)
    print("Step 2: Detecting and loading MoveEncoder...")
    if 'hnefatafl' in network_module.lower():
        encoder_class_name = 'HnefataflMoveEncoder'
    elif 'tablut' in network_module.lower():
        encoder_class_name = 'TablutMoveEncoder'
    else:
        encoder_class_name = 'MoveEncoder'
    
    print(f"  Detection logic selected: {encoder_class_name}")
    MoveEncoderClass = getattr(network_mod, encoder_class_name)
    print(f"  ✓ Encoder class loaded: {MoveEncoderClass}")
    print()
    
    # Create network instance
    print("Step 3: Creating network instance...")
    network = NetworkClass(num_res_blocks=num_res_blocks, num_channels=num_channels)
    print(f"  ✓ Network created")
    print()
    
    # Create encoder instance
    print("Step 4: Creating encoder instance...")
    encoder = MoveEncoderClass()
    print(f"  ✓ Encoder created")
    print()
    
    # Test that encoder works
    print("Step 5: Testing encoder functionality...")
    # Encode a simple move
    test_move = (5, 5, 5, 6)  # King moves one square right
    encoded = encoder.encode_move(test_move)
    print(f"  Test move: {test_move}")
    print(f"  Encoded as: {encoded}")
    decoded = encoder.decode_move(encoded)
    print(f"  Decoded back to: {decoded}")
    assert decoded == test_move, "Round-trip encoding/decoding failed"
    print(f"  ✓ Encoder working correctly")
    print()
    
    # Test with network
    print("Step 6: Testing network forward pass...")
    from hnefatafl import Hnefatafl
    game = Hnefatafl()
    
    # Create input
    state = game.get_state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    with torch.no_grad():
        policy, value = network(state_tensor)
    
    print(f"  Input shape: {state_tensor.shape}")
    print(f"  Policy shape: {policy.shape}")
    print(f"  Value shape: {value.shape}")
    expected_policy_size = 4840  # 121 * 4 * 10 for 11x11 board
    print(f"  Expected policy size: {expected_policy_size}")
    assert policy.shape[1] == expected_policy_size, "Policy size mismatch"
    print(f"  ✓ Network forward pass successful")
    print()
    
    print("=" * 70)
    print("All worker setup steps completed successfully! ✓")
    print("Training should work correctly now.")
    print("=" * 70)

if __name__ == "__main__":
    simulate_worker_setup()
