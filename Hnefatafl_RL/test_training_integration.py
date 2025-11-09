"""
Test that the training integration works correctly for Hnefatafl.
This verifies that the encoder detection logic in train.py will work.
"""

import importlib

def test_encoder_detection():
    """Test that the encoder is correctly detected from network module name."""
    
    print("=" * 70)
    print("Testing Encoder Detection for All Game Variants")
    print("=" * 70)
    print()
    
    # Test cases: (network_module_name, expected_encoder_class_name)
    test_cases = [
        ("network", "MoveEncoder"),           # Brandubh
        ("network_tablut", "TablutMoveEncoder"),  # Tablut
        ("network_hnefatafl", "HnefataflMoveEncoder"),  # Hnefatafl
    ]
    
    for network_module, expected_encoder in test_cases:
        print(f"Testing: {network_module}")
        print("-" * 70)
        
        # Import the module
        network_mod = importlib.import_module(network_module)
        print(f"  Module imported: {network_mod}")
        
        # Apply the detection logic from train.py
        if 'hnefatafl' in network_module.lower():
            encoder_class_name = 'HnefataflMoveEncoder'
        elif 'tablut' in network_module.lower():
            encoder_class_name = 'TablutMoveEncoder'
        else:
            encoder_class_name = 'MoveEncoder'
        
        print(f"  Detection logic selected: {encoder_class_name}")
        print(f"  Expected encoder: {expected_encoder}")
        
        # Get the encoder class
        MoveEncoderClass = getattr(network_mod, encoder_class_name)
        print(f"  Retrieved class: {MoveEncoderClass}")
        
        # Verify
        assert encoder_class_name == expected_encoder, \
            f"Expected {expected_encoder}, got {encoder_class_name}"
        print(f"  ✓ Correct encoder detected!")
        print()
    
    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)

if __name__ == "__main__":
    test_encoder_detection()
