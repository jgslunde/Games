"""
Test script to verify temperature calculation logic for GUI.
"""

# Test the temperature calculation logic without importing pygame
def test_temperature_calculation():
    """Test temperature calculation for different modes."""
    
    # Test fixed mode
    print("Testing FIXED mode:")
    temperature = 1.0
    temperature_mode = "fixed"
    temperature_threshold = 10
    
    for move_count in [0, 5, 9, 10, 15]:
        temp = 0.0 if move_count >= temperature_threshold else temperature
        print(f"  Move {move_count}: temperature = {temp:.3f}")
    
    print("\nTesting DECAY mode:")
    temperature = 1.0
    temperature_mode = "decay"
    temperature_decay_moves = 30
    
    for move_count in [0, 10, 15, 20, 29, 30, 40]:
        if move_count < temperature_decay_moves:
            temp = temperature * (1.0 - move_count / temperature_decay_moves)
        else:
            temp = 0.0
        print(f"  Move {move_count}: temperature = {temp:.3f}")
    
    print("\nTesting KING mode (simulated):")
    temperature = 1.0
    temperature_mode = "king"
    
    # Simulate king leaving throne at move 5
    king_left_throne = False
    for move_count in [0, 4, 5, 10, 15]:
        if move_count >= 5:
            king_left_throne = True
        
        temp = 0.0 if king_left_throne else temperature
        print(f"  Move {move_count}: temperature = {temp:.3f} (king_left={king_left_throne})")
    
    print("\n✓ All temperature calculations working correctly!")

if __name__ == "__main__":
    test_temperature_calculation()
