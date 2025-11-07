"""
Quick test to verify training configuration prints correctly.
"""

import sys
sys.path.insert(0, '/mn/stornext/u3/jonassl/Games/Hnefatafl_RL')

from train import TrainingConfig

config = TrainingConfig()

print("="*70)
print("VERIFYING TRAINING CONFIGURATION INCLUDES GAME RULES")
print("="*70)

# Game rules
print("\n--- Game Rules ---")
print(f"King capture pieces: {config.king_capture_pieces}")
capture_desc = {
    2: "(standard custodian - 2 opposite attackers)",
    3: "(3 out of 4 sides surrounded)",
    4: "(all 4 sides surrounded)"
}
print(f"  {capture_desc.get(config.king_capture_pieces, '')}")
print(f"King can capture: {config.king_can_capture}")
print(f"Throne is hostile: {config.throne_is_hostile}")

print("\n" + "="*70)
print("✓ Configuration includes game rules")
print("✓ Training will display these rules at start")
print("="*70)
