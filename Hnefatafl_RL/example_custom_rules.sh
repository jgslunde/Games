#!/bin/bash
# Example: Train with custom game rules

# Example 1: Standard rules (default)
# python train_brandubh.py

# Example 2: King requires 3 pieces to capture (3 out of 4 sides surrounded)
# python train_brandubh.py --king-capture-pieces 3

# Example 3: King requires 4 pieces to capture (all 4 sides surrounded)
# python train_brandubh.py --king-capture-pieces 4

# Example 4: King cannot participate in captures
# python train_brandubh.py --king-cannot-capture

# Example 5: Throne acts as hostile square for captures
# python train_brandubh.py --throne-is-hostile

# Example 6: Combine multiple rule changes
python train_brandubh.py \
    --king-capture-pieces 3 \
    --king-cannot-capture \
    --throne-is-hostile \
    --iterations 100 \
    --games 256 \
    --checkpoint-dir checkpoints_custom_rules

echo "Training with custom rules:"
echo "  - King capture pieces: 3 (requires 3 out of 4 sides)"
echo "  - King can capture: False"
echo "  - Throne is hostile: True"
