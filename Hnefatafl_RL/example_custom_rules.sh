#!/bin/bash
# Example: train with custom game rules. Every setting of a run lives in its JSON config
# (see configs/ and run_config.py); single settings are changed with --set section.key=value.

# Example 1: the baseline rules of the config
# python run_training.py configs/brandubh/baseline.json

# Example 2: king requires 3 pieces to capture (3 out of 4 sides surrounded)
# python run_training.py configs/brandubh/baseline.json --set game.king_capture_pieces=3

# Example 3: king requires 4 pieces to capture (all 4 sides surrounded)
# python run_training.py configs/brandubh/baseline.json --set game.king_capture_pieces=4

# Example 4: king cannot participate in captures
# python run_training.py configs/brandubh/baseline.json --set game.king_can_capture=false

# Example 5: throne acts as hostile square for captures
# python run_training.py configs/brandubh/baseline.json --set game.throne_enabled=true --set game.throne_is_hostile=true

# Example 6: combine several changes (or copy the JSON file and edit it, which keeps a record)
python run_training.py configs/brandubh/baseline.json \
    --set game.king_capture_pieces=3 \
    --set game.king_can_capture=false \
    --set game.throne_enabled=true \
    --set game.throne_is_hostile=true \
    --set self_play.num_iterations=100 \
    --set self_play.num_games_per_iteration=256 \
    --checkpoint-dir checkpoints_custom_rules

echo "Training with custom rules:"
echo "  - King capture pieces: 3 (requires 3 out of 4 sides)"
echo "  - King can capture: False"
echo "  - Throne is hostile: True"
echo "The resolved settings are in checkpoints_custom_rules/run_config.json"
