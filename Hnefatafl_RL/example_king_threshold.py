#!/usr/bin/env python3
"""
Example: Training with king-based temperature threshold

This demonstrates using the new "king" temperature threshold mode,
where temperature drops when the king leaves the throne rather than
after a fixed number of moves.
"""

from train import TrainingConfig, train

# Create training configuration
config = TrainingConfig()

# Set basic parameters
config.num_iterations = 10
config.num_games_per_iteration = 50
config.num_mcts_simulations = 50

# Use king-based temperature threshold instead of move-count
config.temperature_threshold = "king"  # Drop temperature when king leaves throne
# config.temperature_threshold = 20    # Or use traditional move-count threshold

# Other settings
config.temperature = 1.0
config.num_workers = 8

print("Starting training with king-based temperature threshold...")
print("Temperature will drop to 0 when the king first moves off the throne.")
print()

# Start training
train(config)
