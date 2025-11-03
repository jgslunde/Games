"""
Quick reference and usage examples for the Brandubh AlphaZero implementation.
"""

# =============================================================================
# BASIC USAGE
# =============================================================================

# 1. Play a random game
# ---------------------
from brandubh import Brandubh
from agent import RandomAgent, play_game

agent1 = RandomAgent()
agent2 = RandomAgent()
winner = play_game(agent1, agent2, display=True)

# 2. Use MCTS agent (no neural network)
# --------------------------------------
from agent import MCTSAgent

mcts_agent = MCTSAgent(num_simulations=100)
random_agent = RandomAgent()
winner = play_game(mcts_agent, random_agent, display=True)

# 3. Use neural network agent (requires PyTorch)
# -----------------------------------------------
from agent import BrandubhAgent

nn_agent = BrandubhAgent(num_simulations=100)
move = nn_agent.select_move(game, temperature=0.0)

# =============================================================================
# GAME INTERFACE
# =============================================================================

game = Brandubh()

# Get current state (for neural network)
state = game.get_state()  # Shape: (4, 7, 7) - 4 planes

# Get legal moves
moves = game.get_legal_moves()  # List of (from_r, from_c, to_r, to_c)

# Make a move
game.make_move((3, 3, 3, 5))  # Move king from (3,3) to (3,5)

# Check game status
if game.game_over:
    winner = "Attackers" if game.winner == 0 else "Defenders"
    print(f"Game over! Winner: {winner}")

# Clone game state (for tree search)
game_copy = game.clone()

# =============================================================================
# NEURAL NETWORK
# =============================================================================

import torch
from network import BrandubhNet, MoveEncoder

# Create network
net = BrandubhNet(num_res_blocks=4, num_channels=64)

# Forward pass
state = game.get_state()
state_tensor = torch.from_numpy(state).unsqueeze(0)  # Add batch dim
policy_logits, value = net(state_tensor)
# policy_logits: (1, 1176) - logits for all possible moves
# value: (1, 1) - win probability for current player

# Convert move to policy index
move = (3, 3, 3, 5)
policy_idx = MoveEncoder.encode_move(move)

# Convert policy index back to move
decoded_move = MoveEncoder.decode_move(policy_idx)

# Get legal move mask
legal_mask = MoveEncoder.get_legal_move_mask(game)  # Shape: (1176,)

# =============================================================================
# MCTS
# =============================================================================

from mcts import MCTS, RandomRolloutMCTS

# MCTS with neural network
net = BrandubhNet()
mcts = MCTS(net, num_simulations=100, c_puct=1.4)
move = mcts.select_move(game, temperature=0.0)

# MCTS with random rollouts (no NN)
mcts = RandomRolloutMCTS(num_simulations=100)
move = mcts.select_move(game, temperature=0.0)

# =============================================================================
# AGENT COMPARISON
# =============================================================================

from agent import evaluate_agents

agent1 = MCTSAgent(num_simulations=50)
agent2 = RandomAgent()
results = evaluate_agents(agent1, agent2, num_games=20)
# Results: {'agent1_wins': ..., 'agent2_wins': ..., ...}

# =============================================================================
# SAVING/LOADING WEIGHTS
# =============================================================================

# Save
agent = BrandubhAgent()
agent.save_weights("model.pth")

# Load
agent = BrandubhAgent()
agent.load_weights("model.pth")

# =============================================================================
# POLICY ENCODING SCHEME
# =============================================================================

# Policy vector size: 1,176 = 49 squares × 4 directions × 6 distances
# 
# For a move from square (r, c) to (r', c'):
# 1. from_square = r * 7 + c  (0-48)
# 2. Determine direction:
#    - Up (dr < 0): direction = 0
#    - Down (dr > 0): direction = 1
#    - Left (dc < 0): direction = 2
#    - Right (dc > 0): direction = 3
# 3. distance = abs(dr) or abs(dc)  (1-6)
# 4. policy_idx = from_square * 24 + direction * 6 + (distance - 1)

# =============================================================================
# BOARD REPRESENTATION
# =============================================================================

# Internal representation (numpy array):
# 0 = Empty
# 1 = Attacker
# 2 = Defender
# 3 = King

# Neural network input (4 planes):
# Plane 0: Binary mask of attacker pieces
# Plane 1: Binary mask of defender pieces
# Plane 2: Binary mask of king
# Plane 3: Current player (0=attackers, 1=defenders)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# Network architecture
NUM_RES_BLOCKS = 4      # Number of residual blocks
NUM_CHANNELS = 64       # Channels in conv layers

# MCTS
NUM_SIMULATIONS = 100   # Simulations per move
C_PUCT = 1.4           # Exploration constant

# Move selection
TEMPERATURE = 1.0       # During training (exploration)
TEMPERATURE = 0.0       # During play (deterministic)

# Training (to be implemented)
LEARNING_RATE = 0.001
BATCH_SIZE = 32
GAMES_PER_ITERATION = 100
TRAINING_STEPS = 1000
REPLAY_BUFFER_SIZE = 100000

# =============================================================================
# PERFORMANCE TIPS
# =============================================================================

# 1. Use fewer MCTS simulations during training (25-50)
# 2. Use more simulations for strong play (100-400)
# 3. Batch neural network evaluations if possible
# 4. Use GPU if available: device='cuda'
# 5. Enable torch.compile() for faster inference (PyTorch 2.0+)
# 6. Cache frequently-accessed positions

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# ImportError: No module named 'torch'
# → Install PyTorch: pip install torch

# MCTS is too slow
# → Reduce num_simulations
# → Use smaller network (fewer blocks/channels)
# → Use GPU if available

# Network trains but doesn't improve
# → Check learning rate (try 0.0001-0.01)
# → Ensure sufficient exploration (temperature > 0)
# → Check replay buffer diversity
# → Try more simulations during self-play

# Games end in draws/stalemates
# → This is normal for Tafl games
# → Consider adding draw penalty in training
# → Implement 3-fold repetition detection
