"""
Demo script showing the complete Brandubh AlphaZero system.
Tests agents with and without PyTorch.
"""

from brandubh import Brandubh
from agent import RandomAgent, MCTSAgent, play_game

print("=" * 60)
print("Brandubh AlphaZero Demo")
print("=" * 60)

# Check PyTorch availability
try:
    import torch
    from agent import BrandubhAgent
    
    TORCH_AVAILABLE = True
    print(f"\n✓ PyTorch {torch.__version__} available")
    print("  Full neural network functionality enabled")
except ImportError:
    TORCH_AVAILABLE = False
    print("\n✗ PyTorch not installed")
    print("  Neural network features disabled")
    print("  Install with: pip install torch")

print("\n" + "=" * 60)
print("Testing Game Components")
print("=" * 60)

# Test game
print("\n1. Testing game rules...")
game = Brandubh()
print(f"   ✓ Initial position: {len(game.get_legal_moves())} legal moves")

# Test random agent
print("\n2. Testing random agent...")
random_agent = RandomAgent()
move = random_agent.select_move(game)
print(f"   ✓ Random agent selected move: {move}")

# Test MCTS agent
print("\n3. Testing MCTS agent (random rollouts)...")
mcts_agent = MCTSAgent(num_simulations=50)
print("   Running 50 simulations...")
move = mcts_agent.select_move(game)
print(f"   ✓ MCTS agent selected move: {move}")

# Test neural network agent if available
if TORCH_AVAILABLE:
    print("\n4. Testing neural network agent...")
    print("   Creating untrained network...")
    nn_agent = BrandubhAgent(num_simulations=25)
    print("   Running 25 simulations with neural network...")
    move = nn_agent.select_move(game)
    print(f"   ✓ NN agent selected move: {move}")
    
    # Show network info
    print("\n   Network architecture:")
    total_params = sum(p.numel() for p in nn_agent.network.parameters())
    print(f"   - Total parameters: {total_params:,}")
    print("   - Input: 4 planes (7x7)")
    print("   - Output: 1,176 move logits + 1 value")
else:
    print("\n4. Neural network agent: Skipped (PyTorch not installed)")

print("\n" + "=" * 60)
print("Playing Demo Game: MCTS vs Random")
print("=" * 60)

# Play a short game
game = Brandubh()
print("\nPlaying MCTS (50 sims) vs Random agent...\n")

attacker_agent = MCTSAgent(num_simulations=50)
defender_agent = RandomAgent()

winner = play_game(attacker_agent, defender_agent, display=True)

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)

print("\nNext steps:")
print("1. Install PyTorch: pip install -r requirements.txt")
print("2. Implement training loop (self-play + network training)")
print("3. Train the network and evaluate improvements")
print("4. Create a GUI for human vs AI play")

print("\nFiles created:")
print("  - brandubh.py:  Game rules and board representation")
print("  - network.py:   Neural network architecture")
print("  - mcts.py:      Monte Carlo Tree Search")
print("  - agent.py:     Agent wrappers")
print("  - play.py:      Random self-play demo")
print("  - demo.py:      This demo script")

print("\nFor more information, see README.md")
