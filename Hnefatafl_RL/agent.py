"""
Agent wrapper for playing Brandubh using MCTS with neural network.
"""

import torch
import numpy as np
from typing import Tuple

from brandubh import Brandubh
from network import BrandubhNet
from mcts import MCTS, RandomRolloutMCTS


class BrandubhAgent:
    """
    Agent that uses MCTS with neural network to play Brandubh.
    """
    
    def __init__(self, network: BrandubhNet = None, num_simulations: int = 100, 
                 c_puct: float = 1.4, device: str = 'cpu'):
        """
        Initialize agent.
        
        Args:
            network: trained BrandubhNet (if None, creates untrained network)
            num_simulations: number of MCTS simulations per move
            c_puct: exploration constant
            device: 'cpu' or 'cuda'
        """
        if network is None:
            network = BrandubhNet()
        
        self.network = network
        self.mcts = MCTS(network, num_simulations, c_puct, device)
        self.device = device
    
    def select_move(self, game: Brandubh, temperature: float = 0.0) -> Tuple[int, int, int, int]:
        """
        Select a move for the current game state.
        
        Args:
            game: current game state
            temperature: sampling temperature (0 = deterministic)
        
        Returns:
            move: (from_row, from_col, to_row, to_col)
        """
        return self.mcts.select_move(game, temperature)
    
    def load_weights(self, path: str):
        """Load network weights from file."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()
    
    def save_weights(self, path: str):
        """Save network weights to file."""
        torch.save(self.network.state_dict(), path)


class RandomAgent:
    """Simple random agent for baseline comparison."""
    
    def select_move(self, game: Brandubh, temperature: float = 0.0) -> Tuple[int, int, int, int]:
        """Select a random legal move."""
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        idx = np.random.randint(len(legal_moves))
        return legal_moves[idx]


class MCTSAgent:
    """Agent using MCTS with random rollouts (no neural network)."""
    
    def __init__(self, num_simulations: int = 100):
        """
        Initialize agent.
        
        Args:
            num_simulations: number of MCTS simulations per move
        """
        self.mcts = RandomRolloutMCTS(num_simulations)
    
    def select_move(self, game: Brandubh, temperature: float = 0.0) -> Tuple[int, int, int, int]:
        """Select a move using MCTS with random rollouts."""
        return self.mcts.select_move(game, temperature)


def play_game(agent1, agent2, display: bool = True) -> int:
    """
    Play a game between two agents.
    
    Args:
        agent1: agent playing as attackers
        agent2: agent playing as defenders
        display: whether to print the game
    
    Returns:
        winner: 0 for attackers, 1 for defenders
    """
    game = Brandubh()
    agents = [agent1, agent2]
    move_count = 0
    
    if display:
        print("=" * 50)
        print("Starting game")
        print("=" * 50)
        print("\nInitial board:")
        print(game)
        print("\n")
    
    while not game.game_over:
        agent = agents[game.current_player]
        
        # Select move
        move = agent.select_move(game)
        
        if move is None:
            # No legal moves
            game.game_over = True
            game.winner = 1 - game.current_player
            break
        
        if display:
            player_name = "Attackers" if game.current_player == 0 else "Defenders"
            from_r, from_c, to_r, to_c = move
            print(f"Move {move_count + 1}: {player_name} move ({from_r},{from_c}) → ({to_r},{to_c})")
        
        # Make move
        game.make_move(move)
        move_count += 1
        
        if display:
            print(game)
            print("\n")
        
        # Safety check
        if move_count > 500:
            if display:
                print("Game exceeded 500 moves - draw")
            game.game_over = True
            game.winner = 0
            break
    
    if display:
        print("=" * 50)
        winner_name = "Attackers" if game.winner == 0 else "Defenders"
        print(f"Game Over! Winner: {winner_name}")
        print(f"Total moves: {move_count}")
        print("=" * 50)
    
    return game.winner


def evaluate_agents(agent1, agent2, num_games: int = 10) -> dict:
    """
    Evaluate two agents against each other.
    
    Args:
        agent1: first agent
        agent2: second agent
        num_games: number of games to play
    
    Returns:
        dict with statistics
    """
    agent1_wins = 0
    agent2_wins = 0
    agent1_attacker_wins = 0
    agent2_attacker_wins = 0
    
    print(f"Playing {num_games} games...")
    
    for i in range(num_games):
        # Alternate who plays attackers
        if i % 2 == 0:
            winner = play_game(agent1, agent2, display=False)
            if winner == 0:
                agent1_wins += 1
                agent1_attacker_wins += 1
            else:
                agent2_wins += 1
        else:
            winner = play_game(agent2, agent1, display=False)
            if winner == 0:
                agent2_wins += 1
                agent2_attacker_wins += 1
            else:
                agent1_wins += 1
        
        if (i + 1) % 5 == 0:
            print(f"Completed {i + 1}/{num_games} games...")
    
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    print(f"Agent 1 wins: {agent1_wins}/{num_games} ({100*agent1_wins/num_games:.1f}%)")
    print(f"Agent 2 wins: {agent2_wins}/{num_games} ({100*agent2_wins/num_games:.1f}%)")
    print(f"Agent 1 as attacker: {agent1_attacker_wins}/{num_games//2} wins")
    print(f"Agent 2 as attacker: {agent2_attacker_wins}/{num_games//2} wins")
    print("=" * 50)
    
    return {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'agent1_attacker_wins': agent1_attacker_wins,
        'agent2_attacker_wins': agent2_attacker_wins,
    }


if __name__ == "__main__":
    print("Brandubh Agent Module")
    print("=" * 50)
    print("\nCreating agents...")
    
    # Create agents
    random_agent = RandomAgent()
    print("✓ Random agent created")
    
    mcts_agent = MCTSAgent(num_simulations=50)
    print("✓ MCTS agent created (50 simulations)")
    
    # Note: Neural network agent requires PyTorch to be installed
    try:
        nn_agent = BrandubhAgent(num_simulations=25)
        print("✓ Neural network agent created (25 simulations)")
        
        print("\nTesting neural network agent vs random...")
        play_game(nn_agent, random_agent, display=True)
        
    except Exception as e:
        print(f"✗ Could not create neural network agent: {e}")
        print("\nTesting MCTS agent vs random instead...")
        play_game(mcts_agent, random_agent, display=True)
