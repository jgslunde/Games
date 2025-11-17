"""
Generic agent wrapper for playing Tafl games (Brandubh, Tablut, etc.) using MCTS with neural network.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Any


try:
    from mcts import MCTS, RandomRolloutMCTS
except ImportError:
    # Allow module to be imported even if mcts is not available
    MCTS = None
    RandomRolloutMCTS = None


class Agent:
    """
    Generic agent that uses MCTS with neural network to play Tafl games.
    Works with any game (Brandubh, Tablut, etc.) and network architecture.
    """
    
    def __init__(self, network: Optional[Any] = None, network_class: Optional[type] = None,
                 move_encoder_class: Optional[type] = None,
                 num_simulations: int = 100, 
                 c_puct: float = 1.4, device: str = 'cpu', 
                 add_dirichlet_noise: bool = False, dirichlet_alpha: float = 0.3, 
                 dirichlet_epsilon: float = 0.25):
        """
        Initialize agent.
        
        Args:
            network: trained network instance (if None and network_class provided, creates untrained network)
            network_class: class to instantiate if network is None (e.g., BrandubhNet, TablutNet)
            move_encoder_class: MoveEncoder class for encoding/decoding moves (e.g., MoveEncoder, TablutMoveEncoder)
            num_simulations: number of MCTS simulations per move
            c_puct: exploration constant
            device: 'cpu' or 'cuda'
            add_dirichlet_noise: whether to add Dirichlet noise to root (for evaluation diversity)
            dirichlet_alpha: concentration parameter for Dirichlet noise
            dirichlet_epsilon: weight of Dirichlet noise
        """
        if network is None and network_class is not None:
            network = network_class()
        
        self.network = network
        self.mcts = MCTS(network, num_simulations, c_puct, device, 
                        dirichlet_alpha, dirichlet_epsilon, add_dirichlet_noise,
                        move_encoder_class=move_encoder_class)
        self.device = device
    
    def select_move(self, game, temperature: float = 0.0) -> Tuple[int, int, int, int]:
        """
        Select a move for the current game state.
        
        Args:
            game: current game state (any Tafl game)
            temperature: sampling temperature (0 = deterministic)
        
        Returns:
            move: (from_row, from_col, to_row, to_col)
        """
        return self.mcts.select_move(game, temperature)
    
    def select_move_with_stats(self, game, temperature: float = 0.0):
        """
        Select a move and return statistics.
        
        Args:
            game: current game state
            temperature: sampling temperature (0 = deterministic)
        
        Returns:
            move: (from_row, from_col, to_row, to_col)
            value: network value estimate from current player's perspective
            visit_prob: proportion of MCTS visits to selected move
        """
        # Run MCTS search to get visit distribution
        visit_probs = self.mcts.search(game)
        
        if not visit_probs:
            # No moves available
            legal_moves = game.get_legal_moves()
            move = legal_moves[0] if legal_moves else None
            return move, 0.0, 0.0
        
        moves = list(visit_probs.keys())
        probs = np.array(list(visit_probs.values()))
        
        if temperature == 0:
            # Choose most visited
            move_idx = np.argmax(probs)
            move = moves[move_idx]
        else:
            # Sample proportionally to visit counts
            move_idx = np.random.choice(len(moves), p=probs)
            move = moves[move_idx]
        
        visit_prob = probs[move_idx]
        
        # Get value estimate from root node
        # The MCTS root stores the value from current player's perspective
        value = self.mcts.root.mean_value if self.mcts.root else 0.0
        
        return move, value, visit_prob
    
    def select_move_with_time_limit(self, game, time_limit: float, temperature: float = 0.0):
        """
        Select a move using a time limit instead of fixed simulation count.
        
        Args:
            game: current game state
            time_limit: time limit in seconds for MCTS search
            temperature: sampling temperature (0 = deterministic)
        
        Returns:
            move: (from_row, from_col, to_row, to_col)
            value: network value estimate from current player's perspective
            visit_prob: proportion of MCTS visits to selected move
        """
        import time
        
        # Get legal moves - if none, return immediately
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None, 0.0, 0.0
        
        # Run MCTS for the specified time limit
        start_time = time.time()
        simulations = 0
        
        # Initialize MCTS tree
        self.mcts.root = self.mcts.select_root(game)
        
        # Run simulations until time limit
        while time.time() - start_time < time_limit:
            self.mcts._run_single_simulation(game)
            simulations += 1
            
            # Check time every 10 simulations to reduce overhead
            if simulations % 10 == 0:
                if time.time() - start_time >= time_limit:
                    break
        
        # Get visit counts and select move
        visit_counts = {}
        for move, edge in self.mcts.root.children.items():
            visit_counts[move] = edge.visit_count
        
        if not visit_counts:
            # Fallback to random legal move
            move = legal_moves[0]
            return move, 0.0, 0.0
        
        # Calculate probabilities
        total_visits = sum(visit_counts.values())
        visit_probs = {m: v / total_visits for m, v in visit_counts.items()}
        
        moves = list(visit_probs.keys())
        probs = np.array(list(visit_probs.values()))
        
        if temperature == 0:
            # Choose most visited
            move_idx = np.argmax(probs)
            move = moves[move_idx]
        else:
            # Sample proportionally to visit counts
            move_idx = np.random.choice(len(moves), p=probs)
            move = moves[move_idx]
        
        visit_prob = probs[move_idx]
        
        # Get value estimate from root node
        value = self.mcts.root.mean_value if self.mcts.root else 0.0
        
        return move, value, visit_prob
    
    def load_weights(self, path: str):
        """Load network weights from file."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()
    
    def save_weights(self, path: str):
        """Save network weights to file."""
        torch.save(self.network.state_dict(), path)


class RandomAgent:
    """Simple random agent for baseline comparison. Works with any Tafl game."""
    
    def select_move(self, game, temperature: float = 0.0) -> Optional[Tuple[int, int, int, int]]:
        """Select a random legal move."""
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        idx = np.random.randint(len(legal_moves))
        return legal_moves[idx]


class MCTSAgent:
    """Agent using MCTS with random rollouts (no neural network). Works with any Tafl game."""
    
    def __init__(self, num_simulations: int = 100):
        """
        Initialize agent.
        
        Args:
            num_simulations: number of MCTS simulations per move
        """
        self.mcts = RandomRolloutMCTS(num_simulations)
    
    def select_move(self, game, temperature: float = 0.0) -> Optional[Tuple[int, int, int, int]]:
        """Select a move using MCTS with random rollouts."""
        return self.mcts.select_move(game, temperature)


def play_game(agent1, agent2, game_class, display: bool = True, 
              temperature: float = 0.0, temperature_mode: str = "fixed",
              temperature_threshold: int = 0, temperature_decay_moves: int = 0) -> int:
    """
    Play a game between two agents.
    
    Args:
        agent1: agent playing as attackers
        agent2: agent playing as defenders
        game_class: game class to instantiate (e.g., Brandubh, Tablut)
        display: whether to print the game
        temperature: sampling temperature for move selection (0 = deterministic)
        temperature_mode: "fixed" (drop at threshold), "king" (drop when king leaves), or "decay" (linear)
        temperature_threshold: move number after which temperature drops to 0 (for "fixed" mode)
        temperature_decay_moves: number of moves for linear decay (for "decay" mode)
    
    Returns:
        winner: 0 for attackers, 1 for defenders, None for draw
    """
    game = game_class()
    agents = [agent1, agent2]
    move_count = 0
    
    # Track king leaving throne for "king" threshold
    king_left_throne = False
    
    if display:
        print("=" * 50)
        print("Starting game")
        print("=" * 50)
        print("\nInitial board:")
        print(game)
        print("\n")
    
    while not game.game_over:
        agent = agents[game.current_player]
        
        # Determine temperature for this move
        if temperature_mode == "king":
            # Drop temperature after king leaves throne
            if not king_left_throne:
                # Check if king is still on throne
                king_pos = None
                for r in range(game.board_size):
                    for c in range(game.board_size):
                        if game.board[r, c] == 2:  # King
                            king_pos = (r, c)
                            break
                    if king_pos:
                        break
                
                # Check if king is on throne (center square)
                throne_pos = (game.board_size // 2, game.board_size // 2)
                if king_pos != throne_pos:
                    king_left_throne = True
            
            temp = 0.0 if king_left_throne else temperature
        elif temperature_mode == "decay":
            # Linear decay over specified number of moves
            if move_count < temperature_decay_moves:
                temp = temperature * (1.0 - move_count / temperature_decay_moves)
            else:
                temp = 0.0
        else:  # "fixed" mode
            # Drop temperature after specified number of moves
            temp = 0.0 if move_count >= temperature_threshold else temperature
        
        # Select move with temperature
        move = agent.select_move(game, temperature=temp)
        
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
            game.winner = None  # Draw
            break
    
    if display:
        print("=" * 50)
        if game.winner is None:
            winner_name = "Draw"
        else:
            winner_name = "Attackers" if game.winner == 0 else "Defenders"
        print(f"Game Over! Winner: {winner_name}")
        print(f"Total moves: {move_count}")
        print("=" * 50)
    
    return game.winner


def evaluate_agents(agent1, agent2, game_class, num_games: int = 10) -> dict:
    """
    Evaluate two agents against each other.
    
    Args:
        agent1: first agent
        agent2: second agent
        game_class: game class to instantiate (e.g., Brandubh, Tablut)
        num_games: number of games to play
    
    Returns:
        dict with statistics
    """
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    agent1_attacker_wins = 0
    agent2_attacker_wins = 0
    
    print(f"Playing {num_games} games...")
    
    for i in range(num_games):
        # Alternate who plays attackers
        if i % 2 == 0:
            winner = play_game(agent1, agent2, game_class, display=False)
            if winner is None:
                draws += 1
            elif winner == 0:
                agent1_wins += 1
                agent1_attacker_wins += 1
            else:
                agent2_wins += 1
        else:
            winner = play_game(agent2, agent1, game_class, display=False)
            if winner is None:
                draws += 1
            elif winner == 0:
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
    print(f"Draws: {draws}/{num_games} ({100*draws/num_games:.1f}%)")
    print(f"Agent 1 as attacker: {agent1_attacker_wins}/{num_games//2} wins")
    print(f"Agent 2 as attacker: {agent2_attacker_wins}/{num_games//2} wins")
    print("=" * 50)
    
    return {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'agent1_attacker_wins': agent1_attacker_wins,
        'agent2_attacker_wins': agent2_attacker_wins,
    }


# Backward compatibility: Create aliases with game-specific names
BrandubhAgent = Agent
TablutAgent = Agent


if __name__ == "__main__":
    print("Generic Tafl Agent Module")
    print("=" * 50)
    
    # Try to import Brandubh for testing
    try:
        from brandubh import Brandubh
        from network import BrandubhNet, MoveEncoder
        
        print("\nTesting with Brandubh (7x7)...")
        print("Creating agents...")
        
        # Create agents
        random_agent = RandomAgent()
        print("✓ Random agent created")
        
        mcts_agent = MCTSAgent(num_simulations=10)
        print("✓ MCTS agent created (10 simulations)")
        
        # Note: Neural network agent requires PyTorch to be installed
        try:
            nn_agent = Agent(network_class=BrandubhNet, move_encoder_class=MoveEncoder, num_simulations=10)
            print("✓ Neural network agent created (10 simulations)")
            
            print("\nTesting neural network agent vs random...")
            play_game(nn_agent, random_agent, Brandubh, display=True)
            
        except Exception as e:
            print(f"✗ Could not create neural network agent: {e}")
            print("\nTesting MCTS agent vs random instead...")
            play_game(mcts_agent, random_agent, Brandubh, display=True)
            
    except ImportError:
        print("\nBrandubh not available. Skipping tests.")
        print("To test with a specific game, import it and call:")
        print("  play_game(agent1, agent2, GameClass, display=True)")
