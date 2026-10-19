"""
Agents that play the Tafl games (tafl.py): MCTS with a neural network, random rollouts or random
moves, and play_game to let two of them play one game.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Any

from mcts import MCTS, RandomRolloutMCTS
from tafl import Tafl


class Agent:
    """Plays with MCTS guided by a neural network."""

    def __init__(self, network: Any, num_simulations: int = 100,
                 c_puct: float = 1.4, device: str = 'cpu',
                 add_dirichlet_noise: bool = False, dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25, fpu_reduction: float = -0.5,
                 history_length: int = 1, search_discount: float = 1.0, feature_spec=None):
        """
        Initialize agent.

        Args:
            network: the network (a TaflNet, possibly compiled)
            num_simulations: number of MCTS simulations per move
            c_puct: exploration constant
            device: 'cpu' or 'cuda'
            add_dirichlet_noise: whether to add Dirichlet noise to root (for evaluation diversity)
            dirichlet_alpha: concentration parameter for Dirichlet noise
            dirichlet_epsilon: weight of Dirichlet noise
            fpu_reduction: First Play Urgency reduction relative to parent Q-value (default: -0.5)
            history_length: Number of game states to include in network input (default: 1; ignored
                            when feature_spec is given)
            search_discount: MCTS backup discount per move. Use the value the network was trained with,
                             because its value targets were discounted the same way (default: 1.0)
            feature_spec: The input planes the network expects (features.FeatureSpec). If None, it is
                          taken from network.feature_spec when the network has one, else from
                          history_length.
        """
        if feature_spec is None:
            feature_spec = getattr(network, 'feature_spec', None)

        self.network = network
        self.mcts = MCTS(network, num_simulations, c_puct, device,
                        dirichlet_alpha, dirichlet_epsilon, add_dirichlet_noise,
                        fpu_reduction=fpu_reduction, search_discount=search_discount,
                        history_length=history_length, feature_spec=feature_spec)
        self.feature_spec = self.mcts.feature_spec
        self.history_length = self.feature_spec.history_length
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
    
    def select_move_with_stats(self, game, temperature: float = 0.0, time_limit: Optional[float] = None):
        """
        Select a move and return statistics.

        Args:
            game: current game state
            temperature: sampling temperature (0 = deterministic)
            time_limit: if given, search for this many seconds instead of num_simulations

        Returns:
            move: (from_row, from_col, to_row, to_col)
            value: network value estimate from current player's perspective
            visit_prob: proportion of MCTS visits to selected move
        """
        # Run MCTS search to get visit distribution
        visit_probs = self.mcts.search(game, time_limit=time_limit)

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
        else:
            # Sample from the visit shares raised to 1/temperature (sharper for temperature < 1).
            # Dividing by the largest share first keeps it at 1, so a small temperature cannot give
            # all zeros.
            sharpened = (probs / probs.max()) ** (1.0 / temperature)
            move_idx = np.random.choice(len(moves), p=sharpened / sharpened.sum())
        move = moves[move_idx]

        visit_prob = probs[move_idx]

        # Get value estimate from root node
        # The MCTS root stores the value from current player's perspective
        value = self.mcts.root.mean_value if self.mcts.root else 0.0

        return move, value, visit_prob

    def load_weights(self, path: str):
        """Load network weights from a file written by save_weights or from a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
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


def play_game(agent1, agent2, rules, display: bool = True,
              temperature: float = 0.0, temperature_mode: str = "fixed",
              temperature_threshold: int = 0, temperature_decay_moves: int = 0,
              max_moves: int = 500) -> int:
    """
    Play a game between two agents.

    Args:
        agent1: agent playing as attackers
        agent2: agent playing as defenders
        rules: the rule set (a tafl.Rules object or the name of one)
        display: whether to print the game
        temperature: sampling temperature for move selection (0 = deterministic)
        temperature_mode: "fixed" (drop at threshold), "king" (drop when king leaves), or "decay" (linear)
        temperature_threshold: move number after which temperature drops to 0 (for "fixed" mode)
        temperature_decay_moves: number of moves for linear decay (for "decay" mode)
        max_moves: the game is a draw once more than this many plies have been played

    Returns:
        winner: 0 for attackers, 1 for defenders, None for draw
    """
    game = Tafl(rules)
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
                # Check if king is still on throne (KING = 3)
                king_on_throne = (game.board[game.throne] == 3)
                if not king_on_throne:
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
        
        # Move limit (a win on the last allowed ply still counts as a win)
        if move_count > max_moves and not game.game_over:
            if display:
                print(f"Game exceeded {max_moves} moves - draw")
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


if __name__ == "__main__":
    # Demo: random-rollout MCTS (attackers) against random moves (defenders) under Brandubh 2
    play_game(MCTSAgent(num_simulations=50), RandomAgent(), "brandubh", display=True)
