"""
Monte Carlo Tree Search for Brandubh with neural network evaluation.
Based on AlphaZero MCTS algorithm.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math

from brandubh import Brandubh


class MCTSNode:
    """
    Node in the MCTS tree.
    
    Stores statistics for each state-action pair:
    - N(s,a): visit count
    - W(s,a): total action value
    - Q(s,a): mean action value
    - P(s,a): prior probability from neural network
    """
    
    def __init__(self, game: Brandubh, parent=None, parent_action=None, prior: float = 0.0):
        self.game = game
        self.parent = parent
        self.parent_action = parent_action
        self.prior = prior
        
        # Children: dict mapping action (move tuple) to MCTSNode
        self.children: Dict[Tuple, MCTSNode] = {}
        
        # Statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0
        
        # Cached legal moves and policy
        self._legal_moves = None
        self._policy_probs = None
        self._is_expanded = False
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (not expanded)."""
        return not self._is_expanded
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal game state."""
        return self.game.game_over
    
    def get_legal_moves(self) -> List[Tuple]:
        """Get legal moves for this state."""
        if self._legal_moves is None:
            self._legal_moves = self.game.get_legal_moves()
        return self._legal_moves
    
    def expand(self, policy_probs: np.ndarray):
        """
        Expand this node by creating children for all legal moves.
        
        Args:
            policy_probs: probability distribution over moves (from neural network)
        """
        if self._is_expanded:
            return
        
        legal_moves = self.get_legal_moves()
        
        # Normalize policy over legal moves only
        legal_probs = policy_probs[policy_probs > 0]
        if len(legal_probs) > 0 and legal_probs.sum() > 0:
            legal_probs = legal_probs / legal_probs.sum()
        else:
            # Uniform distribution if no valid probabilities
            legal_probs = np.ones(len(legal_moves)) / len(legal_moves)
        
        # Create child nodes
        for move, prob in zip(legal_moves, legal_probs):
            child_game = self.game.clone()
            child_game.make_move(move)
            self.children[move] = MCTSNode(child_game, parent=self, parent_action=move, prior=prob)
        
        self._is_expanded = True
    
    def select_child(self, c_puct: float = 1.4) -> Tuple[Tuple, 'MCTSNode']:
        """
        Select best child using PUCT algorithm.
        
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            c_puct: exploration constant
        
        Returns:
            (action, child_node)
        """
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # Calculate sqrt(N(s)) once
        sqrt_parent_visits = math.sqrt(self.visit_count)
        
        for action, child in self.children.items():
            # Q value (from child's perspective, so negate for parent)
            q_value = -child.mean_value if child.visit_count > 0 else 0
            
            # U value (exploration bonus)
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def update(self, value: float):
        """
        Update node statistics after a simulation.
        
        Args:
            value: value from the perspective of the player at this node
        """
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count
    
    def get_visit_distribution(self, temperature: float = 1.0) -> Dict[Tuple, float]:
        """
        Get probability distribution over actions based on visit counts.
        
        Args:
            temperature: 
                - temperature = 1: proportional to visit counts
                - temperature -> 0: approaches argmax
                - temperature > 1: more uniform
        
        Returns:
            dict mapping actions to probabilities
        """
        if not self.children:
            return {}
        
        actions = list(self.children.keys())
        visits = np.array([self.children[a].visit_count for a in actions])
        
        if temperature == 0:
            # Deterministic: choose most visited
            probs = np.zeros(len(actions))
            probs[np.argmax(visits)] = 1.0
        else:
            # Apply temperature
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
        
        return {action: prob for action, prob in zip(actions, probs)}


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.
    """
    
    def __init__(self, network, num_simulations: int = 100, c_puct: float = 1.4, 
                 device: str = 'cpu'):
        """
        Initialize MCTS.
        
        Args:
            network: BrandubhNet instance
            num_simulations: number of MCTS simulations per move
            c_puct: exploration constant
            device: 'cpu' or 'cuda'
        """
        self.network = network
        self.network.eval()
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.network.to(device)
        
        # Timing statistics (accumulated across searches)
        self.timing_stats = {
            'selection': 0.0,
            'terminal_eval': 0.0,
            'network_eval': 0.0,
            'expansion': 0.0,
            'backup': 0.0,
            'game_clone': 0.0,
            'get_legal_moves': 0.0
        }
    
    def reset_timing_stats(self):
        """Reset timing statistics."""
        for key in self.timing_stats:
            self.timing_stats[key] = 0.0
    
    def get_timing_stats(self):
        """Get timing statistics."""
        return self.timing_stats.copy()
    
    def search(self, game: Brandubh) -> Dict[Tuple, float]:
        """
        Run MCTS from the given game state.
        
        Args:
            game: current game state
        
        Returns:
            dict mapping moves to visit probabilities
        """
        import time
        
        t0 = time.perf_counter()
        root = MCTSNode(game.clone())
        self.timing_stats['game_clone'] += time.perf_counter() - t0
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree until leaf
            t0 = time.perf_counter()
            while not node.is_leaf() and not node.is_terminal():
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
            self.timing_stats['selection'] += time.perf_counter() - t0
            
            # Evaluate leaf with neural network
            value = 0
            if node.is_terminal():
                # Terminal node: use game result
                t0 = time.perf_counter()
                if node.game.winner == node.game.current_player:
                    value = 1.0
                elif node.game.winner == 1 - node.game.current_player:
                    value = -1.0
                else:
                    value = 0.0
                self.timing_stats['terminal_eval'] += time.perf_counter() - t0
            else:
                # Non-terminal leaf: evaluate with network and expand
                t0 = time.perf_counter()
                policy_probs, value = self._evaluate(node.game)
                self.timing_stats['network_eval'] += time.perf_counter() - t0
                
                t0 = time.perf_counter()
                node.expand(policy_probs)
                self.timing_stats['expansion'] += time.perf_counter() - t0
            
            # Backup: propagate value up the tree
            t0 = time.perf_counter()
            for node in reversed(search_path):
                node.update(value)
                value = -value  # Flip value for opponent
            self.timing_stats['backup'] += time.perf_counter() - t0
        
        return root.get_visit_distribution()
    
    def _evaluate(self, game: Brandubh) -> Tuple[np.ndarray, float]:
        """
        Evaluate a game state with the neural network.
        
        Args:
            game: game state to evaluate
        
        Returns:
            policy_probs: probability distribution over legal moves
            value: estimated value for current player
        """
        import time
        from network import MoveEncoder
        
        # Get state representation
        state = game.get_state()
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
        # Evaluate with network
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
        
        policy_logits = policy_logits.cpu().numpy()[0]
        value = value.cpu().item()
        
        # Mask illegal moves
        t0 = time.perf_counter()
        legal_mask = MoveEncoder.get_legal_move_mask(game)
        policy_logits = policy_logits * legal_mask + (1 - legal_mask) * (-1e8)
        
        # Convert to probabilities
        policy_probs = self._softmax(policy_logits)
        
        # Extract probabilities for legal moves only
        legal_moves = game.get_legal_moves()
        move_indices = [MoveEncoder.encode_move(move) for move in legal_moves]
        legal_probs = policy_probs[move_indices]
        self.timing_stats['get_legal_moves'] += time.perf_counter() - t0
        
        return legal_probs, value
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values for array x."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def select_move(self, game: Brandubh, temperature: float = 1.0) -> Tuple[int, int, int, int]:
        """
        Select a move using MCTS.
        
        Args:
            game: current game state
            temperature: sampling temperature (0 = deterministic)
        
        Returns:
            move: (from_row, from_col, to_row, to_col)
        """
        visit_probs = self.search(game)
        
        if not visit_probs:
            # No moves available (shouldn't happen)
            legal_moves = game.get_legal_moves()
            return legal_moves[0] if legal_moves else None
        
        moves = list(visit_probs.keys())
        probs = np.array(list(visit_probs.values()))
        
        if temperature == 0:
            # Choose most visited
            move = moves[np.argmax(probs)]
        else:
            # Sample proportionally to visit counts
            move_idx = np.random.choice(len(moves), p=probs)
            move = moves[move_idx]
        
        return move


class RandomRolloutMCTS:
    """
    Simple MCTS with random rollouts (no neural network).
    Useful for baseline comparison.
    """
    
    def __init__(self, num_simulations: int = 100):
        self.num_simulations = num_simulations
    
    def search(self, game: Brandubh) -> Dict[Tuple, float]:
        """Run MCTS with random rollouts."""
        root = MCTSNode(game.clone())
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while not node.is_leaf() and not node.is_terminal():
                action, node = node.select_child()
                search_path.append(node)
            
            # Expansion
            if not node.is_terminal():
                legal_moves = node.get_legal_moves()
                uniform_probs = np.ones(len(legal_moves)) / len(legal_moves)
                node.expand(uniform_probs)
                
                # Select random child for rollout
                if node.children:
                    actions = list(node.children.keys())
                    action = actions[np.random.randint(len(actions))]
                    node = node.children[action]
                    search_path.append(node)
            
            # Simulation: random rollout
            value = self._rollout(node.game)
            
            # Backpropagation
            for node in reversed(search_path):
                node.update(value)
                value = -value
        
        return root.get_visit_distribution()
    
    def _rollout(self, game: Brandubh) -> float:
        """Simulate random play until terminal state."""
        game = game.clone()
        
        while not game.game_over:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            move_idx = np.random.randint(len(legal_moves))
            move = legal_moves[move_idx]
            game.make_move(move)
        
        if game.winner == game.current_player:
            return 1.0
        elif game.winner == 1 - game.current_player:
            return -1.0
        else:
            return 0.0
    
    def select_move(self, game: Brandubh, temperature: float = 0.0) -> Tuple[int, int, int, int]:
        """Select a move using random rollout MCTS."""
        visit_probs = self.search(game)
        
        if not visit_probs:
            legal_moves = game.get_legal_moves()
            return legal_moves[0] if legal_moves else None
        
        moves = list(visit_probs.keys())
        probs = np.array(list(visit_probs.values()))
        
        if temperature == 0:
            move = moves[np.argmax(probs)]
        else:
            move = np.random.choice(moves, p=probs)
        
        return move


if __name__ == "__main__":
    print("Testing MCTS with random rollouts...")
    
    game = Brandubh()
    mcts = RandomRolloutMCTS(num_simulations=50)
    
    print("Initial board:")
    print(game)
    print("\nRunning MCTS...")
    
    move = mcts.select_move(game)
    print(f"\nBest move: {move}")
    
    game.make_move(move)
    print("\nBoard after move:")
    print(game)
    
    print("\nMCTS test completed!")
