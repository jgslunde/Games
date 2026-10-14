"""
Monte Carlo Tree Search for Brandubh with neural network evaluation.
Based on AlphaZero MCTS algorithm.

Supports history planes: The network can receive T timesteps of history
rather than just the current state. When history_length > 1, MCTS tracks
the game state history and constructs the appropriate input for evaluation.
"""

import sys
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math

from brandubh import Brandubh
from network import StateHistory


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
        # Lazy-initialized nodes are never terminal (game state not yet created)
        if self.game is None:
            return False
        return self.game.game_over
    
    def get_legal_moves(self) -> List[Tuple]:
        """Get legal moves for this state."""
        if self._legal_moves is None:
            self._legal_moves = self.game.get_legal_moves()
        return self._legal_moves
    
    def expand(self, policy_probs: np.ndarray):
        """
        Expand this node by creating children for all legal moves.
        Uses lazy initialization - child game states are created only when visited.
        
        Args:
            policy_probs: probability distribution over moves (from neural network)
        """
        if self._is_expanded:
            return
        
        legal_moves = self.get_legal_moves()
        
        # Handle case where there are no legal moves
        # This can happen in rare game states (though the game should be over)
        if len(legal_moves) == 0:
            # Don't mark as expanded - this is effectively a terminal node
            return
        
        # Use the probs as-is. If sum is 0 (unlikely), handle gracefully.
        if policy_probs.sum() > 0:
            legal_probs = policy_probs / policy_probs.sum()
        else:
            # Uniform distribution if no valid probabilities
            legal_probs = np.ones(len(legal_moves)) / len(legal_moves)
        
        # Create child nodes without game states (lazy initialization)
        for move, prob in zip(legal_moves, legal_probs):
            self.children[move] = MCTSNode(None, parent=self, parent_action=move, prior=prob)
        
        self._is_expanded = True
    
    def select_child(self, c_puct: float = 1.4, fpu_reduction: float = -0.5) -> Tuple[Tuple, 'MCTSNode']:
        """
        Select best child using PUCT algorithm.
        Lazily initializes child game state on first selection.
        
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        For unvisited nodes, uses First Play Urgency (FPU) relative to parent:
        Q_unvisited = parent_Q + fpu_reduction
        
        Args:
            c_puct: exploration constant
            fpu_reduction: First Play Urgency reduction relative to parent's Q-value (default: -0.5)
                          Negative values make unvisited nodes look worse than parent (pessimistic/conservative)
        
        Returns:
            (action, child_node) or (None, None) if no children available
        """
        # Safety check: handle case where there are no children
        # This can happen if expand() was called with no legal moves
        if not self.children:
            return None, None
        
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # Calculate sqrt(N(s)) once
        sqrt_parent_visits = math.sqrt(self.visit_count)

        # FPU value for unvisited children: the parent's own value plus the reduction. Both q_value below
        # and self.mean_value are seen from the player to move at this node, so no sign flip is needed.
        fpu_value = self.mean_value + fpu_reduction

        for action, child in self.children.items():
            # Q value (child.mean_value is from the child's player, so negate for the player at this node)
            q_value = -child.mean_value if child.visit_count > 0 else fpu_value
            
            # U value (exploration bonus)
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        # Lazy initialization: create game state for selected child if not already done
        if best_child.game is None:
            best_child.game = self.game.clone()
            best_child.game.make_move(best_action)
        
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
        if visits.sum() == 0:
            # No child visited yet (only one simulation was run): fall back to the network priors
            visits = np.array([self.children[a].prior for a in actions])

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
    Monte Carlo Tree Search with neural network evaluation.
    Generic implementation that works with any game and move encoder.
    """
    
    def __init__(self, network, num_simulations: int = 100, c_puct: float = 1.4, 
                 device: str = 'cpu', dirichlet_alpha: float = 0.3, 
                 dirichlet_epsilon: float = 0.25, add_dirichlet_noise: bool = False,
                 move_encoder_class=None, fpu_reduction: float = -0.5,
                 search_discount: float = 1.0, history_length: int = 1):
        """
        Initialize MCTS.
        
        Args:
            network: neural network for evaluating positions
            num_simulations: number of simulations per search
            c_puct: exploration constant for PUCT
            device: 'cpu' or 'cuda'
            dirichlet_alpha: concentration parameter for Dirichlet noise
            dirichlet_epsilon: weight of Dirichlet noise at root
            add_dirichlet_noise: whether to add exploration noise
            move_encoder_class: MoveEncoder class for encoding/decoding moves
                               (if None, imports from network module for backward compatibility)
            fpu_reduction: First Play Urgency reduction relative to parent's Q-value (default: -0.5)
                          Negative = pessimistic (unvisited nodes look worse than parent)
                          This follows Leela Chess Zero's implementation
            search_discount: Discount factor applied per move during value backup (default: 1.0 = no discount)
                            Values < 1 make the network prefer shorter paths to victory
            history_length: Number of timesteps of game history to include in network input (default: 1)
                           When > 1, the network receives T*3 + 1 input planes instead of 4
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_dirichlet_noise = add_dirichlet_noise
        self.move_encoder_class = move_encoder_class
        self.fpu_reduction = fpu_reduction
        self.search_discount = search_discount
        self.history_length = history_length
        
        # Performance tracking
        self.timing_stats = {
            'selection': 0.0,
            'network_eval': 0.0,
            'expansion': 0.0,
            'backup': 0.0,
            'terminal_eval': 0.0,
            'get_legal_moves': 0.0,
            'game_clone': 0.0,
        }
        
        # For debugging
        self.root = None
    
    def reset_timing_stats(self):
        """Reset timing statistics."""
        for key in self.timing_stats:
            self.timing_stats[key] = 0.0
    
    def get_timing_stats(self):
        """Get timing statistics."""
        return self.timing_stats.copy()
    
    def search(self, game: Brandubh, state_history: Optional[StateHistory] = None,
               time_limit: Optional[float] = None) -> Dict[Tuple, float]:
        """
        Run MCTS from the given game state.

        Args:
            game: current game state
            state_history: Optional StateHistory of the game so far. Its most recent entry must be the
                          current position (as tracked during self-play).
                          If None and history_length > 1, it is rebuilt from game.position_history.
                          If history_length == 1, this parameter is ignored.
            time_limit: If given, run simulations for this many seconds instead of num_simulations.

        Returns:
            dict mapping moves to visit probabilities
        """
        import time

        t_start = t0 = time.perf_counter()
        root = MCTSNode(game.clone())
        self.timing_stats['game_clone'] += time.perf_counter() - t0
        
        # Store root for later access to statistics
        self.root = root
        
        # Set up root history for history planes support
        # We only need to track history if history_length > 1
        if self.history_length > 1:
            if state_history is not None:
                # Use provided history (should already contain prior states)
                root_history = state_history.clone()
            else:
                # Create new history initialized with current state
                root_history = StateHistory.from_game(game, self.history_length)
        else:
            root_history = None  # No history tracking needed
        
        # Run simulations: num_simulations of them, or as many as fit in time_limit seconds
        num_simulations = self.num_simulations if time_limit is None else sys.maxsize
        for sim_idx in range(num_simulations):
            if time_limit is not None and time.perf_counter() - t_start > time_limit:
                break
            node = root
            search_path = [node]

            # Track history for this simulation (if needed)
            if root_history is not None:
                sim_history = root_history.clone()
            else:
                sim_history = None

            # Selection: traverse tree until leaf. An expanded node always has children, because
            # expand() leaves a node without legal moves unexpanded (such a node is terminal).
            t0 = time.perf_counter()
            while not node.is_leaf() and not node.is_terminal():
                action, node = node.select_child(self.c_puct, self.fpu_reduction)
                search_path.append(node)

                # Update history with the new state after the action
                if sim_history is not None:
                    sim_history.push(node.game.get_piece_planes())
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
                # Non-terminal leaf: evaluate with network and expand.
                # Legal moves are generated once here and cached on the node, so expand() reuses them.
                t0 = time.perf_counter()
                legal_moves = node.get_legal_moves()
                self.timing_stats['get_legal_moves'] += time.perf_counter() - t0

                t0 = time.perf_counter()
                policy_probs, value = self._evaluate(node.game, legal_moves, sim_history)
                self.timing_stats['network_eval'] += time.perf_counter() - t0

                t0 = time.perf_counter()
                node.expand(policy_probs)
                self.timing_stats['expansion'] += time.perf_counter() - t0

                # Add Dirichlet noise to root node after first expansion
                if sim_idx == 0 and node is root and self.add_dirichlet_noise:
                    self._add_dirichlet_noise_to_node(root)

            # Backup: propagate value up the tree with optional discounting
            t0 = time.perf_counter()
            for n in reversed(search_path):
                n.update(value)
                value = -value * self.search_discount  # Flip value for opponent and apply discount
            self.timing_stats['backup'] += time.perf_counter() - t0

        return root.get_visit_distribution()
    
    def _evaluate(self, game: Brandubh, legal_moves: List[Tuple],
                  state_history: Optional[StateHistory] = None) -> Tuple[np.ndarray, float]:
        """
        Evaluate a game state with the neural network.

        Args:
            game: game state to evaluate
            legal_moves: legal moves of `game`, in the order used for the returned probabilities
            state_history: Optional StateHistory for history planes support, whose most recent entry
                          is `game` itself. If provided, uses history to construct input tensor.
                          If None, uses game.get_state() (backward compatible).

        Returns:
            policy_probs: probability distribution over legal_moves (same order)
            value: estimated value for current player
        """
        # Get MoveEncoder class (backward compatible)
        if self.move_encoder_class is None:
            # Try to infer from network module name
            network_module = self.network.__class__.__module__
            if 'tablut' in network_module.lower():
                from network_tablut import TablutMoveEncoder
                move_encoder = TablutMoveEncoder
            elif 'hnefatafl' in network_module.lower():
                from network_hnefatafl import HnefataflMoveEncoder
                move_encoder = HnefataflMoveEncoder
            else:
                from network import MoveEncoder
                move_encoder = MoveEncoder
        else:
            move_encoder = self.move_encoder_class
        
        # Get state representation
        if state_history is not None and self.history_length > 1:
            # Use history planes. The history already ends with this position (search() pushed it), so it
            # gives the same layout as the training states: [current, previous, ..., player plane].
            state = state_history.get_state_with_history(game.current_player)
        else:
            # Standard single-state input (backward compatible)
            state = game.get_state()
        
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
        # Evaluate with network
        # Use inference_mode for better performance than no_grad
        with torch.inference_mode():
            policy_logits, value = self.network(state_tensor)
        
        policy_logits = policy_logits.cpu().numpy()[0]
        value = value.cpu().item()

        # Softmax over the legal moves only. This equals a softmax over all moves with illegal moves
        # masked out, followed by picking the legal entries.
        move_indices = [move_encoder.encode_move(move) for move in legal_moves]
        legal_logits = policy_logits[move_indices]
        legal_probs = self._softmax(legal_logits) if len(legal_moves) > 0 else legal_logits

        return legal_probs, value
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values for array x."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _add_dirichlet_noise_to_node(self, node: MCTSNode):
        """
        Add Dirichlet noise to the priors of a node's children.
        This adds exploration at the root node during evaluation.
        
        Args:
            node: node to add noise to (typically the root)
        """
        if not node.children:
            return
        
        actions = list(node.children.keys())
        num_actions = len(actions)
        
        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)
        
        # Mix noise with priors: P = (1-ε)*P_prior + ε*noise
        for action, noise_value in zip(actions, noise):
            child = node.children[action]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise_value
    
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
            # Sample from the visit shares raised to 1/temperature (sharper for temperature < 1)
            probs = probs ** (1.0 / temperature)
            move = moves[np.random.choice(len(moves), p=probs / probs.sum())]

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
                    
                    # Lazy initialization: create game state if not already done
                    if node.game is None:
                        parent = search_path[-2]
                        node.game = parent.game.clone()
                        node.game.make_move(action)
            
            # Simulation: random rollout
            value = self._rollout(node.game)
            
            # Backpropagation
            for node in reversed(search_path):
                node.update(value)
                value = -value
        
        return root.get_visit_distribution()
    
    def _rollout(self, game: Brandubh) -> float:
        """Simulate random play until terminal state. Returns the result for the player to move in `game`."""
        player = game.current_player
        game = game.clone()

        while not game.game_over:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            move_idx = np.random.randint(len(legal_moves))
            move = legal_moves[move_idx]
            game.make_move(move)

        if game.winner == player:
            return 1.0
        elif game.winner == 1 - player:
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
            # Sample an index: np.random.choice cannot pick from a list of move tuples directly
            probs = probs ** (1.0 / temperature)
            move = moves[np.random.choice(len(moves), p=probs / probs.sum())]

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
