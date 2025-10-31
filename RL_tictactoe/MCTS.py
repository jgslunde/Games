import numpy as np
import torch
import math
from copy import deepcopy

class MCTSNode:
    """
    A node in the MCTS tree.
    Each node represents a game state.
    """
    def __init__(self, game_state, parent=None, prior_prob=1.0, action_taken=None):
        self.game_state = game_state  # TicTacToeGame instance
        self.parent = parent
        self.children = {}  # Dictionary mapping action -> MCTSNode
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.action_taken = action_taken  # Action that led to this node
        
    def is_fully_expanded(self):
        """Check if all legal moves have been tried."""
        legal_moves = self.game_state.get_legal_moves()
        return len(self.children) == len(legal_moves)
    
    def is_terminal(self):
        """Check if this is a terminal game state."""
        return self.game_state.is_game_over()
    
    def get_value(self):
        """Get the average value (Q-value) of this node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct=1.0):
        """
        Select the best child using the PUCT formula.
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        best_score = -float('inf')
        best_child = None
        
        for action, child in self.children.items():
            # Q-value (average value from child's perspective)
            q_value = child.get_value()
            
            # Exploration term
            u_value = c_puct * child.prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)
            
            # PUCT score (note: we negate q_value because child is opponent's turn)
            score = -q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child


class MCTS:
    """
    Monte Carlo Tree Search guided by a neural network.
    """
    def __init__(self, network, num_simulations=50, c_puct=2.0):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def search(self, game_state):
        """
        Run MCTS from the given game state.
        Returns: action probabilities based on visit counts.
        """
        # Create root node
        root = MCTSNode(deepcopy(game_state))
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 1. Selection: traverse tree until we find a node to expand
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # 2. Expansion & Evaluation
            if not node.is_terminal():
                # Get network predictions for this state
                value = self._expand_node(node)
            else:
                # Terminal node: use actual game outcome
                winner = node.game_state.check_winner()
                # Value from the perspective of the player who just moved
                value = winner if winner is not None else 0
                # Flip perspective (we want value from current player's perspective)
                value = -value
            
            # 3. Backpropagation: update all nodes in the search path
            self._backpropagate(search_path, value)
        
        # Return action probabilities based on visit counts
        return self._get_action_probs(root)
    
    def _expand_node(self, node):
        """
        Expand a node by creating children for all legal moves.
        Returns the value estimate from the neural network.
        """
        with torch.no_grad():
            board_tensor = node.game_state.get_nn_input()
            policy_output, value_output = self.network(board_tensor)
            
            # Extract policy and value
            policy_probs = policy_output.squeeze(0).numpy()
            value = value_output.item()
            
        # Get legal moves
        legal_moves = node.game_state.get_legal_moves()
        
        # Create mask for legal moves
        legal_mask = np.zeros(9)
        legal_mask[legal_moves] = 1
        
        # Mask and renormalize policy
        masked_policy = policy_probs * legal_mask
        if masked_policy.sum() > 0:
            masked_policy = masked_policy / masked_policy.sum()
        else:
            # Fallback to uniform distribution
            masked_policy = legal_mask / legal_mask.sum()
        
        # Create child nodes for all legal moves
        for action in legal_moves:
            # Create a copy of the game state and make the move
            child_game_state = deepcopy(node.game_state)
            child_game_state.make_move(action)
            
            # Create child node with prior probability from policy network
            child_node = MCTSNode(
                game_state=child_game_state,
                parent=node,
                prior_prob=masked_policy[action],
                action_taken=action
            )
            node.children[action] = child_node
        
        return value
    
    def _backpropagate(self, search_path, value):
        """
        Update all nodes in the search path with the value.
        Value is alternated because players alternate.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
    
    def _get_action_probs(self, root, temperature=1.0):
        """
        Get action probabilities based on visit counts.
        Temperature controls exploration:
        - temperature → 0: pick best move deterministically
        - temperature = 1: sample proportional to visit counts
        - temperature > 1: more uniform sampling
        """
        legal_moves = root.game_state.get_legal_moves()
        action_probs = np.zeros(9)
        
        if len(root.children) == 0:
            # No children (shouldn't happen in normal operation)
            action_probs[legal_moves] = 1.0 / len(legal_moves)
            return action_probs
        
        # Get visit counts for each action
        visits = np.zeros(9)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        
        if temperature == 0:
            # Deterministic: pick most visited
            best_action = visits.argmax()
            action_probs[best_action] = 1.0
        else:
            # Apply temperature and normalize
            visits_temp = visits ** (1.0 / temperature)
            action_probs = visits_temp / visits_temp.sum()
        
        return action_probs
