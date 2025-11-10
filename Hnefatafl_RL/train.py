"""
Self-play training loop for Tafl-style games using AlphaZero.

This is a generic training module that can be used with different board sizes
and game variants. For Brandubh (7x7), use train_brandubh.py.

Implements the complete AlphaZero training pipeline:
- Self-play game generation with MCTS
- Replay buffer with experience storage
- Neural network training
- Model evaluation and checkpointing
- Data augmentation (symmetries)

To use with a different game:
1. Create a game class similar to Brandubh with the same interface
2. Create a network class that takes board_size as a parameter
3. Create a training script that configures TrainingConfig and calls train()
"""

import os
import sys
import time
import json
from collections import deque
from typing import List, Tuple, Dict
import numpy as np
import multiprocessing as mp
from functools import partial

try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is required for training. Install with: pip install torch")
    exit(1)

from brandubh import Brandubh
from network import BrandubhNet, MoveEncoder
from agent import Agent


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

class TrainingConfig:
    """Configuration for training."""
    
    # Game and architecture configuration (set these when initializing)
    game_class = None                 # Game class (e.g., Brandubh)
    network_class = None              # Network class (e.g., BrandubhNet)
    agent_class = None                # Agent class (e.g., BrandubhAgent)
    move_encoder_class = None         # Move encoder class (e.g., MoveEncoder)
    board_size = 7                    # Board size (e.g., 7 for Brandubh)
    
    # Computed properties (derived from board_size)
    @property
    def num_squares(self):
        return self.board_size * self.board_size
    
    @property
    def policy_size(self):
        """Policy size: num_squares * 4 directions * (board_size-1) max distance"""
        return self.num_squares * 4 * (self.board_size - 1)
    
    # Game rules (tunable)
    king_capture_pieces = 2           # Pieces required to capture king (2, 3, or 4)
    king_can_capture = True           # Whether king can help capture enemy pieces
    throne_is_hostile = False         # Whether throne counts as hostile square for captures
    throne_enabled = True             # Whether throne exists and blocks non-king movement
    
    # Self-play
    num_iterations = 100              # Number of training iterations
    num_games_per_iteration = 100     # Self-play games per iteration
    num_mcts_simulations = 100        # MCTS simulations per move (deprecated, use attacker/defender specific)
    num_mcts_sims_attacker = 100      # MCTS simulations for attacker in self-play
    num_mcts_sims_defender = 100      # MCTS simulations for defender in self-play
    c_puct = 1.4                      # MCTS exploration constant
    num_workers = mp.cpu_count()      # Number of parallel workers (default: all CPUs)
    
    # Temperature (exploration during self-play)
    temperature = 1.0                 # Sampling temperature for moves
    temperature_threshold = 15        # Move number after which temperature = 0, or "king" to drop when king leaves throne
    
    # Neural network
    num_res_blocks = 4                # Residual blocks in network
    num_channels = 64                 # Channels in convolutional layers
    
    # Training
    batch_size = 32                   # Training batch size
    num_epochs = 10                   # Epochs per iteration
    batches_per_epoch = 100           # Number of batches to sample per epoch (limits training data)
    learning_rate = 0.001             # Initial learning rate
    lr_decay = 0.95                   # Learning rate decay per iteration
    weight_decay = 1e-4               # L2 regularization
    value_loss_weight = 10.0           # Weight for value loss (policy loss weight is always 1.0)
    
    # Dynamic loss boosting (set use_dynamic_boosting=False to use static boost)
    use_dynamic_boosting = True       # Use dynamic loss boosting based on win rates
    dynamic_boost_alpha = 0.1         # Smoothing factor for win rate tracking (0-1, higher = more reactive)
    dynamic_boost_min = 0.5           # Minimum boost factor
    dynamic_boost_max = 3.0           # Maximum boost factor
    attacker_win_loss_boost = 1.0     # Static boost (only used if use_dynamic_boosting=False)
    
    draw_penalty_attacker = 0.0       # Draw value for attackers (neutral)
    draw_penalty_defender = 0.0       # Draw value for defenders (neutral)
    
    # Replay buffer
    replay_buffer_size = 50000        # Maximum samples in replay buffer
    min_buffer_size = 1000            # Minimum samples before training
    use_data_augmentation = True      # Use symmetry-based data augmentation (8x data)
    
    # Evaluation
    eval_games = 20                   # Games for evaluation
    eval_win_rate = 0.55              # Win rate threshold to replace best model
    eval_frequency = 5                # Evaluate every N iterations
    eval_mcts_sims_attacker = 100     # MCTS simulations for attacker in evaluation
    eval_mcts_sims_defender = 100     # MCTS simulations for defender in evaluation
    
    # Random baseline evaluation
    eval_vs_random_games = 10         # Games vs random (per color)
    eval_vs_random_frequency = 1      # Evaluate vs random every N iterations
    
    # Checkpointing
    checkpoint_dir = "checkpoints"    # Directory for model checkpoints
    save_frequency = 5                # Save model every N iterations
    
    # Device
    device = "cpu"  # Force CPU for multiprocessing compatibility
    
    # Logging
    log_frequency = 1                 # Log every N iterations
    verbose = True                    # Print detailed logs


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """
    Stores experience from self-play games.
    Each sample is (state, policy, value, attacker_won).
    The attacker_won flag is used to boost losses from attacker-won games.
    """
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float, attacker_won: bool = False):
        """Add a sample to the buffer."""
        self.buffer.append((state, policy, value, attacker_won))
    
    def add_game(self, states: List[np.ndarray], policies: List[np.ndarray], 
                 winner: int, player_perspectives: List[int]):
        """
        Add a complete game to the buffer.
        
        Args:
            states: list of game states
            policies: list of policy distributions
            winner: 0 for attackers, 1 for defenders, None for draw
            player_perspectives: list of which player made each move
        """
        attacker_won = (winner == 0)
        for state, policy, player in zip(states, policies, player_perspectives):
            # Value from perspective of player who made the move
            if winner is None:
                value = 0.0  # Draw
            elif winner == player:
                value = 1.0
            elif winner == 1 - player:
                value = -1.0
            else:
                value = 0.0  # Shouldn't happen, but safe default
            
            self.add(state, policy, value, attacker_won)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch from the buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        policies = []
        values = []
        attacker_won_flags = []
        
        for idx in indices:
            state, policy, value, attacker_won = self.buffer[idx]
            states.append(state)
            policies.append(policy)
            values.append(value)
            attacker_won_flags.append(attacker_won)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32),
            np.array(attacker_won_flags, dtype=bool)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def get_win_statistics(self) -> Dict[str, int]:
        """
        Get win statistics from samples in the buffer.
        
        Returns:
            dict with 'attacker_wins', 'defender_wins', 'total_samples'
        """
        attacker_wins = 0
        defender_wins = 0
        
        for _, _, _, attacker_won in self.buffer:
            if attacker_won:
                attacker_wins += 1
            else:
                defender_wins += 1
        
        return {
            'attacker_wins': attacker_wins,
            'defender_wins': defender_wins,
            'total_samples': len(self.buffer)
        }


class WinRateTracker:
    """
    Tracks win rates over time to compute dynamic loss boosting.
    Uses exponential moving average for smoothing.
    """
    
    def __init__(self, alpha: float = 0.1, min_boost: float = 0.5, max_boost: float = 3.0):
        """
        Args:
            alpha: Smoothing factor (0-1, higher = more weight to recent games)
            min_boost: Minimum boost factor to prevent extreme values
            max_boost: Maximum boost factor to prevent extreme values
        """
        self.alpha = alpha
        self.min_boost = min_boost
        self.max_boost = max_boost
        
        # Smoothed win counts
        self.attacker_wins_smooth = 1.0  # Start with pseudocounts to avoid division by zero
        self.defender_wins_smooth = 1.0
        
        # Total games tracked
        self.total_games = 0
    
    def update(self, attacker_wins: int, defender_wins: int):
        """
        Update win rate tracking with new game results.
        
        Args:
            attacker_wins: Number of attacker wins in this batch
            defender_wins: Number of defender wins in this batch
        """
        total_new_games = attacker_wins + defender_wins
        
        if total_new_games == 0:
            return  # No games to update with
        
        # Update with exponential moving average
        self.attacker_wins_smooth = (1 - self.alpha) * self.attacker_wins_smooth + self.alpha * attacker_wins
        self.defender_wins_smooth = (1 - self.alpha) * self.defender_wins_smooth + self.alpha * defender_wins
        self.total_games += total_new_games
    
    def get_boost_factors(self) -> Tuple[float, float]:
        """
        Compute boost factors for attacker and defender losses.
        
        The boost is inversely proportional to win rate:
        - If attackers win less, boost their loss more
        - If defenders win less, boost their loss more
        
        Returns:
            (attacker_boost, defender_boost) tuple
        """
        total_wins = self.attacker_wins_smooth + self.defender_wins_smooth
        
        if total_wins == 0:
            return 1.0, 1.0  # No data yet, equal boost
        
        # Win rates
        attacker_rate = self.attacker_wins_smooth / total_wins
        defender_rate = self.defender_wins_smooth / total_wins
        
        # Boost is inversely proportional to win rate
        # If attacker wins 10% of games, they get more boost than defenders
        # Clamp to reasonable range to prevent extreme values
        eps = 1e-6
        safe_attacker_rate = max(attacker_rate, eps)
        safe_defender_rate = max(defender_rate, eps)
        ratio = safe_defender_rate / safe_attacker_rate
        ratio = np.clip(ratio, self.min_boost, self.max_boost)

        # We apply the square root so the effective ratio after normalization
        # matches the intended ratio instead of being squared.
        attacker_boost = np.sqrt(ratio)
        defender_boost = 1.0 / attacker_boost
        
        return attacker_boost, defender_boost
    
    def get_win_rates(self) -> Tuple[float, float]:
        """
        Get current smoothed win rates.
        
        Returns:
            (attacker_rate, defender_rate) tuple
        """
        total = self.attacker_wins_smooth + self.defender_wins_smooth
        if total == 0:
            return 0.5, 0.5
        return self.attacker_wins_smooth / total, self.defender_wins_smooth / total


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def augment_sample(state: np.ndarray, policy: np.ndarray, value: float, board_size: int = 7) -> List[Tuple]:
    """
    Generate all 8 symmetric transformations of a sample.
    Returns list of (state, policy, value) tuples.
    
    The board has 4-fold rotational symmetry and 2-fold reflective symmetry.
    This gives us 8 unique transformations total.
    
    Args:
        state: board state array
        policy: policy vector
        value: value scalar
        board_size: size of the board (default 7 for Brandubh)
    """
    augmented = []
    num_squares = board_size * board_size
    max_distance = board_size - 1
    num_directions = 4
    policy_moves_per_square = num_directions * max_distance
    
    # Helper to rotate a move in policy space
    def rotate_policy(policy_vec, k):
        """Rotate policy vector k times 90 degrees clockwise."""
        policy_arr = policy_vec.reshape(num_squares, policy_moves_per_square)  # (squares, directions*distances)
        
        # Rotate board positions
        rotated = np.zeros_like(policy_arr)
        for old_sq in range(num_squares):
            old_r, old_c = old_sq // board_size, old_sq % board_size
            # Rotate position
            for _ in range(k):
                old_r, old_c = old_c, (board_size - 1) - old_r
            new_sq = old_r * board_size + old_c
            
            # Rotate directions (up->right->down->left)
            for old_dir in range(num_directions):
                new_dir = (old_dir + k) % num_directions
                for dist in range(max_distance):
                    old_idx = old_dir * max_distance + dist
                    new_idx = new_dir * max_distance + dist
                    rotated[new_sq, new_idx] = policy_arr[old_sq, old_idx]
        
        return rotated.reshape(-1)
    
    def flip_policy(policy_vec, horizontal=True):
        """Flip policy vector horizontally or vertically."""
        policy_arr = policy_vec.reshape(num_squares, policy_moves_per_square)
        flipped = np.zeros_like(policy_arr)
        
        for old_sq in range(num_squares):
            old_r, old_c = old_sq // board_size, old_sq % board_size
            # Flip position
            if horizontal:
                new_r, new_c = old_r, (board_size - 1) - old_c
                dir_map = {0: 0, 1: 1, 2: 3, 3: 2}  # Swap left/right
            else:
                new_r, new_c = (board_size - 1) - old_r, old_c
                dir_map = {0: 1, 1: 0, 2: 2, 3: 3}  # Swap up/down
            
            new_sq = new_r * board_size + new_c
            
            for old_dir in range(num_directions):
                new_dir = dir_map[old_dir]
                for dist in range(max_distance):
                    old_idx = old_dir * max_distance + dist
                    new_idx = new_dir * max_distance + dist
                    flipped[new_sq, new_idx] = policy_arr[old_sq, old_idx]
        
        return flipped.reshape(-1)
    
    # Original
    augmented.append((state.copy(), policy.copy(), value))
    
    # 4 rotations
    for k in range(1, 4):
        rotated_state = np.rot90(state, k=k, axes=(1, 2)).copy()
        rotated_policy = rotate_policy(policy, k)
        augmented.append((rotated_state, rotated_policy, value))
    
    # Horizontal flip + 4 rotations
    flipped_state = np.flip(state, axis=2).copy()
    flipped_policy = flip_policy(policy, horizontal=True)
    augmented.append((flipped_state, flipped_policy, value))
    
    for k in range(1, 4):
        rotated_state = np.rot90(flipped_state, k=k, axes=(1, 2)).copy()
        rotated_policy = rotate_policy(flipped_policy, k)
        augmented.append((rotated_state, rotated_policy, value))
    
    return augmented


# =============================================================================
# SELF-PLAY
# =============================================================================

def _play_self_play_game_worker(network_path, num_res_blocks, num_channels, 
                                num_sims_attacker, num_sims_defender, c_puct, temperature, temperature_threshold, game_idx,
                                king_capture_pieces, king_can_capture, throne_is_hostile, throne_enabled, board_size,
                                network_module, network_class_name, game_module, game_class_name):
    """
    Worker function for parallel self-play game generation.
    Must be at module level for multiprocessing. Imports torch inside to avoid pickling issues.
    
    Args:
        network_path: Path to saved network weights file
        num_res_blocks: Number of residual blocks in network
        num_channels: Number of channels in network
        num_sims_attacker: MCTS simulations per move for attacker
        num_sims_defender: MCTS simulations per move for defender
        c_puct: MCTS exploration constant
        temperature: Sampling temperature
        temperature_threshold: Move threshold or "king" for king-based threshold
        game_idx: Game index for seeding
        king_capture_pieces: Number of pieces to capture king
        king_can_capture: Whether king can capture
        throne_is_hostile: Whether throne is hostile for captures
        network_module: Module name for network (e.g., 'network', 'network_tablut')
        network_class_name: Network class name (e.g., 'BrandubhNet', 'TablutNet')
        game_module: Module name for game (e.g., 'brandubh', 'tablut')
        game_class_name: Game class name (e.g., 'Brandubh', 'Tablut')
        throne_enabled: Whether throne exists and blocks movement
        temperature_threshold: Move number threshold for temperature
        game_idx: Game index (unused, for pool.map)
        king_capture_pieces: Number of pieces required to capture king (2, 3, or 4)
        king_can_capture: Whether king can help capture enemy pieces
        throne_is_hostile: Whether throne counts as hostile square
    
    Returns:
        dict with game data and MCTS timing information
    """
    try:
        # Seed random number generators uniquely for each worker
        # Use game_idx combined with process ID and time to ensure uniqueness
        import time
        seed = (game_idx + os.getpid() + int(time.time() * 1000)) % (2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Set torch to use only 1 thread per worker to avoid conflicts
        torch.set_num_threads(1)
        
        # Set per-worker compilation cache to avoid lock contention
        # Each worker gets its own cache directory, so no delays needed
        import tempfile
        worker_cache_dir = os.path.join(tempfile.gettempdir(), f'torchinductor_{os.getpid()}')
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = worker_cache_dir
        
        # Import dynamically based on config
        import importlib
        game_mod = importlib.import_module(game_module)
        GameClass = getattr(game_mod, game_class_name)
        
        network_mod = importlib.import_module(network_module)
        NetworkClass = getattr(network_mod, network_class_name)
        
        # Import MoveEncoder from the same network module
        # For Brandubh: MoveEncoder from network.py
        # For Tablut: TablutMoveEncoder from network_tablut.py
        # For Hnefatafl: HnefataflMoveEncoder from network_hnefatafl.py
        if 'hnefatafl' in network_module.lower():
            MoveEncoderClass = getattr(network_mod, 'HnefataflMoveEncoder')
        elif 'tablut' in network_module.lower():
            MoveEncoderClass = getattr(network_mod, 'TablutMoveEncoder')
        else:
            MoveEncoderClass = getattr(network_mod, 'MoveEncoder')
        
        from mcts import MCTS
        
        # Reconstruct network on CPU and load from file
        network = NetworkClass(num_res_blocks=num_res_blocks, num_channels=num_channels)
        checkpoint = torch.load(network_path, map_location='cpu', weights_only=False)
        network.load_state_dict(checkpoint['model_state_dict'])
        network.to('cpu')
        network.eval()
        
        # Optimize for CPU inference with compilation for 2.5-2.75x speedup
        # Each worker has its own cache directory, so no contention
        network = network.optimize_for_inference(use_compile=True, compile_mode='default')
        
        # Create MCTS instances for each player (with different simulation counts)
        # Pass MoveEncoderClass to ensure correct encoder is used (especially after torch.compile)
        mcts_attacker = MCTS(network, num_simulations=num_sims_attacker, c_puct=c_puct, device='cpu', 
                            move_encoder_class=MoveEncoderClass)
        mcts_defender = MCTS(network, num_simulations=num_sims_defender, c_puct=c_puct, device='cpu',
                            move_encoder_class=MoveEncoderClass)
        mcts_attacker.reset_timing_stats()
        mcts_defender.reset_timing_stats()
        
        # Play game with configured rules
        game = GameClass(
            king_capture_pieces=king_capture_pieces,
            king_can_capture=king_can_capture,
            throne_is_hostile=throne_is_hostile,
            throne_enabled=throne_enabled
        )
        states = []
        policies = []
        players = []
        move_count = 0
        
        while not game.game_over:
            current_player = game.current_player
            
            # Select MCTS based on current player
            mcts = mcts_attacker if current_player == 0 else mcts_defender
            
            # Determine temperature based on move count or king position
            if temperature_threshold == "king":
                # Drop temperature when king is off the throne
                # KING piece value is 3 in all game variants
                king_on_throne = (game.board[game.throne] == 3)
                temp = temperature if king_on_throne else 0.0
            else:
                # Drop temperature after a fixed number of moves
                temp = temperature if move_count < temperature_threshold else 0.0
            
            # Get state
            state = game.get_state()
            
            # Run MCTS to get policy
            visit_probs = mcts.search(game)
            
            # Convert to policy vector
            policy_size = board_size * board_size * 4 * (board_size - 1)
            policy = np.zeros(policy_size, dtype=np.float32)
            for move, prob in visit_probs.items():
                idx = MoveEncoderClass.encode_move(move)
                policy[idx] = prob
            
            # Store experience
            states.append(state)
            policies.append(policy)
            players.append(current_player)
            
            # Select and make move
            moves = list(visit_probs.keys())
            probs = np.array(list(visit_probs.values()))
            
            if temp == 0:
                move = moves[np.argmax(probs)]
            else:
                # Apply temperature
                probs = probs ** (1.0 / temp)
                probs = probs / probs.sum()
                move_idx = np.random.choice(len(moves), p=probs)
                move = moves[move_idx]
            
            game.make_move(move)
            move_count += 1
            
            # Safety check for move limit
            if move_count > 500:
                break
        
        # Determine draw reason (only move limit can cause draws - repetitions are illegal moves)
        draw_reason = None
        if not game.game_over and move_count > 500:
            # Hit move limit without natural game end
            draw_reason = 'move_limit'
        
        # Combine timing stats from both MCTS instances
        timing_attacker = mcts_attacker.get_timing_stats()
        timing_defender = mcts_defender.get_timing_stats()
        
        # Build combined timing with explicit float conversion to ensure picklability
        combined_timing = {
            key: float(timing_attacker[key] + timing_defender.get(key, 0.0))
            for key in timing_attacker.keys()
        }
        
        # Extract game results before cleanup
        winner = int(game.winner) if game.winner is not None else None
        num_moves = int(move_count)
        draw_reason_clean = str(draw_reason) if draw_reason else None
        
        # Clean up references to prevent pickle errors
        # Convert states/policies to ensure they're picklable numpy arrays without references
        states_clean = [np.array(s, dtype=np.float32).copy() for s in states]
        policies_clean = [np.array(p, dtype=np.float32).copy() for p in policies]
        players_clean = [int(p) for p in players]
        
        # Delete ALL objects that might hold references to compiled network
        del mcts_attacker
        del mcts_defender
        del network
        del game  # Delete game object too
        del states
        del policies
        del players
        del timing_attacker
        del timing_defender
        
        # Force garbage collection to clean up any circular references
        import gc
        gc.collect()
        
        return {
            'states': states_clean,
            'policies': policies_clean,
            'winner': winner,
            'players': players_clean,
            'num_moves': num_moves,
            'draw_reason': draw_reason_clean,
            'timing': combined_timing
        }
    except Exception as e:
        # Catch any exception and re-raise as a simple picklable RuntimeError
        # This prevents "cannot pickle 'frame' object" errors in multiprocessing
        import traceback
        error_msg = f"Error in worker {game_idx} (PID {os.getpid()}):\n"
        error_msg += "".join(traceback.format_exception(type(e), e, e.__traceback__))
        # Print to stderr immediately so user sees it even if pickling fails
        import sys
        print(error_msg, file=sys.stderr, flush=True)
        # Raise simple picklable exception
        raise RuntimeError(error_msg)


def play_self_play_game(agent: Agent, config: TrainingConfig) -> Dict:
    """
    Play a single self-play game and collect training data.
    
    Returns:
        dict with 'states', 'policies', 'winner', 'players', 'num_moves'
    """
    game = Brandubh(
        king_capture_pieces=config.king_capture_pieces,
        king_can_capture=config.king_can_capture,
        throne_is_hostile=config.throne_is_hostile,
        throne_enabled=config.throne_enabled
    )
    states = []
    policies = []
    players = []
    
    move_count = 0
    
    while not game.game_over:
        current_player = game.current_player
        
        # Determine temperature based on move count or king position
        if config.temperature_threshold == "king":
            # Drop temperature when king is off the throne
            # KING piece value is 3 in all game variants
            king_on_throne = (game.board[game.throne] == 3)
            temperature = config.temperature if king_on_throne else 0.0
        elif move_count < config.temperature_threshold:
            temperature = config.temperature
        else:
            temperature = 0.0  # Deterministic after threshold
        
        # Get state
        state = game.get_state()
        
        # Run MCTS to get policy
        visit_probs = agent.mcts.search(game)
        
        # Convert to policy vector
        policy = np.zeros(config.policy_size, dtype=np.float32)
        for move, prob in visit_probs.items():
            idx = MoveEncoder.encode_move(move)
            policy[idx] = prob
        
        # Store experience
        states.append(state)
        policies.append(policy)
        players.append(current_player)
        
        # Select and make move
        moves = list(visit_probs.keys())
        probs = np.array(list(visit_probs.values()))
        
        if temperature == 0:
            move = moves[np.argmax(probs)]
        else:
            # Apply temperature
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
            move_idx = np.random.choice(len(moves), p=probs)
            move = moves[move_idx]
        
        game.make_move(move)
        move_count += 1
        
        # Safety check for move limit
        if move_count > 500:
            break
    
    # Determine draw reason (only move limit can cause draws - repetitions are illegal moves)
    draw_reason = None
    if not game.game_over and move_count > 500:
        # Hit move limit without natural game end
        draw_reason = 'move_limit'
    
    return {
        'states': states,
        'policies': policies,
        'winner': game.winner,  # Can be 0, 1, or None (only if move limit reached)
        'players': players,
        'num_moves': move_count,
        'draw_reason': draw_reason  # 'move_limit' or None
    }


def generate_self_play_data(agent: Agent, config: TrainingConfig, pool=None, temp_network_path=None) -> Tuple[ReplayBuffer, int, int]:
    """
    Generate self-play games and store in replay buffer.
    Uses multiprocessing to parallelize game generation.
    
    Args:
        agent: Agent to use for self-play
        config: Training configuration
        pool: Optional multiprocessing pool to use (if None, creates temporary pool)
        temp_network_path: Path to temporary network file (avoids pickling large tensors)
    
    Returns:
        Tuple of (buffer, attacker_wins, defender_wins) where:
        - buffer: ReplayBuffer with collected experience
        - attacker_wins: Number of games won by attacker
        - defender_wins: Number of games won by defender
    """
    buffer = ReplayBuffer(config.replay_buffer_size)
    
    print(f"Generating {config.num_games_per_iteration} self-play games using {config.num_workers} workers...")
    
    # Save network to temporary file to avoid pickling issues with many workers
    import tempfile
    if temp_network_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pth', delete=False)
        temp_network_path = temp_file.name
        temp_file.close()
        # Save as checkpoint format (workers expect 'model_state_dict' key)
        # Strip _orig_mod. prefix if network is compiled (torch.compile adds this)
        checkpoint = {'model_state_dict': clean_state_dict(agent.network.state_dict())}
        torch.save(checkpoint, temp_network_path)
        cleanup_temp_file = True
    else:
        cleanup_temp_file = False
    
    try:
        # Create a list of argument tuples for each worker call
        # Each tuple contains all arguments needed for _play_self_play_game_worker
        worker_args = [
            (temp_network_path, config.num_res_blocks, config.num_channels,
             config.num_mcts_sims_attacker, config.num_mcts_sims_defender,
             config.c_puct, config.temperature, config.temperature_threshold,
             i, config.king_capture_pieces, config.king_can_capture,
             config.throne_is_hostile, config.throne_enabled, config.board_size,
             config.network_class.__module__, config.network_class.__name__,
             config.game_class.__module__, config.game_class.__name__)
            for i in range(config.num_games_per_iteration)
        ]
        
        # Play games in parallel
        if config.num_workers > 1:
            if pool is not None:
                # Use provided persistent pool with starmap
                try:
                    game_results = list(pool.starmap(_play_self_play_game_worker, worker_args, chunksize=1))
                except Exception as e:
                    # Re-raise to ensure proper error propagation
                    print(f"\nError during self-play: {e}", file=sys.stderr, flush=True)
                    raise
            else:
                # Create temporary pool (for backward compatibility)
                with mp.Pool(processes=config.num_workers, maxtasksperchild=10) as temp_pool:
                    try:
                        game_results = list(temp_pool.starmap(_play_self_play_game_worker, worker_args, chunksize=1))
                    except Exception as e:
                        print(f"\nError during self-play: {e}", file=sys.stderr, flush=True)
                        raise
        else:
            # Single-threaded fallback
            game_results = [_play_self_play_game_worker(*args) for args in worker_args]
    finally:
        # Clean up temporary file
        if cleanup_temp_file:
            try:
                os.unlink(temp_network_path)
            except Exception:
                pass
    
    # Process results
    total_moves = 0
    attacker_wins = 0
    defender_wins = 0
    draws = 0
    move_limit_draws = 0
    
    # MCTS timing aggregation
    mcts_timing_totals = {
        'selection': 0.0,
        'terminal_eval': 0.0,
        'network_eval': 0.0,
        'expansion': 0.0,
        'backup': 0.0,
        'game_clone': 0.0,
        'get_legal_moves': 0.0
    }
    
    for game_data in game_results:
        total_moves += game_data['num_moves']
        winner = game_data['winner']
        draw_reason = game_data.get('draw_reason', None)
        
        if winner == 0:
            attacker_wins += 1
        elif winner == 1:
            defender_wins += 1
        else:  # winner is None (only possible if move limit reached)
            draws += 1
            if draw_reason == 'move_limit':
                move_limit_draws += 1
        
        # Aggregate MCTS timing data
        if 'timing' in game_data:
            for key, value in game_data['timing'].items():
                if key in mcts_timing_totals:
                    mcts_timing_totals[key] += value
        
        # Add to buffer
        for state, policy, player in zip(game_data['states'], 
                                         game_data['policies'], 
                                         game_data['players']):
            # Determine value from player's perspective
            if winner is None or (not game_data.get('winner') and draw_reason == 'move_limit'):
                # Draw (only by move limit - repetitions are illegal moves)
                # Apply player-specific draw penalty
                if player == 0:  # Attacker
                    value = config.draw_penalty_attacker
                else:  # Defender (player == 1)
                    value = config.draw_penalty_defender
            elif winner == player:
                value = 1.0
            elif winner == 1 - player:
                value = -1.0
            else:
                value = 0.0  # Shouldn't happen
            
            # Mark if attacker won this game
            attacker_won = (winner == 0)
            
            # Add original sample
            buffer.add(state, policy, value, attacker_won)
            
            # Add augmented samples using board symmetries
            if config.use_data_augmentation:
                augmented = augment_sample(state, policy, value, config.board_size)
                for aug_state, aug_policy, aug_value in augmented:
                    buffer.add(aug_state, aug_policy, aug_value, attacker_won)
    
    print(f"Generated {len(buffer)} training samples")
    total_games = config.num_games_per_iteration
    print(f"Results: {attacker_wins} attacker wins ({100*attacker_wins/total_games:.1f}%), "
          f"{defender_wins} defender wins ({100*defender_wins/total_games:.1f}%), "
          f"{draws} draws ({100*draws/total_games:.1f}%)")
    if draws > 0:
        print(f"  All draws by move limit (500+ moves): {move_limit_draws}")
    print(f"Average game length: {total_moves/total_games:.1f} moves")
    
    # Print MCTS timing breakdown
    total_mcts_time = sum(mcts_timing_totals.values())
    if total_mcts_time > 0:
        print("\nMCTS timing breakdown:")
        print(f"  Selection:         {mcts_timing_totals['selection']:7.1f}s ({100*mcts_timing_totals['selection']/total_mcts_time:5.1f}%)")
        print(f"  Network eval:      {mcts_timing_totals['network_eval']:7.1f}s ({100*mcts_timing_totals['network_eval']/total_mcts_time:5.1f}%)")
        print(f"  Expansion:         {mcts_timing_totals['expansion']:7.1f}s ({100*mcts_timing_totals['expansion']/total_mcts_time:5.1f}%)")
        print(f"  Backup:            {mcts_timing_totals['backup']:7.1f}s ({100*mcts_timing_totals['backup']/total_mcts_time:5.1f}%)")
        print(f"  Game clone:        {mcts_timing_totals['game_clone']:7.1f}s ({100*mcts_timing_totals['game_clone']/total_mcts_time:5.1f}%)")
        print(f"  Get legal moves:   {mcts_timing_totals['get_legal_moves']:7.1f}s ({100*mcts_timing_totals['get_legal_moves']/total_mcts_time:5.1f}%)")
        print(f"  Terminal eval:     {mcts_timing_totals['terminal_eval']:7.1f}s ({100*mcts_timing_totals['terminal_eval']/total_mcts_time:5.1f}%)")
        print(f"  Total MCTS time:   {total_mcts_time:7.1f}s")
    
    return buffer, attacker_wins, defender_wins


# =============================================================================
# TRAINING
# =============================================================================

def train_network(network: BrandubhNet, buffer: ReplayBuffer, 
                  optimizer: optim.Optimizer, config: TrainingConfig, 
                  attacker_boost: float = 1.0, defender_boost: float = 1.0) -> Dict[str, float]:
    """
    Train the neural network on samples from the replay buffer.
    
    Note: This function compiles the network with torch.compile() for faster
    training (forward + backward passes). The compiled version is only used
    within this function and doesn't affect the original network object.
    
    Args:
        network: Neural network to train
        buffer: Replay buffer with training samples
        optimizer: Optimizer for network parameters
        config: Training configuration
        attacker_boost: Loss boost factor for attacker-won games
        defender_boost: Loss boost factor for defender-won games
    
    Returns:
        dict with 'policy_loss', 'value_loss', 'total_loss'
    """
    # Compile network for faster training (~18% speedup)
    # This creates a compiled wrapper that's only used during training
    network_compiled = torch.compile(network, mode='default')
    network_compiled.train()
    device = config.device
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    samples_per_epoch = min(len(buffer), config.batch_size * config.batches_per_epoch)
    batches_per_epoch = samples_per_epoch // config.batch_size
    
    for epoch in range(config.num_epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        
        for batch_idx in range(batches_per_epoch):
            # Sample batch
            states, policies, values, attacker_won_flags = buffer.sample(config.batch_size)
            
            # Convert to tensors
            states = torch.from_numpy(states).to(device)
            policies = torch.from_numpy(policies).to(device)
            values = torch.from_numpy(values).unsqueeze(1).to(device)
            attacker_won_flags = torch.from_numpy(attacker_won_flags).to(device)
            
            # Forward pass (using compiled network for speed)
            pred_policies, pred_values = network_compiled(states)
            
            # Compute per-sample losses
            policy_loss_per_sample = -torch.sum(policies * 
                                      torch.log_softmax(pred_policies, dim=1), dim=1)
            value_loss_per_sample = (pred_values.squeeze(1) - values.squeeze(1)) ** 2
            
            # Apply dynamic boost factors based on which side won
            # Attacker-won games get attacker_boost, defender-won games get defender_boost
            if attacker_boost != 1.0 or defender_boost != 1.0:
                boost_weights = torch.where(attacker_won_flags, 
                                           attacker_boost, 
                                           defender_boost)
                
                # Normalize by mean weight to keep effective learning rate stable
                # This prevents gradient explosion when boost factors are very different
                mean_weight = torch.mean(boost_weights)
                normalized_weights = boost_weights / mean_weight
                
                policy_loss_per_sample = policy_loss_per_sample * normalized_weights
                value_loss_per_sample = value_loss_per_sample * normalized_weights
            
            # Aggregate losses
            policy_loss = torch.mean(policy_loss_per_sample)
            value_loss = torch.mean(value_loss_per_sample)
            loss = policy_loss + config.value_loss_weight * value_loss
            
            # Backward pass (set_to_none=True is faster than default zero_grad)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            batch_policy_loss = policy_loss.item()
            batch_value_loss = value_loss.item()
            batch_total_loss = loss.item()
            
            total_policy_loss += batch_policy_loss
            total_value_loss += batch_value_loss
            total_loss += batch_total_loss
            
            epoch_policy_loss += batch_policy_loss
            epoch_value_loss += batch_value_loss
            epoch_total_loss += batch_total_loss
            
            num_batches += 1
        
        # Print per-epoch losses
        avg_epoch_policy = epoch_policy_loss / batches_per_epoch
        avg_epoch_value = epoch_value_loss / batches_per_epoch
        avg_epoch_total = epoch_total_loss / batches_per_epoch
        print(f"  Epoch {epoch+1}/{config.num_epochs}: "
              f"policy={avg_epoch_policy:.4f}, value={avg_epoch_value:.4f}, "
              f"total={avg_epoch_total:.4f}")
    
    return {
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
        'total_loss': total_loss / num_batches
    }


# =============================================================================
# EVALUATION
# =============================================================================

def _evaluate_vs_random_worker(network_path, num_res_blocks, num_channels,
                                num_sims_attacker, num_sims_defender, c_puct, nn_plays_attacker, game_idx,
                                network_module, network_class_name, game_module, game_class_name):
    """
    Worker function for parallel evaluation against random player.
    Must be at module level for multiprocessing. Imports inside to avoid pickling issues.
    
    Args:
        network_path: Path to saved network weights file
        num_res_blocks: Number of residual blocks
        num_channels: Number of channels
        num_sims_attacker: MCTS simulations per move for attacker
        num_sims_defender: MCTS simulations per move for defender
        c_puct: MCTS exploration constant
        nn_plays_attacker: Whether NN plays as attacker
        game_idx: Game index (unused, for pool.map)
        network_module: Module name for network (e.g., 'network', 'network_tablut')
        network_class_name: Network class name (e.g., 'BrandubhNet', 'TablutNet')
        game_module: Module name for game (e.g., 'brandubh', 'tablut')
        game_class_name: Game class name (e.g., 'Brandubh', 'Tablut')
    
    Returns:
        1 if NN wins, 0 otherwise
    """
    # Seed random number generators uniquely for each worker
    import time
    seed = (game_idx + os.getpid() + int(time.time() * 1000)) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set torch to use only 1 thread per worker to avoid conflicts
    torch.set_num_threads(1)
    
    # Set per-worker compilation cache to avoid lock contention
    # Each worker gets its own cache directory, so no delays needed
    import tempfile
    worker_cache_dir = os.path.join(tempfile.gettempdir(), f'torchinductor_{os.getpid()}')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = worker_cache_dir
    
    # Import dynamically based on config
    import importlib
    game_mod = importlib.import_module(game_module)
    GameClass = getattr(game_mod, game_class_name)
    
    network_mod = importlib.import_module(network_module)
    NetworkClass = getattr(network_mod, network_class_name)
    
    # Import MoveEncoder from the same network module
    # For Brandubh: MoveEncoder from network.py
    # For Tablut: TablutMoveEncoder from network_tablut.py
    # For Hnefatafl: HnefataflMoveEncoder from network_hnefatafl.py
    if 'hnefatafl' in network_module.lower():
        MoveEncoderClass = getattr(network_mod, 'HnefataflMoveEncoder')
    elif 'tablut' in network_module.lower():
        MoveEncoderClass = getattr(network_mod, 'TablutMoveEncoder')
    else:
        MoveEncoderClass = getattr(network_mod, 'MoveEncoder')
    
    from agent import Agent, RandomAgent, play_game
    
    # Reconstruct network on CPU and load from file
    network = NetworkClass(num_res_blocks=num_res_blocks, num_channels=num_channels)
    checkpoint = torch.load(network_path, map_location='cpu', weights_only=False)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.to('cpu')
    network.eval()
    
    # Optimize for CPU inference with compilation for 2.5-2.75x speedup
    # Each worker has its own cache directory, so no contention
    network = network.optimize_for_inference(use_compile=True, compile_mode='default')
    
    # Determine which simulations to use based on NN's role
    num_simulations = num_sims_attacker if nn_plays_attacker else num_sims_defender
    
    # Create agents on CPU
    # Pass MoveEncoderClass to ensure correct encoder is used (especially after torch.compile)
    nn_agent = Agent(network, num_simulations=num_simulations,
                     c_puct=c_puct, device='cpu', move_encoder_class=MoveEncoderClass)
    random_agent = RandomAgent()
    
    # Play game
    if nn_plays_attacker:
        winner = play_game(nn_agent, random_agent, GameClass, display=False)
        return 1 if winner == 0 else 0
    else:
        winner = play_game(random_agent, nn_agent, GameClass, display=False)
        return 1 if winner == 1 else 0


def calculate_elo_difference(win_rate: float) -> float:
    """
    Calculate ELO rating difference from win rate.
    
    ELO formula: diff = -400 * log10((1/win_rate) - 1)
    
    Args:
        win_rate: win rate between 0 and 1
    
    Returns:
        ELO difference (positive means stronger)
    """
    # Clamp win rate to avoid log(0)
    win_rate = np.clip(win_rate, 0.01, 0.99)
    
    # ELO difference from win rate
    elo_diff = -400 * np.log10((1 / win_rate) - 1)
    
    return elo_diff


def evaluate_vs_random(network: BrandubhNet, config: TrainingConfig, 
                       num_games: int = 10, pool=None, temp_network_path=None) -> Dict[str, float]:
    """
    Evaluate network against random player.
    Uses multiprocessing to parallelize evaluation games.
    
    Args:
        network: neural network to evaluate
        config: training configuration
        num_games: number of games to play (will play num_games as each color)
        pool: Optional multiprocessing pool to use (if None, creates temporary pool)
        temp_network_path: Path to temporary network file (avoids pickling large tensors)
    
    Returns:
        dict with 'total_wins', 'total_games', 'win_rate', 'elo_diff',
        'attacker_wins', 'defender_wins'
    """
    print(f"\nEvaluating vs Random player ({num_games * 2} games using {config.num_workers} workers)...")
    
    # Save network to temporary file to avoid pickling issues with many workers
    import tempfile
    if temp_network_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pth', delete=False)
        temp_network_path = temp_file.name
        temp_file.close()
        # Save as checkpoint format (workers expect 'model_state_dict' key)
        # Strip _orig_mod. prefix if network is compiled (torch.compile adds this)
        checkpoint = {'model_state_dict': clean_state_dict(network.state_dict())}
        torch.save(checkpoint, temp_network_path)
        cleanup_temp_file = True
    else:
        cleanup_temp_file = False
    
    try:
        # Create worker functions with partial for attacker and defender games
        worker_func_attacker = partial(
            _evaluate_vs_random_worker,
            temp_network_path,
            config.num_res_blocks,
            config.num_channels,
            config.eval_mcts_sims_attacker,
            config.eval_mcts_sims_defender,
            config.c_puct,
            True,  # nn_plays_attacker
            network_module=config.network_class.__module__,
            network_class_name=config.network_class.__name__,
            game_module=config.game_class.__module__,
            game_class_name=config.game_class.__name__
        )
        
        worker_func_defender = partial(
            _evaluate_vs_random_worker,
            temp_network_path,
            config.num_res_blocks,
            config.num_channels,
            config.eval_mcts_sims_attacker,
            config.eval_mcts_sims_defender,
            config.c_puct,
            False,  # nn_plays_attacker
            network_module=config.network_class.__module__,
            network_class_name=config.network_class.__name__,
            game_module=config.game_class.__module__,
            game_class_name=config.game_class.__name__
        )
        
        # Play games in parallel
        if config.num_workers > 1:
            if pool is not None:
                # Use provided persistent pool
                attacker_results = pool.map(worker_func_attacker, range(num_games), chunksize=1)
                defender_results = pool.map(worker_func_defender, range(num_games), chunksize=1)
            else:
                # Create temporary pool (for backward compatibility)
                with mp.Pool(processes=config.num_workers, maxtasksperchild=10) as temp_pool:
                    attacker_results = temp_pool.map(worker_func_attacker, range(num_games), chunksize=1)
                    defender_results = temp_pool.map(worker_func_defender, range(num_games), chunksize=1)
        else:
            # Single-threaded fallback
            attacker_results = [worker_func_attacker(i) for i in range(num_games)]
            defender_results = [worker_func_defender(i) for i in range(num_games)]
    finally:
        # Clean up temporary file
        if cleanup_temp_file:
            try:
                os.unlink(temp_network_path)
            except Exception:
                pass
    
    # Process results
    attacker_wins = sum(attacker_results)
    defender_wins = sum(defender_results)
    
    total_wins = attacker_wins + defender_wins
    total_games = num_games * 2
    win_rate = total_wins / total_games
    elo_diff = calculate_elo_difference(win_rate)
    
    print(f"  As Attacker: {attacker_wins}/{num_games} wins")
    print(f"  As Defender: {defender_wins}/{num_games} wins")
    print(f"  Total: {total_wins}/{total_games} wins ({100*win_rate:.1f}%)")
    print(f"  ELO vs Random: {elo_diff:+.0f}")
    
    return {
        'total_wins': total_wins,
        'total_games': total_games,
        'win_rate': win_rate,
        'elo_diff': elo_diff,
        'attacker_wins': attacker_wins,
        'defender_wins': defender_wins
    }


def _evaluate_networks_worker(new_network_path, old_network_path,
                             num_res_blocks, num_channels, num_sims_attacker, num_sims_defender, c_puct,
                             new_plays_attacker,
                             network_module, network_class_name, game_module, game_class_name,
                             game_idx):
    """
    Worker function for parallel network evaluation.
    Must be at module level for multiprocessing. Imports inside to avoid pickling issues.
    
    Args:
        new_network_path: Path to saved new network weights file
        old_network_path: Path to saved old network weights file
        num_res_blocks: Number of residual blocks
        num_channels: Number of channels
        num_sims_attacker: MCTS simulations per move for attacker
        num_sims_defender: MCTS simulations per move for defender
        c_puct: MCTS exploration constant
        new_plays_attacker: Whether new network plays as attacker
        network_module: Module name for network (e.g., 'network', 'network_tablut')
        network_class_name: Network class name (e.g., 'BrandubhNet', 'TablutNet')
        game_module: Module name for game (e.g., 'brandubh', 'tablut')
        game_class_name: Game class name (e.g., 'Brandubh', 'Tablut')
        game_idx: Game index (unused, for pool.map)
    
    Returns:
        1 if new network wins, 0 otherwise
    """
    # Seed random number generators uniquely for each worker
    import time
    seed = (game_idx + os.getpid() + int(time.time() * 1000)) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set torch to use only 1 thread per worker to avoid conflicts
    torch.set_num_threads(1)
    
    # Set per-worker compilation cache to avoid lock contention
    # Each worker gets its own cache directory, so no delays needed
    import tempfile
    worker_cache_dir = os.path.join(tempfile.gettempdir(), f'torchinductor_{os.getpid()}')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = worker_cache_dir
    
    # Import dynamically based on config
    import importlib
    game_mod = importlib.import_module(game_module)
    GameClass = getattr(game_mod, game_class_name)
    
    network_mod = importlib.import_module(network_module)
    NetworkClass = getattr(network_mod, network_class_name)
    
    # Import MoveEncoder from the same network module
    # For Brandubh: MoveEncoder from network.py
    # For Tablut: TablutMoveEncoder from network_tablut.py
    # For Hnefatafl: HnefataflMoveEncoder from network_hnefatafl.py
    if 'hnefatafl' in network_module.lower():
        MoveEncoderClass = getattr(network_mod, 'HnefataflMoveEncoder')
    elif 'tablut' in network_module.lower():
        MoveEncoderClass = getattr(network_mod, 'TablutMoveEncoder')
    else:
        MoveEncoderClass = getattr(network_mod, 'MoveEncoder')
    
    from agent import Agent, play_game
    
    # Reconstruct networks on CPU and load from files
    new_network = NetworkClass(num_res_blocks=num_res_blocks, num_channels=num_channels)
    checkpoint = torch.load(new_network_path, map_location='cpu', weights_only=False)
    new_network.load_state_dict(checkpoint['model_state_dict'])
    new_network.to('cpu')
    new_network.eval()
    
    # Optimize for CPU inference with compilation for 2.5-2.75x speedup
    # Each worker has its own cache directory, so no contention
    new_network = new_network.optimize_for_inference(use_compile=True, compile_mode='default')
    
    old_network = NetworkClass(num_res_blocks=num_res_blocks, num_channels=num_channels)
    checkpoint = torch.load(old_network_path, map_location='cpu', weights_only=False)
    old_network.load_state_dict(checkpoint['model_state_dict'])
    old_network.to('cpu')
    old_network.eval()
    
    # Optimize for CPU inference with compilation for 2.5-2.75x speedup
    old_network = old_network.optimize_for_inference(use_compile=True, compile_mode='default')
    
    # Determine simulations based on roles
    # Both agents use role-appropriate simulation counts
    new_sims = num_sims_attacker if new_plays_attacker else num_sims_defender
    old_sims = num_sims_defender if new_plays_attacker else num_sims_attacker
    
    # Create agents on CPU
    # Pass MoveEncoderClass to ensure correct encoder is used (especially after torch.compile)
    new_agent = Agent(new_network, num_simulations=new_sims,
                      c_puct=c_puct, device='cpu', add_dirichlet_noise=True,
                      move_encoder_class=MoveEncoderClass)
    old_agent = Agent(old_network, num_simulations=old_sims,
                      c_puct=c_puct, device='cpu', add_dirichlet_noise=True,
                      move_encoder_class=MoveEncoderClass)
    
    # Play game
    if new_plays_attacker:
        winner = play_game(new_agent, old_agent, GameClass, display=False)
        return 1 if winner == 0 else 0
    else:
        winner = play_game(old_agent, new_agent, GameClass, display=False)
        return 1 if winner == 1 else 0


def evaluate_networks(new_network: BrandubhNet, old_network: BrandubhNet, 
                     config: TrainingConfig, pool=None, temp_new_path=None, temp_old_path=None) -> float:
    """
    Evaluate new network against old network.
    Uses multiprocessing to parallelize evaluation games.
    
    Args:
        new_network: New network to evaluate
        old_network: Old network to compare against
        config: Training configuration
        pool: Optional multiprocessing pool to use (if None, creates temporary pool)
        temp_new_path: Path to temporary new network file (avoids pickling large tensors)
        temp_old_path: Path to temporary old network file (avoids pickling large tensors)
    
    Returns:
        win_rate: fraction of games won by new network
    """
    print(f"\nEvaluating new network vs old network ({config.eval_games} games using {config.num_workers} workers)...")
    
    # Save networks to temporary files to avoid pickling issues with many workers
    import tempfile
    if temp_new_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_new.pth', delete=False)
        temp_new_path = temp_file.name
        temp_file.close()
        # Save as checkpoint format (workers expect 'model_state_dict' key)
        # Strip _orig_mod. prefix if network is compiled (torch.compile adds this)
        checkpoint = {'model_state_dict': clean_state_dict(new_network.state_dict())}
        torch.save(checkpoint, temp_new_path)
        cleanup_new = True
    else:
        cleanup_new = False
    
    if temp_old_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_old.pth', delete=False)
        temp_old_path = temp_file.name
        temp_file.close()
        # Save as checkpoint format (workers expect 'model_state_dict' key)
        # Strip _orig_mod. prefix if network is compiled (torch.compile adds this)
        checkpoint = {'model_state_dict': clean_state_dict(old_network.state_dict())}
        torch.save(checkpoint, temp_old_path)
        cleanup_old = True
    else:
        cleanup_old = False
    
    try:
        # Create worker function with partial - alternates who plays attacker based on game index
        # Game indices 0, 2, 4, ... have new network as attacker
        # Game indices 1, 3, 5, ... have new network as defender
        # This is handled by using modulo inside the worker function parameters
        
        # Create two partial functions - one for new as attacker, one for new as defender
        half_games = config.eval_games // 2
        remainder = config.eval_games % 2
        
        worker_func_new_attacker = partial(
            _evaluate_networks_worker,
            temp_new_path,
            temp_old_path,
            config.num_res_blocks,
            config.num_channels,
            config.eval_mcts_sims_attacker,
            config.eval_mcts_sims_defender,
            config.c_puct,
            True,  # new_plays_attacker
            config.network_class.__module__,
            config.network_class.__name__,
            config.game_class.__module__,
            config.game_class.__name__
        )
        
        worker_func_new_defender = partial(
            _evaluate_networks_worker,
            temp_new_path,
            temp_old_path,
            config.num_res_blocks,
            config.num_channels,
            config.eval_mcts_sims_attacker,
            config.eval_mcts_sims_defender,
            config.c_puct,
            False,  # new_plays_attacker
            config.network_class.__module__,
            config.network_class.__name__,
            config.game_class.__module__,
            config.game_class.__name__
        )
        
        # Play games in parallel using pool.map for true parallelization
        if config.num_workers > 1:
            if pool is not None:
                # Use provided persistent pool
                attacker_results = pool.map(worker_func_new_attacker, range(half_games + remainder), chunksize=1)
                defender_results = pool.map(worker_func_new_defender, range(half_games), chunksize=1)
                results = attacker_results + defender_results
            else:
                # Create temporary pool (for backward compatibility)
                with mp.Pool(processes=config.num_workers, maxtasksperchild=10) as temp_pool:
                    attacker_results = temp_pool.map(worker_func_new_attacker, range(half_games + remainder), chunksize=1)
                    defender_results = temp_pool.map(worker_func_new_defender, range(half_games), chunksize=1)
                    results = attacker_results + defender_results
        else:
            # Single-threaded fallback
            results = [worker_func_new_attacker(i) for i in range(half_games + remainder)]
            results += [worker_func_new_defender(i) for i in range(half_games)]
    finally:
        # Clean up temporary files
        if cleanup_new:
            try:
                os.unlink(temp_new_path)
            except Exception:
                pass
        if cleanup_old:
            try:
                os.unlink(temp_old_path)
            except Exception:
                pass
    
    # Process results - split by attacker/defender
    attacker_wins = sum(results[:half_games + remainder])
    defender_wins = sum(results[half_games + remainder:])
    total_wins = attacker_wins + defender_wins
    win_rate = total_wins / config.eval_games
    
    print(f"  As Attacker: {attacker_wins}/{half_games + remainder} wins")
    print(f"  As Defender: {defender_wins}/{half_games} wins")
    print(f"New network win rate: {100*win_rate:.1f}% ({total_wins}/{config.eval_games})")
    
    return win_rate


# =============================================================================
# CHECKPOINTING
# =============================================================================

def clean_state_dict(state_dict):
    """
    Remove _orig_mod. prefix from state dict keys if present.
    
    torch.compile() wraps models and adds _orig_mod. prefix to all parameters.
    This function strips that prefix to make the state dict compatible with
    uncompiled models.
    
    Args:
        state_dict: State dict possibly containing _orig_mod. prefixes
        
    Returns:
        Cleaned state dict without _orig_mod. prefixes
    """
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            cleaned[key[10:]] = value  # Remove '_orig_mod.' prefix (10 chars)
        else:
            cleaned[key] = value
    return cleaned


def save_checkpoint(network: BrandubhNet, optimizer: optim.Optimizer,
                   iteration: int, config: TrainingConfig, 
                   metrics: Dict, filename: str = None):
    """Save model checkpoint."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_iter_{iteration}.pth"
    
    filepath = os.path.join(config.checkpoint_dir, filename)
    
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.__dict__
    }
    
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint: {filepath}")


def load_checkpoint(filepath: str, network: BrandubhNet, 
                   optimizer: optim.Optimizer = None) -> int:
    """Load model checkpoint. Returns iteration number."""
    checkpoint = torch.load(filepath, weights_only=False)
    
    network.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    iteration = checkpoint.get('iteration', 0)
    print(f"Loaded checkpoint from iteration {iteration}")
    
    return iteration


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(config: TrainingConfig, resume_from: str = None):
    """
    Main training loop.
    
    Args:
        config: TrainingConfig object
        resume_from: path to checkpoint to resume from
    """
    # Validate that required game configuration is set
    if config.game_class is None:
        raise ValueError("config.game_class must be set (e.g., to Brandubh)")
    if config.network_class is None:
        raise ValueError("config.network_class must be set (e.g., to BrandubhNet)")
    if config.agent_class is None:
        raise ValueError("config.agent_class must be set (e.g., to BrandubhAgent)")
    if config.move_encoder_class is None:
        raise ValueError("config.move_encoder_class must be set (e.g., to MoveEncoder)")
    
    print("=" * 70)
    print(f"{config.game_class.__name__} AlphaZero Training")
    print("=" * 70)
    
    # Training configuration
    print("\n--- Training Configuration ---")
    print(f"Device: {config.device}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Games per iteration: {config.num_games_per_iteration}")
    print(f"Parallel workers: {config.num_workers}")
    
    # Game rules
    print("\n--- Game Rules ---")
    print(f"King capture pieces: {config.king_capture_pieces}")
    capture_desc = {
        2: "(standard custodian - 2 opposite attackers)",
        3: "(3 out of 4 sides surrounded)",
        4: "(all 4 sides surrounded)"
    }
    print(f"  {capture_desc.get(config.king_capture_pieces, '')}")
    print(f"King can capture: {config.king_can_capture}")
    print(f"Throne enabled: {config.throne_enabled}")
    if config.throne_enabled:
        print(f"Throne is hostile: {config.throne_is_hostile}")
    else:
        print(f"  (Throne disabled - center square acts as normal square)")
    
    # Network architecture
    print("\n--- Network Architecture ---")
    print(f"Residual blocks: {config.num_res_blocks}")
    print(f"Channels: {config.num_channels}")
    print(f"Input: 4 planes ({config.board_size}×{config.board_size}) - [attackers, defenders, king, current_player]")
    print(f"Policy output: {config.policy_size} moves ({config.num_squares} squares × 4 directions × {config.board_size - 1} distances)")
    print("Value output: single scalar (win probability)")
    
    # MCTS parameters
    print("\n--- MCTS Parameters ---")
    print("Self-play simulations:")
    print(f"  Attacker: {config.num_mcts_sims_attacker}")
    print(f"  Defender: {config.num_mcts_sims_defender}")
    print("Evaluation simulations:")
    print(f"  Attacker: {config.eval_mcts_sims_attacker}")
    print(f"  Defender: {config.eval_mcts_sims_defender}")
    print(f"Exploration constant (c_puct): {config.c_puct}")
    print(f"Temperature: {config.temperature}")
    if config.temperature_threshold == "king":
        print(f"Temperature threshold: drop when king leaves throne")
    else:
        print(f"Temperature threshold: {config.temperature_threshold} moves")
    
    # Training parameters
    print("\n--- Training Parameters ---")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs per iteration: {config.num_epochs}")
    print(f"Batches per epoch: {config.batches_per_epoch}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LR decay per iteration: {config.lr_decay}")
    print(f"Weight decay (L2): {config.weight_decay}")
    print(f"Value loss weight: {config.value_loss_weight}")
    
    # Loss boosting
    print("\n--- Loss Boosting ---")
    if config.use_dynamic_boosting:
        print("Mode: Dynamic (adaptive based on win rates)")
        print(f"  Smoothing alpha: {config.dynamic_boost_alpha}")
        print(f"  Boost range: [{config.dynamic_boost_min}, {config.dynamic_boost_max}]")
    else:
        print("Mode: Static")
        print(f"  Attacker boost: {config.attacker_win_loss_boost}x")
    print(f"Draw penalties: attacker={config.draw_penalty_attacker}, defender={config.draw_penalty_defender}")
    
    # Calculate training data statistics
    print("\n--- Training Data Usage ---")
    # Estimate moves generated per iteration (rough average: ~50 moves per game)
    avg_moves_per_game = 50
    if config.use_data_augmentation:
        moves_generated_per_iter = config.num_games_per_iteration * avg_moves_per_game * 8  # 8x from symmetries
    else:
        moves_generated_per_iter = config.num_games_per_iteration * avg_moves_per_game
    
    # Calculate training samples used per iteration
    samples_per_epoch = config.batch_size * config.batches_per_epoch
    total_training_samples = samples_per_epoch * config.num_epochs
    
    print(f"New positions generated per iteration: ~{moves_generated_per_iter:,}")
    print(f"  ({config.num_games_per_iteration} games × ~{avg_moves_per_game} moves" + 
          (f" × 8 augmentations)" if config.use_data_augmentation else ")"))
    print(f"Positions used in training per iteration: {total_training_samples:,}")
    print(f"  ({config.batches_per_epoch} batches × {config.batch_size} samples × {config.num_epochs} epochs)")
    
    training_usage_ratio = total_training_samples / moves_generated_per_iter
    print(f"Training usage ratio: {training_usage_ratio:.2f}x")
    if training_usage_ratio > 1:
        print(f"  → Each new position trained on ~{training_usage_ratio:.1f} times on average")
    else:
        print(f"  → Only ~{training_usage_ratio*100:.0f}% of new positions used in training")
    
    # Replay buffer
    print("\n--- Replay Buffer ---")
    print(f"Max size: {config.replay_buffer_size:,} samples")
    print(f"Min size for training: {config.min_buffer_size:,} samples")
    print(f"Data augmentation: {'Enabled (8x symmetries)' if config.use_data_augmentation else 'Disabled'}")
    if config.use_data_augmentation:
        samples_per_iter = config.num_games_per_iteration * 50 * 8  # rough estimate
        print(f"Estimated samples/iteration: ~{samples_per_iter:,} ({config.num_games_per_iteration} games × ~50 moves × 8 augmentations)")
        print(f"Buffer turnover: ~{samples_per_iter/config.replay_buffer_size:.1f}x per iteration")
    
    # Draw handling
    print("\n--- Draw Penalties ---")
    print(f"Attacker draws: {config.draw_penalty_attacker}")
    print(f"Defender draws: {config.draw_penalty_defender}")
    
    # Evaluation
    print("\n--- Evaluation ---")
    print(f"Eval frequency (network vs network): every {config.eval_frequency} iterations")
    print(f"Eval games (network vs network): {config.eval_games}")
    print(f"Win rate threshold: {config.eval_win_rate:.1%}")
    print(f"Eval vs random frequency: every {config.eval_vs_random_frequency} iteration(s)")
    print(f"Eval vs random games: {config.eval_vs_random_games} per color ({config.eval_vs_random_games * 2} total)")
    print("Note: Will stop evaluating vs random once 100% win rate is achieved")
    
    # Checkpointing
    print("\n--- Checkpointing ---")
    print(f"Checkpoint directory: {config.checkpoint_dir}")
    print(f"Save frequency: every {config.save_frequency} iteration(s)")
    print()
    
    # Initialize network
    network = config.network_class(num_res_blocks=config.num_res_blocks,
                                   num_channels=config.num_channels).to(config.device)
    
    # Calculate and display network size
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Network initialized: {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Initialize optimizer with fused=True for faster CPU training
    # Note: fused Adam is faster but only available on CPU/CUDA, not all devices
    try:
        optimizer = optim.Adam(network.parameters(), 
                              lr=config.learning_rate,
                              weight_decay=config.weight_decay,
                              fused=True)
        print("Using fused Adam optimizer for better training performance")
    except Exception:
        # Fall back to standard Adam if fused not available
        optimizer = optim.Adam(network.parameters(), 
                              lr=config.learning_rate,
                              weight_decay=config.weight_decay)
        print("Using standard Adam optimizer (fused not available)")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if resume_from is not None:
        start_iteration = load_checkpoint(resume_from, network, optimizer)
    
    # Note: We do NOT compile the network here because:
    # 1. Compiled networks can't be pickled for multiprocessing
    # 2. We only need compilation speedup during training phase
    # We'll compile a separate copy for training only
    
    # Initialize best network (for evaluation)
    best_network = config.network_class(num_res_blocks=config.num_res_blocks,
                                        num_channels=config.num_channels).to(config.device)
    
    # Load best network from best_model.pth if resuming and it exists, otherwise copy current network
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    if resume_from is not None and os.path.exists(best_model_path):
        print(f"Loading best network from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=False)
        best_network.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Starting fresh or no best model exists yet
        # Clean state dict to remove _orig_mod. prefix from compiled network
        best_network.load_state_dict(clean_state_dict(network.state_dict()))
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(config.replay_buffer_size)
    
    # Initialize win rate tracker for dynamic boosting
    if config.use_dynamic_boosting:
        win_rate_tracker = WinRateTracker(
            alpha=config.dynamic_boost_alpha,
            min_boost=config.dynamic_boost_min,
            max_boost=config.dynamic_boost_max
        )
        print(f"Using dynamic loss boosting (alpha={config.dynamic_boost_alpha}, "
              f"range=[{config.dynamic_boost_min}, {config.dynamic_boost_max}])")
    else:
        win_rate_tracker = None
        print(f"Using static loss boosting (attacker={config.attacker_win_loss_boost}x)")
    
    # ELO tracking - cumulative ELO relative to initial random network
    cumulative_elo = 0.0  # Start at 0 (random network baseline)
    
    # Track if 100% win rate vs random has been achieved
    achieved_100_percent_vs_random = False
    
    # Training metrics
    training_history = {
        'iterations': [],
        'policy_loss': [],
        'value_loss': [],
        'total_loss': [],
        'win_rates': [],
        'buffer_size': [],
        'vs_random_win_rate': [],
        'vs_random_elo': [],
        'vs_random_attacker_wins': [],
        'vs_random_defender_wins': [],
        'cumulative_elo': [],  # ELO relative to iteration 0
        'elo_gain': []  # ELO gained in network vs network evaluation
    }
    
    # Create persistent worker pool to avoid file descriptor leaks
    # Use maxtasksperchild to periodically recycle workers and free resources
    worker_pool = None
    if config.num_workers > 1:
        print(f"Creating persistent worker pool with {config.num_workers} workers...")
        worker_pool = mp.Pool(processes=config.num_workers, maxtasksperchild=50)
    
    try:
        # Main training loop
        for iteration in range(start_iteration, config.num_iterations):
            iter_start_time = time.time()
            
            print("\n" + "=" * 70)
            print(f"Iteration {iteration + 1}/{config.num_iterations}")
            print("=" * 70)
            
            # 1. Self-play
            print("\n[1/4] Self-play game generation...")
            selfplay_start = time.time()
            agent = Agent(network, 
                         num_simulations=config.num_mcts_simulations,
                         c_puct=config.c_puct,
                         device=config.device)
            
            new_buffer, attacker_game_wins, defender_game_wins = generate_self_play_data(agent, config, pool=worker_pool)
            selfplay_time = time.time() - selfplay_start
            
            # Update win rate tracker with actual game results
            if config.use_dynamic_boosting:
                win_rate_tracker.update(attacker_game_wins, defender_game_wins)
                attacker_boost, defender_boost = win_rate_tracker.get_boost_factors()
                attacker_rate, defender_rate = win_rate_tracker.get_win_rates()
                print(f"Win rates (smoothed): Attacker {attacker_rate:.1%}, Defender {defender_rate:.1%}")
                ratio = attacker_boost / defender_boost if defender_boost > 0 else float('inf')
                print(
                    f"Dynamic loss boosts: Attacker {attacker_boost:.2f}x, "
                    f"Defender {defender_boost:.2f}x (ratio {ratio:.2f}x)"
                )
            else:
                attacker_boost = config.attacker_win_loss_boost
                defender_boost = 1.0
            
            # Add to main replay buffer
            for state, policy, value, attacker_won in new_buffer.buffer:
                replay_buffer.add(state, policy, value, attacker_won)
            
            print(f"Replay buffer size: {len(replay_buffer)}")
            print(f"Self-play time: {selfplay_time:.1f}s")
            
            # 2. Train network
            if len(replay_buffer) >= config.min_buffer_size:
                print(f"\n[2/4] Training network for {config.num_epochs} epochs...")
                training_start = time.time()
                losses = train_network(network, replay_buffer, optimizer, config,
                                     attacker_boost=attacker_boost,
                                     defender_boost=defender_boost)
                training_time = time.time() - training_start
                
                print(f"Training time: {training_time:.1f}s")
                print(f"Final losses - Policy: {losses['policy_loss']:.4f}, "
                      f"Value: {losses['value_loss']:.4f}, "
                      f"Total: {losses['total_loss']:.4f}")
                
                # Record metrics
                training_history['iterations'].append(iteration + 1)
                training_history['policy_loss'].append(losses['policy_loss'])
                training_history['value_loss'].append(losses['value_loss'])
                training_history['total_loss'].append(losses['total_loss'])
                training_history['buffer_size'].append(len(replay_buffer))
            else:
                print(f"\n[2/4] Skipping training: buffer size {len(replay_buffer)} < "
                      f"minimum {config.min_buffer_size}")
                training_time = 0
            
            # 3. Evaluate vs random player (only if 100% win rate not yet achieved)
            if not achieved_100_percent_vs_random and (iteration + 1) % config.eval_vs_random_frequency == 0:
                print("\n[3/4] Evaluating vs Random player...")
                eval_random_start = time.time()
                random_eval = evaluate_vs_random(network, config, 
                                                num_games=config.eval_vs_random_games,
                                                pool=worker_pool)
                eval_random_time = time.time() - eval_random_start
                print(f"Random evaluation time: {eval_random_time:.1f}s")
                training_history['vs_random_win_rate'].append(random_eval['win_rate'])
                training_history['vs_random_elo'].append(random_eval['elo_diff'])
                training_history['vs_random_attacker_wins'].append(random_eval['attacker_wins'])
                training_history['vs_random_defender_wins'].append(random_eval['defender_wins'])
                
                # Check if 100% win rate achieved
                if random_eval['win_rate'] >= 1.0:
                    achieved_100_percent_vs_random = True
                    print("\n" + "=" * 70)
                    print("🎉 MILESTONE: Achieved 100% win rate vs random player!")
                    print("No longer evaluating vs random for remaining iterations.")
                    print("=" * 70)
            elif achieved_100_percent_vs_random:
                print("\n[3/4] Skipping random evaluation (100% win rate already achieved)")
                eval_random_time = 0
            else:
                print(f"\n[3/4] Skipping random evaluation (every {config.eval_vs_random_frequency} iterations)")
                eval_random_time = 0
            
            # 4. Evaluate and update best network
            if (iteration + 1) % config.eval_frequency == 0:
                print(f"\n[4/4] Evaluating new vs old network...")
                eval_network_start = time.time()
                win_rate = evaluate_networks(network, best_network, config, pool=worker_pool)
                eval_network_time = time.time() - eval_network_start
                print(f"Network evaluation time: {eval_network_time:.1f}s")
                training_history['win_rates'].append(win_rate)
                
                # Calculate ELO gain from this evaluation
                elo_gain = calculate_elo_difference(win_rate)
                training_history['elo_gain'].append(elo_gain)
                
                if win_rate >= config.eval_win_rate:
                    print(f"New network wins {100*win_rate:.1f}% - updating best network!")
                    print(f"ELO gain: {elo_gain:+.1f} (new network vs previous best)")
                    # Update cumulative ELO by adding this gain
                    cumulative_elo += elo_gain
                    # Clean state dict to remove _orig_mod. prefix from compiled network
                    best_network.load_state_dict(clean_state_dict(network.state_dict()))
                    # Save the new best model immediately
                    save_checkpoint(best_network, optimizer, iteration + 1, config,
                                  training_history, "best_model.pth")
                else:
                    print(f"New network wins {100*win_rate:.1f}% - keeping old network")
                    print(f"ELO difference: {elo_gain:+.1f} (new network vs previous best, not applied)")
                
                # Record cumulative ELO (relative to iteration 0)
                training_history['cumulative_elo'].append(cumulative_elo)
                print(f"Cumulative ELO (vs iteration 0): {cumulative_elo:+.1f}")
            else:
                print(f"\n[4/4] Skipping network evaluation (every {config.eval_frequency} iterations)")
                eval_network_time = 0
            
            # 5. Learning rate decay
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nLearning rate: {current_lr:.6f}")
            
            # 6. Save checkpoint (only save the current network checkpoint, not best)
            if (iteration + 1) % config.save_frequency == 0:
                save_checkpoint(network, optimizer, iteration + 1, config, 
                              training_history, f"checkpoint_iter_{iteration + 1}.pth")
            
            # 7. Save training history
            history_path = os.path.join(config.checkpoint_dir, "training_history.json")
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
            
            iter_time = time.time() - iter_start_time
            
            # Print timing summary
            print("\n" + "-" * 70)
            print("Timing Summary:")
            print(f"  Self-play:          {selfplay_time:6.1f}s  ({100*selfplay_time/iter_time:.1f}%)")
            if training_time > 0:
                print(f"  Training:           {training_time:6.1f}s  ({100*training_time/iter_time:.1f}%)")
            if eval_random_time > 0:
                print(f"  Random evaluation:  {eval_random_time:6.1f}s  ({100*eval_random_time/iter_time:.1f}%)")
            if eval_network_time > 0:
                print(f"  Network evaluation: {eval_network_time:6.1f}s  ({100*eval_network_time/iter_time:.1f}%)")
            print(f"  Total:              {iter_time:6.1f}s")
            
            # Print ELO summary
            if len(training_history['vs_random_elo']) > 0 or len(training_history['cumulative_elo']) > 0:
                print("\nELO Summary:")
                if len(training_history['vs_random_elo']) > 0:
                    latest_random_elo = training_history['vs_random_elo'][-1]
                    print(f"  Current network vs Random: {latest_random_elo:+.0f} ELO")
                if len(training_history['cumulative_elo']) > 0:
                    print(f"  Best network vs Iteration 0: {cumulative_elo:+.0f} ELO")
                    print(f"    (Chained through {len(training_history['cumulative_elo'])} evaluations)")
            print("-" * 70)
    
    finally:
        # Clean up worker pool
        if worker_pool is not None:
            print("\nClosing worker pool...")
            worker_pool.close()
            worker_pool.join()
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    # Final save
    save_checkpoint(network, optimizer, config.num_iterations, config,
                   training_history, "final_model.pth")
    save_checkpoint(best_network, optimizer, config.num_iterations, config,
                   training_history, "best_model.pth")
    
    return network, best_network, training_history


# =============================================================================
# MAIN (DEPRECATED - use train_brandubh.py or create a similar game-specific script)
# =============================================================================

# The __main__ block has been moved to train_brandubh.py to make this module reusable
# for different board sizes and game variants. To train Brandubh:
#   python train_brandubh.py [OPTIONS]
#
# For other game variants, create a similar script that:
#   1. Imports TrainingConfig and train from this module
#   2. Imports your game class, network class, agent class
#   3. Configures the TrainingConfig with game-specific settings
#   4. Calls train(config, resume_from=...)
