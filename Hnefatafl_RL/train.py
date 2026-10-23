"""
Self-play training loop for Tafl-style games using AlphaZero.

This is a generic training module for all board sizes and game variants. Runs are started with
run_training.py and a JSON config (see run_config.py and the configs/ directory).

Implements the complete AlphaZero training pipeline:
- Self-play game generation with MCTS
- Replay buffer with experience storage
- Neural network training
- Model evaluation and checkpointing
- Data augmentation (symmetries)

The rule set (tafl.py) and the network architecture (network_config.py, tafl_network.py) are both
chosen by the TrainingConfig, so no game-specific training script is needed.
"""

import os
import sys
import time
import json
import glob
import shutil
import socket
from datetime import datetime
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

import torch.nn as nn
import torch.nn.functional as F

from agent import Agent
from features import StateHistory, build_features
from network_config import NetworkConfig
from run_config import TrainingConfig, save_config  # TrainingConfig lives in run_config.py (JSON form there)
from tafl import ENDINGS, Tafl, move_index
from tafl_network import MOVES_LEFT_SCALE, TaflNet, load_weights, describe_network


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """
    Ring buffer of self-play positions, each stored once in preallocated NumPy arrays.

    With data augmentation, one stored position stands for its 8 board symmetries: sizes are counted
    in samples (positions x num_symmetries), the unit of replay_buffer_size and min_buffer_size, and
    train_network applies a random symmetry to every position it draws. States are stored as uint8,
    since all input planes are 0 or 1. winners holds the game result: 0 attacker, 1 defender,
    -1 draw. endings holds how the game ended (an index into tafl.ENDINGS), moves_left the number
    of plies from the position to the end of the game.
    """

    def __init__(self, max_samples: int, state_shape: Tuple[int, ...], policy_size: int,
                 num_symmetries: int = 8):
        self.num_symmetries = num_symmetries
        capacity = max_samples // num_symmetries
        self.states = np.zeros((capacity,) + tuple(state_shape), dtype=np.uint8)
        self.policies = np.zeros((capacity, policy_size), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.winners = np.zeros(capacity, dtype=np.int8)
        self.endings = np.zeros(capacity, dtype=np.int64)
        self.moves_left = np.zeros(capacity, dtype=np.float32)
        self.num_positions = 0  # number of filled entries
        self.next_index = 0     # entry written next; once the buffer is full, the oldest entry

    def add(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray,
            winners: np.ndarray, endings: np.ndarray, moves_left: np.ndarray):
        """Add positions (one row each), overwriting the oldest ones when the buffer is full."""
        if not np.all((states == 0) | (states == 1)):
            raise ValueError("ReplayBuffer stores states as uint8: planes must hold only 0 and 1")
        capacity = len(self.values)
        indices = (self.next_index + np.arange(len(values))) % capacity
        self.states[indices] = states
        self.policies[indices] = policies
        self.values[indices] = values
        self.winners[indices] = winners
        self.endings[indices] = endings
        self.moves_left[indices] = moves_left
        self.next_index = (self.next_index + len(values)) % capacity
        self.num_positions = min(self.num_positions + len(values), capacity)

    def __len__(self):
        """Number of samples, i.e. positions times symmetries."""
        return self.num_positions * self.num_symmetries


# =============================================================================
# TEMP DIRECTORY MANAGEMENT
# =============================================================================

def setup_temp_directory(temp_dir: str = None) -> str:
    """
    Configure and create temporary directory for PyTorch operations.
    
    This sets up a temporary directory for:
    - TorchInductor compilation cache (torch.compile)
    - Temporary checkpoint files for multiprocessing
    - Other PyTorch temporary files
    
    Args:
        temp_dir: Path to temporary directory (e.g., "/dev/shm" for RAM disk)
                 If None, uses system default (/tmp)
    
    Returns:
        Path to the configured temp directory
    """
    import tempfile
    
    if temp_dir is not None:
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Set environment variables for Python tempfile module
        os.environ['TMPDIR'] = temp_dir
        os.environ['TEMP'] = temp_dir
        os.environ['TMP'] = temp_dir
        
        # Verify tempfile will use our directory
        actual_temp = tempfile.gettempdir()
        if os.path.normpath(actual_temp) != os.path.normpath(temp_dir):  # "/dev/shm/" is "/dev/shm"
            print(f"Warning: tempfile.gettempdir() returns {actual_temp}, expected {temp_dir}")
        
        return temp_dir
    else:
        # Use system default
        return tempfile.gettempdir()


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


class SimulationBalancer:
    """
    Dynamically adjusts MCTS simulation counts for attacker and defender, so that neither side wins so
    rarely that learning stalls, while keeping the handicap as small as possible.

    The attacker's score s = (attacker wins + draws / 2) / games is kept in the band
    [min_win_rate, 1 - min_win_rate]; the defender's score 1 - s is then in the same band. Above the
    band, simulations move from the attacker to the defender; below it, the other way. Inside the band
    the split drifts back towards equal, so the balancer settles at the smallest handicap that keeps
    the score at the band edge, or at no handicap if the game is balanced enough without one.
    The total simulation count stays constant. min_win_rate = 0.5 always pushes towards a 50 % score.
    """

    def __init__(self, initial_attacker_sims: int, initial_defender_sims: int,
                 adjustment_rate: float = 0.1, min_win_rate: float = 0.5):
        """
        Args:
            initial_attacker_sims: Starting simulation count for attacker
            initial_defender_sims: Starting simulation count for defender
            adjustment_rate: Fraction of the distance to the band (or, inside the band, of the distance to
                             an equal split) that is corrected per update (e.g., 0.1 = 10%)
            min_win_rate: Lowest score either side may have before simulations are moved to it (0 < x <= 0.5)
        """
        if not 0 < min_win_rate <= 0.5:
            raise ValueError(f"min_win_rate must be in (0, 0.5], got {min_win_rate}")
        self.total_sims = initial_attacker_sims + initial_defender_sims
        self.attacker_sims = initial_attacker_sims
        self.defender_sims = initial_defender_sims
        self.adjustment_rate = adjustment_rate
        self.min_win_rate = min_win_rate

        # Track history for debugging
        self.history = []

    def update(self, attacker_wins: int, defender_wins: int, draws: int = 0):
        """
        Update simulation counts based on the attacker's score in the last games.

        Outside the band, adjustment_rate * (distance to the band) * total_sims simulations move to the
        side that scores too little. Example: band [0.2, 0.8], adjustment_rate 0.1, 1000 + 1000 sims,
        attacker score 0.9: 0.1 * 0.1 * 2000 = 20 simulations move, giving attacker 980, defender 1020.
        With min_win_rate = 0.5 and no draws this is the earlier rule (transfer proportional to the
        difference between the two win rates).
        Inside the band, the attacker's count moves adjustment_rate of the way back to total_sims / 2.

        Args:
            attacker_wins: Number of attacker wins
            defender_wins: Number of defender wins
            draws: Number of draws (each counts as half a win for both sides)
        """
        total_games = attacker_wins + defender_wins + draws

        if total_games == 0:
            return  # No data to adjust with

        attacker_score = (attacker_wins + 0.5 * draws) / total_games
        low, high = self.min_win_rate, 1 - self.min_win_rate

        if attacker_score > high:
            # Attacker scores too much: give simulations to the defender
            self.attacker_sims -= self.adjustment_rate * (attacker_score - high) * self.total_sims
        elif attacker_score < low:
            # Defender scores too much: give simulations to the attacker
            self.attacker_sims += self.adjustment_rate * (low - attacker_score) * self.total_sims
        elif attacker_score != low and attacker_score != high:
            # Strictly inside the band: drift back towards an equal split (no change exactly on an edge,
            # so that a zero-width band at 0.5 behaves like the earlier rule)
            self.attacker_sims += self.adjustment_rate * (self.total_sims / 2 - self.attacker_sims)

        # Keep at least 1 simulation per side, round to integers, and keep the total constant
        self.attacker_sims = int(round(max(1, min(self.total_sims - 1, self.attacker_sims))))
        self.defender_sims = self.total_sims - self.attacker_sims

        # Record history
        self.history.append({
            'attacker_wins': attacker_wins,
            'defender_wins': defender_wins,
            'draws': draws,
            'attacker_score': attacker_score,
            'attacker_sims': self.attacker_sims,
            'defender_sims': self.defender_sims,
        })
    
    def get_simulation_counts(self) -> Tuple[int, int]:
        """
        Get current simulation counts.
        
        Returns:
            (attacker_sims, defender_sims) tuple
        """
        return self.attacker_sims, self.defender_sims


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
    
    # Explicit mapping for 90 degree clockwise rotation based on the policy index (tafl.move_index)
    # 0 (Up)    -> 3 (Right)
    # 3 (Right) -> 1 (Down)
    # 1 (Down)  -> 2 (Left)
    # 2 (Left)  -> 0 (Up)
    ROT_90_MAP = {0: 3, 3: 1, 1: 2, 2: 0}
    
    # Helper to rotate a move in policy space
    def rotate_policy(policy_vec, k):
        """Rotate policy vector k times 90 degrees clockwise."""
        policy_arr = policy_vec.reshape(num_squares, policy_moves_per_square)
        
        # Create mapping for k rotations
        dir_map = {}
        for d in range(4):
            curr = d
            for _ in range(k):
                curr = ROT_90_MAP[curr]
            dir_map[d] = curr

        # Rotate board positions and directions
        rotated = np.zeros_like(policy_arr)
        for old_sq in range(num_squares):
            old_r, old_c = old_sq // board_size, old_sq % board_size
            
            # Rotate position k times
            curr_r, curr_c = old_r, old_c
            for _ in range(k):
                curr_r, curr_c = curr_c, (board_size - 1) - curr_r
            new_sq = curr_r * board_size + curr_c
            
            # Rotate directions
            for old_dir in range(num_directions):
                new_dir = dir_map[old_dir]
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
                # Up(0)->Up(0), Down(1)->Down(1), Left(2)->Right(3), Right(3)->Left(2)
                dir_map = {0: 0, 1: 1, 2: 3, 3: 2}
            else:
                new_r, new_c = (board_size - 1) - old_r, old_c
                # Up(0)->Down(1), Down(1)->Up(0), Left(2)->Left(2), Right(3)->Right(3)
                dir_map = {0: 1, 1: 0, 2: 2, 3: 3}
            
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

    # 4 rotations. np.rot90 with positive k turns counter-clockwise, so k=-k is used to match the
    # clockwise rotation (r, c) -> (c, n-1-r) in rotate_policy.
    for k in range(1, 4):
        rotated_state = np.rot90(state, k=-k, axes=(1, 2)).copy()
        rotated_policy = rotate_policy(policy, k)
        augmented.append((rotated_state, rotated_policy, value))

    # Horizontal flip + 4 rotations
    flipped_state = np.flip(state, axis=2).copy()
    flipped_policy = flip_policy(policy, horizontal=True)
    augmented.append((flipped_state, flipped_policy, value))

    for k in range(1, 4):
        rotated_state = np.rot90(flipped_state, k=-k, axes=(1, 2)).copy()
        rotated_policy = rotate_policy(flipped_policy, k)
        augmented.append((rotated_state, rotated_policy, value))

    return augmented


def symmetry_permutations(board_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Index tables of the 8 board symmetries, in the order of augment_sample (identity first).

    They are made by applying augment_sample to a board whose cells hold their own index and a
    policy whose entries hold their own index: each output entry then names the input entry it came
    from.

    Returns:
        cell_perm: shape (8, board_size**2); symmetry k turns planes p (flattened over the board)
                   into p[..., cell_perm[k]]
        move_perm: shape (8, policy_size); symmetry k turns a policy q into q[move_perm[k]]
    """
    cell_index = np.arange(board_size**2, dtype=np.float32).reshape(1, board_size, board_size)
    move_index = np.arange(board_size * board_size * 4 * (board_size - 1), dtype=np.float32)
    augmented = augment_sample(cell_index, move_index, 0.0, board_size)
    cell_perm = np.array([state.reshape(-1) for state, _, _ in augmented]).astype(np.int64)
    move_perm = np.array([policy for _, policy, _ in augmented]).astype(np.int64)
    return cell_perm, move_perm


# =============================================================================
# GAME RECORDS
# =============================================================================

def game_record(game: Tafl, game_type: str, network_side: str = None) -> dict:
    """
    What games.jsonl keeps of a finished game: its type ("selfplay", "vs_random" or "vs_best"), who
    won ("attackers", "defenders" or None for a draw), how (one of tafl.ENDINGS), after how many
    plies, and the final board (one string per row, as in tafl setups: A, D, K and . for empty).
    In evaluation games, network_side is the side the network under evaluation played.
    """
    winner = None if game.winner is None else ('attackers', 'defenders')[game.winner]
    record = {'type': game_type, 'winner': winner, 'ending': game.end_reason,
              'plies': game.move_count,
              'board':["".join(".ADK"[piece] for piece in row) for row in game.board]}
    if network_side is not None:
        record['network_side'] = network_side
    return record


def print_endings(columns: Dict[str, List[dict]], rules):
    """Print how many games of each column (a list of game records) ended in each way the rules
    allow, with the share of the column's games."""
    print("  " + f"{'Ending':<20}" + "".join(f"{name:>24}" for name in columns))
    for ending in rules.endings():
        counts = [(sum(record['ending'] == ending for record in records), len(records))
                  for records in columns.values()]
        print("  " + f"{ending:<20}"
              + "".join(f"{n:>15} ({100 * n / max(total, 1):5.1f}%)" for n, total in counts))


def evaluation_summary(games: List[dict], rules) -> dict:
    """The results of the evaluated network in evaluation games (game records with
    'network_side'), for the training history: for each side it played ('attackers',
    'defenders'), the number of games, wins and draws, and how many games ended in each way."""
    summary = {}
    for side in ('attackers', 'defenders'):
        records = [record for record in games if record['network_side'] == side]
        summary[side] = {
            'games': len(records),
            'wins': sum(record['winner'] == side for record in records),
            'draws': sum(record['winner'] is None for record in records),
            'endings': {ending: sum(record['ending'] == ending for record in records)
                        for ending in rules.endings()},
        }
    return summary


# =============================================================================
# SELF-PLAY
# =============================================================================

def _play_self_play_game_worker_wrapper(args):
    """
    Wrapper for _play_self_play_game_worker that unpacks tuple arguments.
    Required for pool.imap_unordered() which passes a single argument.
    """
    return _play_self_play_game_worker(*args)


def _play_self_play_game_worker(network_path, network_config, rules,
                                num_sims_attacker, num_sims_defender, c_puct, temperature, temperature_mode,
                                temperature_threshold, temperature_decay_moves, game_idx,
                                add_dirichlet_noise, dirichlet_alpha, dirichlet_epsilon, fpu_reduction, temp_dir=None,
                                track_unique_states=False, use_random_opening_moves=False, max_game_length=500,
                                search_discount=1.0):
    """
    Worker function for parallel self-play game generation.
    Must be at module level for multiprocessing. Imports torch inside to avoid pickling issues.

    Args:
        network_path: Path to saved network weights file
        network_config: NetworkConfig.to_dict() of the network (architecture and input planes)
        rules: the rule set (tafl.Rules)
        num_sims_attacker: MCTS simulations per move for attacker
        num_sims_defender: MCTS simulations per move for defender
        c_puct: MCTS exploration constant
        temperature: Sampling temperature
        temperature_mode: "fixed", "king", or "decay"
        temperature_threshold: Move number for temperature drop (for "fixed" mode)
        temperature_decay_moves: Number of moves for linear decay (for "decay" mode)
        game_idx: Game index for seeding
        add_dirichlet_noise: Whether to add Dirichlet noise for exploration
        dirichlet_alpha: Concentration parameter for Dirichlet noise
        dirichlet_epsilon: Weight of Dirichlet noise (0-1)
        fpu_reduction: First Play Urgency reduction relative to parent Q-value
        temp_dir: Temporary directory for worker files
        track_unique_states: If True, collect state hashes for diversity analysis
        use_random_opening_moves: If True, 50% of games start with 1-4 random moves (2-8 ply)
        max_game_length: Maximum number of moves before declaring draw
        search_discount: Discount factor per move during MCTS value backup (1.0 = no discount)

    Returns:
        dict with game data and MCTS timing information
    """
    try:
        # Fresh seeds from the operating system for every game. Forked workers otherwise share one
        # random state, and a seed computed from game index + PID + time gave ~2-4 % of an
        # iteration's games the same seed as another game, i.e. exact copies.
        np.random.seed()
        torch.seed()

        # Set torch to use only 1 thread per worker to avoid conflicts
        torch.set_num_threads(1)
        # Compile in this process. The Inductor compile thread pool inherited from the parent does not
        # survive the fork: if the parent already used it (torch.compile during training), jobs
        # submitted to it never run and the worker hangs.
        from torch._inductor import config as inductor_config
        inductor_config.compile_threads = 1
        # Call the compiled kernels from generated C++ instead of Python: 0.3-0.4 ms (13-24 %) less
        # per batch-1 network call and about half the compile time, with identical results
        inductor_config.cpp_wrapper = True

        # Set up temp directory for this worker
        import tempfile
        if temp_dir is not None:
            # Use specified temp directory (e.g., RAM disk)
            os.environ['TMPDIR'] = temp_dir
            os.environ['TEMP'] = temp_dir
            os.environ['TMP'] = temp_dir
            base_temp_dir = temp_dir
        else:
            base_temp_dir = tempfile.gettempdir()
        
        # Set per-worker compilation cache to avoid lock contention
        # Each worker gets its own cache directory
        worker_cache_dir = os.path.join(base_temp_dir,
                                        f'torchinductor_{socket.gethostname()}_{os.getpid()}')
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = worker_cache_dir
        
        # Network architecture and input planes from the config
        board_size = rules.board_size
        net_config = NetworkConfig.from_dict(network_config)
        feature_spec = net_config.features

        from mcts import MCTS

        # Reconstruct network on CPU and load from file
        network = TaflNet(net_config, board_size)
        checkpoint = torch.load(network_path, map_location='cpu', weights_only=False)
        load_weights(network, checkpoint)
        network.to('cpu')
        network.eval()

        # Optimize for CPU inference with compilation for 2.5-2.75x speedup
        # Each worker has its own cache directory, so no contention
        network = network.optimize_for_inference(use_compile=True, compile_mode='default')

        # Create MCTS instances for each player (with different simulation counts)
        mcts_attacker = MCTS(network, num_simulations=num_sims_attacker, c_puct=c_puct, device='cpu',
                            add_dirichlet_noise=add_dirichlet_noise, dirichlet_alpha=dirichlet_alpha,
                            dirichlet_epsilon=dirichlet_epsilon, fpu_reduction=fpu_reduction,
                            search_discount=search_discount, feature_spec=feature_spec)
        mcts_defender = MCTS(network, num_simulations=num_sims_defender, c_puct=c_puct, device='cpu',
                            add_dirichlet_noise=add_dirichlet_noise, dirichlet_alpha=dirichlet_alpha,
                            dirichlet_epsilon=dirichlet_epsilon, fpu_reduction=fpu_reduction,
                            search_discount=search_discount, feature_spec=feature_spec)
        mcts_attacker.reset_timing_stats()
        mcts_defender.reset_timing_stats()

        game = Tafl(rules)
        states = []
        policies = []
        players = []
        plies = []  # the ply number of each stored position (for the moves-left target)
        mate_in_1_flags = []  # Track which positions had mate-in-1
        move_count = 0
        
        # State hash tracking (only if enabled)
        state_hashes = [] if track_unique_states else None
        
        # Initialize state history for history planes support
        if feature_spec.history_length > 1:
            state_history = StateHistory(history_length=feature_spec.history_length, board_size=board_size)
            state_history.push(game.get_piece_planes())
        else:
            state_history = None
        
        # Random opening moves (if enabled)
        # In 50% of games, play 1-4 random moves (2-8 ply) before starting MCTS
        if use_random_opening_moves:
            # Use 50% probability
            if np.random.random() < 0.5:
                # Randomly select 1-4 moves (each move is 2 ply - one for each player)
                num_random_moves = np.random.randint(1, 5)  # 1, 2, 3, or 4
                num_random_plies = num_random_moves * 2
                
                # Play random moves
                for _ in range(num_random_plies):
                    if game.game_over:
                        break
                    
                    legal_moves = game.get_legal_moves()
                    if not legal_moves:
                        break
                    
                    # Select random move
                    random_move = legal_moves[np.random.randint(len(legal_moves))]
                    game.make_move(random_move)
                    move_count += 1
                    
                    # Update state history after each random move
                    if state_history is not None:
                        state_history.push(game.get_piece_planes())
                # The random plies are not stored: the first MCTS move gives the first training sample

        king_left_throne = False  # for temperature_mode "king"
        while not game.game_over:
            current_player = game.current_player

            # Select MCTS based on current player
            mcts = mcts_attacker if current_player == 0 else mcts_defender

            # Determine temperature based on mode
            if temperature_mode == "king":
                # Temperature drops to 0 once the king has left the throne, and stays 0 (as in play_game)
                # KING piece value is 3 in all game variants
                king_left_throne = king_left_throne or game.board[game.throne] != 3
                temp = 0.0 if king_left_throne else temperature
            elif temperature_mode == "decay":
                # Linear decay over specified number of moves
                if move_count < temperature_decay_moves:
                    temp = temperature * (1.0 - move_count / temperature_decay_moves)
                else:
                    temp = 0.0
            else:  # "fixed" mode
                # Drop temperature after a fixed number of moves
                temp = temperature if move_count < temperature_threshold else 0.0
            
            # The network input of this position (the same planes the search builds)
            state = build_features(game, feature_spec, state_history)
            
            # Check for mate-in-1 (winning move in current position)
            mate_in_1_move = None
            legal_moves = game.get_legal_moves()
            for test_move in legal_moves:
                # Clone game and test the move
                test_game = game.clone()
                test_game.make_move(test_move)
                # Check if this move wins the game for current player
                if test_game.game_over and test_game.winner == current_player:
                    mate_in_1_move = test_move
                    break
            
            # If mate-in-1 exists, create forced policy and skip MCTS
            if mate_in_1_move is not None:
                # Create policy with 100% on winning move
                policy_size = board_size * board_size * 4 * (board_size - 1)
                policy = np.zeros(policy_size, dtype=np.float32)
                winning_move_idx = move_index(mate_in_1_move, board_size)
                policy[winning_move_idx] = 1.0
                
                # Create visit_probs for move selection (100% on winning move)
                visit_probs = {mate_in_1_move: 1.0}
            else:
                # Run MCTS to get policy (normal case)
                # Pass state_history for history planes support
                visit_probs = mcts.search(game, state_history=state_history)
                
                # Convert to policy vector
                policy_size = board_size * board_size * 4 * (board_size - 1)
                policy = np.zeros(policy_size, dtype=np.float32)
                for move, prob in visit_probs.items():
                    idx = move_index(move, board_size)
                    policy[idx] = prob
            
            # Store experience
            states.append(state)
            policies.append(policy)
            players.append(current_player)
            plies.append(move_count)
            mate_in_1_flags.append(mate_in_1_move is not None)  # Mark if this was a mate-in-1 position
            
            # Track state hash if enabled (for diversity analysis)
            if track_unique_states:
                # Create efficient hash of the board state (first 3 planes: pieces only)
                # Exclude the current_player plane (plane 3) as it alternates
                state_hash = hash(state[:3].tobytes())
                state_hashes.append(state_hash)
            
            # Select and make move
            moves = list(visit_probs.keys())
            probs = np.array(list(visit_probs.values()))
            
            if temp == 0:
                move = moves[np.argmax(probs)]
            else:
                # Apply temperature. Dividing by the largest share first keeps it at 1, so a small
                # temperature cannot round every entry down to 0.
                probs = (probs / probs.max()) ** (1.0 / temp)
                probs = probs / probs.sum()
                move_idx = np.random.choice(len(moves), p=probs)
                move = moves[move_idx]
            
            game.make_move(move)
            move_count += 1
            
            # Update state history after the move
            if state_history is not None:
                state_history.push(game.get_piece_planes())
            
            # Move limit (a win on the last allowed ply still counts as a win)
            if move_count > max_game_length and not game.game_over:
                game._end(None, "move_limit")

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
        record = game_record(game, 'selfplay')

        # Clean up references to prevent pickle errors
        # Convert states/policies to ensure they're picklable numpy arrays without references
        states_clean = [np.array(s, dtype=np.float32).copy() for s in states]
        policies_clean = [np.array(p, dtype=np.float32).copy() for p in policies]
        players_clean = [int(p) for p in players]
        plies_clean = [int(p) for p in plies]
        mate_in_1_flags_clean = [bool(f) for f in mate_in_1_flags]
        
        # Delete ALL objects that might hold references to compiled network
        del mcts_attacker
        del mcts_defender
        del network
        del game  # Delete game object too
        del states
        del policies
        del players
        del mate_in_1_flags
        del timing_attacker
        del timing_defender
        
        # Force garbage collection to clean up any circular references
        import gc
        gc.collect()
        
        result = {
            'states': states_clean,
            'policies': policies_clean,
            'winner': winner,
            'players': players_clean,
            'plies': plies_clean,
            'mate_in_1_flags': mate_in_1_flags_clean,
            'num_moves': num_moves,
            'record': record,  # for the ending statistics and games.jsonl
            'timing': combined_timing
        }
        
        # Add state hashes if tracking is enabled
        if track_unique_states and state_hashes is not None:
            result['state_hashes'] = state_hashes
        
        return result
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


def generate_self_play_data(agent: Agent, config: TrainingConfig, pool=None, temp_network_path=None,
                           attacker_sims: int = None,
                           defender_sims: int = None) -> Tuple[Tuple, List[dict], dict]:
    """
    Generate self-play games and their training positions.
    Uses multiprocessing to parallelize game generation.

    Args:
        agent: Agent to use for self-play
        config: Training configuration
        pool: Optional multiprocessing pool to use (if None, creates temporary pool). It is terminated
              at the end (the unfinished games are killed), so the caller must create a new one.
        temp_network_path: Path to temporary network file (avoids pickling large tensors)
        attacker_sims: Override for attacker simulation count (uses config value if None)
        defender_sims: Override for defender simulation count (uses config value if None)

    Returns:
        Tuple of (positions, games, stats) where:
        - positions: (states, policies, values, winners, endings, moves_left) arrays with one row
          per position, for ReplayBuffer.add (without symmetries; train_network applies those)
        - games: one game_record per game (plies include the random opening plies)
        - stats: 'unique_states', the number of board states and of distinct ones, of all states
          and without the first 1 and 6 plies of each game (None unless
          config.track_unique_states), and 'mcts_timing', the seconds spent in each part of MCTS
    """
    # Use provided simulation counts or fall back to config
    num_sims_attacker = attacker_sims if attacker_sims is not None else config.num_mcts_sims_attacker
    num_sims_defender = defender_sims if defender_sims is not None else config.num_mcts_sims_defender
    
    print(f"Generating {config.num_games_per_iteration} self-play games using {config.num_workers} workers...")
    if attacker_sims is not None or defender_sims is not None:
        print(f"  Using dynamic simulation counts: Attacker {num_sims_attacker}, Defender {num_sims_defender}")
    
    # Save network to temporary file to avoid pickling issues with many workers
    import tempfile
    
    # Use configured temp directory if specified
    if config.temp_dir is not None:
        temp_base_dir = config.temp_dir
    else:
        temp_base_dir = tempfile.gettempdir()
    
    if temp_network_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pth', delete=False, dir=temp_base_dir)
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
        # Worker arguments for up to 2 * num_games_per_iteration games (only the first N finished are used)
        network_config = config.network.to_dict()
        worker_args = [(temp_network_path, network_config, config.rules,
                        num_sims_attacker, num_sims_defender,
                        config.c_puct, config.temperature, config.temperature_mode,
                        config.temperature_threshold, config.temperature_decay_moves,
                        game_idx, config.add_dirichlet_noise, config.dirichlet_alpha, config.dirichlet_epsilon,
                        config.fpu_reduction, config.temp_dir, config.track_unique_states,
                        config.use_random_opening_moves, config.max_game_length, config.search_discount)
                       for game_idx in range(2 * config.num_games_per_iteration)]

        # Play games in parallel. imap_unordered with chunksize=1 hands out one game at a time, and once
        # num_games_per_iteration games are done the games still running are killed. This keeps every
        # worker busy until the end (waiting for the last games instead leaves ~10-18 % of worker time
        # idle). The cost is a small bias against long games, since a long game started late is more
        # likely to be killed: with ~4 games per worker about a third to a half of the draws are lost; the
        # bias roughly halves with each doubling of the games per worker.
        if config.num_workers > 1:
            run_pool = pool if pool is not None else mp.Pool(processes=config.num_workers, maxtasksperchild=10)
            game_results = []
            try:
                for result in run_pool.imap_unordered(_play_self_play_game_worker_wrapper, worker_args, chunksize=1):
                    game_results.append(result)
                    # Progress indicator at 25%, 50%, and 75% completion
                    if len(game_results) in (config.num_games_per_iteration // 4, config.num_games_per_iteration // 2,
                                             3 * config.num_games_per_iteration // 4):
                        print(f"  Progress: {len(game_results)}/{config.num_games_per_iteration} games")
                    if len(game_results) == config.num_games_per_iteration:
                        print(f"  Reached {config.num_games_per_iteration} games, terminating remaining workers...")
                        break
            except Exception as e:
                print(f"\nError during self-play: {e}", file=sys.stderr, flush=True)
                raise
            finally:
                # Kill the games still running. The pool cannot be used afterwards (the caller recreates it).
                run_pool.terminate()
                run_pool.join()
                # Collect the dead workers' file descriptors before the next pool is created
                import gc
                gc.collect()
        else:
            game_results = [_play_self_play_game_worker(*args) for args in worker_args[:config.num_games_per_iteration]]
    finally:
        # Clean up temporary file
        if cleanup_temp_file:
            try:
                os.unlink(temp_network_path)
            except Exception:
                pass
    
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

    # Training positions of all games
    position_states, position_policies, position_values, position_winners = [], [], [], []
    position_endings, position_moves_left = [], []
    for game_data in game_results:
        winner = game_data['winner']

        # Aggregate MCTS timing data
        if 'timing' in game_data:
            for key, value in game_data['timing'].items():
                if key in mcts_timing_totals:
                    mcts_timing_totals[key] += value
        
        # Get mate-in-1 flags if present
        mate_in_1_flags = game_data.get('mate_in_1_flags', [False] * len(game_data['states']))
        
        # Compute time-weighted discount factors for each position
        # discount[i] = reward_discount^(num_moves - 1 - i) where i is position index
        # Last move gets discount=1.0, earlier moves get increasingly discounted
        num_positions = len(game_data['states'])
        gamma = config.reward_discount
        discount_factors = [gamma ** (num_positions - 1 - i) for i in range(num_positions)]
        
        # Add to buffer
        for i, (state, policy, player, is_mate_in_1) in enumerate(zip(game_data['states'], 
                                                         game_data['policies'], 
                                                         game_data['players'],
                                                         mate_in_1_flags)):
            # Determine base value from player's perspective
            if is_mate_in_1:
                # Mate-in-1 position: always value +1.0 for the player to move (no discount)
                value = 1.0
            elif winner is None:
                # Draw (move limit, or repetition where the rules make it a draw)
                # Apply player-specific draw penalty
                if player == 0:  # Attacker
                    base_value = config.draw_penalty_attacker
                else:  # Defender (player == 1)
                    base_value = config.draw_penalty_defender
                # Apply time-weighted discount
                value = base_value * discount_factors[i]
            elif winner == player:
                # Apply time-weighted discount to winning value
                value = 1.0 * discount_factors[i]
            elif winner == 1 - player:
                # Apply time-weighted discount to losing value
                value = -1.0 * discount_factors[i]
            else:
                value = 0.0  # Shouldn't happen

            position_states.append(state)
            position_policies.append(policy)
            position_values.append(value)
            position_winners.append(-1 if winner is None else winner)  # -1 = draw
        position_endings += [ENDINGS.index(game_data['record']['ending'])] * num_positions
        position_moves_left += [game_data['num_moves'] - ply for ply in game_data['plies']]
    positions = (np.array(position_states, dtype=np.uint8),
                 np.array(position_policies, dtype=np.float32),
                 np.array(position_values, dtype=np.float32),
                 np.array(position_winners, dtype=np.int8),
                 np.array(position_endings, dtype=np.int64),
                 np.array(position_moves_left, dtype=np.float32))

    games = [game_data['record'] for game_data in game_results]
    total_games = len(games)
    attacker_wins = sum(record['winner'] == 'attackers' for record in games)
    defender_wins = sum(record['winner'] == 'defenders' for record in games)
    draws = total_games - attacker_wins - defender_wins
    print(f"Generated {len(position_values)} training positions")
    print(f"Results: {attacker_wins} attacker wins ({100*attacker_wins/total_games:.1f}%), "
          f"{defender_wins} defender wins ({100*defender_wins/total_games:.1f}%), "
          f"{draws} draws ({100*draws/total_games:.1f}%)")
    print(f"Average game length: {np.mean([record['plies'] for record in games]):.1f} moves")
    print("How the games ended:")
    print_endings({'self-play games': games}, config.rules)
    
    # Analyze unique states if tracking is enabled
    unique_states = None
    if config.track_unique_states:
        # Collect all state hashes from all games
        all_state_hashes = []
        all_state_hashes_skip_initial = []
        all_state_hashes_skip_6ply = []
        
        for game_data in game_results:
            if 'state_hashes' in game_data:
                hashes = game_data['state_hashes']
                if len(hashes) > 0:
                    all_state_hashes.extend(hashes)
                    # Skip first state (initial position)
                    if len(hashes) > 1:
                        all_state_hashes_skip_initial.extend(hashes[1:])
                    # Skip first 6 ply (3 moves per side)
                    if len(hashes) > 6:
                        all_state_hashes_skip_6ply.extend(hashes[6:])
        
        # Calculate unique state statistics efficiently using sets
        total_states = len(all_state_hashes)
        unique_all = len(set(all_state_hashes))
        unique_skip_initial = len(set(all_state_hashes_skip_initial))
        unique_skip_6ply = len(set(all_state_hashes_skip_6ply))
        unique_states = {'total': total_states, 'unique': unique_all,
                         'total_skip_initial': len(all_state_hashes_skip_initial),
                         'unique_skip_initial': unique_skip_initial,
                         'total_skip_6ply': len(all_state_hashes_skip_6ply),
                         'unique_skip_6ply': unique_skip_6ply}

        print("\n--- Board State Diversity Analysis ---")
        if total_states > 0:
            print(f"Total states encountered: {total_states}")
            print(f"Unique states (all): {unique_all} ({100*unique_all/total_states:.2f}%)")
            
            total_skip_initial = len(all_state_hashes_skip_initial)
            if total_skip_initial > 0:
                print(f"Unique states (skip initial): {unique_skip_initial} / {total_skip_initial} ({100*unique_skip_initial/total_skip_initial:.2f}%)")
            
            total_skip_6ply = len(all_state_hashes_skip_6ply)
            if total_skip_6ply > 0:
                print(f"Unique states (skip 6 ply): {unique_skip_6ply} / {total_skip_6ply} ({100*unique_skip_6ply/total_skip_6ply:.2f}%)")
        else:
            print("No state hashes collected")
    
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

    return positions, games, {'unique_states': unique_states, 'mcts_timing': mcts_timing_totals}


# =============================================================================
# TRAINING
# =============================================================================

# The keys of train_network's result, which are also lists in the training history
LOSS_KEYS = ['policy_loss', 'value_loss', 'ending_loss', 'moves_left_loss', 'l2_loss', 'total_loss',
             'policy_grad', 'value_grad',
             'val_policy_loss', 'val_value_loss', 'val_ending_loss', 'val_moves_left_loss',
             'val_l2_loss', 'val_total_loss', 'epochs']


def train_network(network: nn.Module, buffer: ReplayBuffer,
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
        dict (keys LOSS_KEYS) with the training losses '<name>_loss' (name: policy, value, ending,
        moves_left, total) averaged over all batches, 'policy_grad', 'value_grad', 'l2_loss', the
        validation losses 'val_<name>_loss' and 'val_l2_loss' of the last epoch, and 'epochs': the
        same keys, each with a list of the per-epoch values (the printed table)
    """
    # Compile network for faster training (~18% speedup)
    # This creates a compiled wrapper that's only used during training. The default dynamic=None
    # recompiles once when the batch size changes (the last validation batch) and then keeps only
    # the batch size variable. dynamic=True makes the board size variable too, which the compiler
    # cannot handle for the attention blocks.
    network_compiled = torch.compile(network, mode='default')
    network_compiled.train()
    device = config.device
    
    # Generator.choice without replacement costs O(batch); the legacy np.random.choice shuffles all
    # indices on every call (~100 ms per batch for a 6M-sample buffer).
    rng = np.random.default_rng()
    num_symmetries = buffer.num_symmetries
    cell_perm, move_perm = symmetry_permutations(config.board_size)

    def get_batch(indices, symmetries):
        """Tensors (states, policies, values, winners, endings, moves_left) of buffer entries
        `indices`, each entry turned by its board symmetry (an index into cell_perm / move_perm)."""
        # Flatten the board, so that a symmetry reorders the last axis
        states = buffer.states[indices].reshape(len(indices), buffer.states.shape[1], -1)
        policies = buffer.policies[indices]
        # Turn the rows that drew symmetry k, for each k (faster than one gather per row)
        for k in range(num_symmetries):
            rows = symmetries == k
            states[rows] = states[rows][:, :, cell_perm[k]]
            policies[rows] = policies[rows][:, move_perm[k]]
        states = states.reshape((len(indices),) + buffer.states.shape[1:]).astype(np.float32)
        return (torch.from_numpy(states).to(device),
                torch.from_numpy(policies).to(device),
                torch.from_numpy(buffer.values[indices]).to(device),
                torch.from_numpy(buffer.winners[indices]).to(device),
                torch.from_numpy(buffer.endings[indices]).to(device),
                torch.from_numpy(buffer.moves_left[indices]).to(device))

    # Training: batch_size * batches_per_epoch samples per epoch (fewer if the buffer is smaller).
    # Validation: 0.25x that volume, from the newest positions in the buffer, each under all its
    # symmetries. They are left out of training in this call (and trained on from the next iteration
    # on). Positions are stored game by game, so only the first held-out game is split.
    total_samples = len(buffer)
    train_samples_per_epoch = min(total_samples, config.batch_size * config.batches_per_epoch)
    num_val_positions = train_samples_per_epoch // 4 // num_symmetries
    capacity = len(buffer.values)
    oldest = buffer.next_index if buffer.num_positions == capacity else 0
    positions_by_age = (oldest + np.arange(buffer.num_positions)) % capacity  # oldest first
    train_positions = positions_by_age[:buffer.num_positions - num_val_positions]
    val_positions = positions_by_age[buffer.num_positions - num_val_positions:]
    val_indices = np.repeat(val_positions, num_symmetries)
    val_symmetries = np.tile(np.arange(num_symmetries), num_val_positions)

    batches_per_epoch = min(config.batches_per_epoch, total_samples // config.batch_size)
    if len(train_positions) < config.batch_size or batches_per_epoch == 0:
        raise ValueError(f"Replay buffer too small for training: {len(train_positions)} positions "
                         f"outside the validation set, batch size {config.batch_size}. "
                         "Increase min_buffer_size.")

    # Endings the rules do not allow get no probability in the ending output
    allowed_endings = torch.tensor([ending in config.rules.endings() for ending in ENDINGS],
                                   device=device)
    loss_names = ['policy', 'value', 'ending', 'moves_left', 'total']

    def batch_losses(batch) -> Dict[str, torch.Tensor]:
        """The mean losses (loss_names) of one batch from get_batch."""
        states, policies, values, winners, endings, moves_left = batch
        pred_policies, pred_values, pred_endings, pred_moves_left = network_compiled(states)
        policy = -torch.sum(policies * torch.log_softmax(pred_policies, dim=1), dim=1)
        value = (pred_values.squeeze(1) - values) ** 2
        # Boost factors: attacker-won games get attacker_boost, defender-won games defender_boost,
        # draws 1, divided by the mean weight to keep the effective learning rate stable
        if attacker_boost != 1.0 or defender_boost != 1.0:
            weights = torch.where(winners == 0, attacker_boost,
                                  torch.where(winners == 1, defender_boost, 1.0))
            policy, value = policy * weights / weights.mean(), value * weights / weights.mean()
        losses = {
            'policy': policy.mean(),
            'value': value.mean(),
            'ending': F.cross_entropy(pred_endings.masked_fill(~allowed_endings, float('-inf')),
                                      endings),
            # In units of MOVES_LEFT_SCALE plies: the squared error up to one unit, linear beyond
            # it, so that the rare very long games do not dominate
            'moves_left': F.huber_loss(pred_moves_left.squeeze(1) / MOVES_LEFT_SCALE,
                                       moves_left / MOVES_LEFT_SCALE),
        }
        losses['total'] = (losses['policy'] + config.value_loss_weight * losses['value']
                           + config.ending_loss_weight * losses['ending']
                           + config.moves_left_loss_weight * losses['moves_left'])
        return losses

    totals = dict.fromkeys(loss_names + ['policy_grad', 'value_grad'], 0.0)  # sums over all batches
    num_batches = 0
    epochs = {key: [] for key in LOSS_KEYS if key != 'epochs'}  # one value per epoch for each key

    # Print header for training output
    header = " ".join(f"{name:>9}"
                      for name in ['Policy', 'Value', 'Ending', 'MovesLeft', 'L2', 'Total'])
    print("\n  " + "-" * 149)
    print(f"  {'':>5} | {'Training losses':^59} | {'Gradients':^17} | {'Validation losses':^59}")
    print(f"  {'Epoch':>5} | {header} | {'Policy':>8} {'Value':>8} | {header}")
    print("  " + "-" * 149)

    for epoch in range(config.num_epochs):
        epoch_sums = dict.fromkeys(totals, 0.0)
        for batch_idx in range(batches_per_epoch):
            # Sample positions randomly from the buffer, without the validation positions, each
            # under a random board symmetry
            chosen = rng.choice(len(train_positions), config.batch_size, replace=False)
            symmetries = rng.integers(0, num_symmetries, config.batch_size)
            losses = batch_losses(get_batch(train_positions[chosen], symmetries))

            # Backward pass (set_to_none=True is faster than default zero_grad)
            optimizer.zero_grad(set_to_none=True)
            losses['total'].backward()

            # Compute gradient norms after backward pass, separately for policy and value heads.
            # Squared norms are summed as tensors so that only two .item() syncs happen per batch.
            policy_grad_sq = torch.zeros((), device=device)
            value_grad_sq = torch.zeros((), device=device)

            for name, param in network.named_parameters():
                if param.grad is not None:
                    grad_sq = param.grad.pow(2).sum()
                    # Policy head has 'policy' in the name, value head has 'value'
                    if 'policy' in name.lower():
                        policy_grad_sq += grad_sq
                    elif 'value' in name.lower():
                        value_grad_sq += grad_sq
                    elif name.startswith(('ending_head.', 'moves_left_head.')):
                        continue  # the extra outputs are in neither norm
                    else:
                        # Shared layers contribute to both
                        policy_grad_sq += grad_sq / 2
                        value_grad_sq += grad_sq / 2

            batch_values = {name: losses[name].item() for name in loss_names}
            batch_values['policy_grad'] = policy_grad_sq.item() ** 0.5
            batch_values['value_grad'] = value_grad_sq.item() ** 0.5

            # Stop on NaN or infinity instead of writing it into the weights (and all later
            # checkpoints)
            if not np.isfinite(sum(batch_values.values())):
                raise RuntimeError(f"Non-finite loss or gradient in epoch {epoch + 1}, batch "
                                   f"{batch_idx + 1}: {batch_values}")

            optimizer.step()
            for key, value in batch_values.items():
                epoch_sums[key] += value
            num_batches += 1
        for key in totals:
            totals[key] += epoch_sums[key]

        # The L2 penalty of the weights after this epoch, for reporting (AdamW applies the decay)
        l2_loss = 0.5 * config.weight_decay * sum(
            torch.sum(param ** 2).item() for param in network.parameters() if param.requires_grad)

        # Validation pass for this epoch - evaluate on validation set without gradients
        network_compiled.eval()
        val_sums = dict.fromkeys(loss_names, 0.0)
        with torch.no_grad():
            # Process the validation samples in batches (the last one may be smaller)
            for start in range(0, len(val_indices), config.batch_size):
                batch = slice(start, start + config.batch_size)
                batch_indices = val_indices[batch]
                losses = batch_losses(get_batch(batch_indices, val_symmetries[batch]))
                # Sum the batch means weighted by batch size, for the mean over all validation
                # samples
                for name in loss_names:
                    val_sums[name] += losses[name].item() * len(batch_indices)
        network_compiled.train()

        # Mean validation losses (NaN if the buffer is too small for a validation set)
        val = {name: val_sums[name] / len(val_indices) if len(val_indices) > 0 else float('nan')
               for name in loss_names}

        # Print per-epoch losses in columnar format
        means = {key: value / batches_per_epoch for key, value in epoch_sums.items()}
        train_row = [means[name] for name in loss_names[:4]] + [l2_loss, means['total']]
        val_row = [val[name] for name in loss_names[:4]] + [l2_loss, val['total']]
        print(f"  {epoch + 1:>5} | " + " ".join(f"{x:>9.4f}" for x in train_row)
              + f" | {means['policy_grad']:>8.4f} {means['value_grad']:>8.4f} | "
              + " ".join(f"{x:>9.4f}" for x in val_row))
        for name in loss_names:
            epochs[f'{name}_loss'].append(means[name])
            epochs[f'val_{name}_loss'].append(val[name])
        for key, value in [('l2_loss', l2_loss), ('val_l2_loss', l2_loss),
                           ('policy_grad', means['policy_grad']), ('value_grad', means['value_grad'])]:
            epochs[key].append(value)

    print("  " + "-" * 149)

    # Training losses and gradients averaged over all batches; validation losses of the last epoch
    result = {f'{name}_loss': totals[name] / num_batches for name in loss_names}
    result.update(policy_grad=totals['policy_grad'] / num_batches,
                  value_grad=totals['value_grad'] / num_batches,
                  l2_loss=l2_loss, val_l2_loss=l2_loss)
    result.update({f'val_{name}_loss': val[name] for name in loss_names})
    result['epochs'] = epochs
    return result


# =============================================================================
# EVALUATION
# =============================================================================

def _evaluate_vs_random_worker(network_path, network_config, rules,
                                num_sims_attacker, num_sims_defender, c_puct, nn_plays_attacker, game_idx,
                                fpu_reduction=-0.5, max_moves=500, search_discount=1.0):
    """
    Worker function for parallel evaluation against random player.
    Must be at module level for multiprocessing. Imports inside to avoid pickling issues.

    Args:
        network_path: Path to saved network weights file
        network_config: NetworkConfig.to_dict() of the network
        rules: the rule set (tafl.Rules, the same as in self-play)
        num_sims_attacker: MCTS simulations per move for attacker
        num_sims_defender: MCTS simulations per move for defender
        c_puct: MCTS exploration constant
        nn_plays_attacker: Whether NN plays as attacker
        game_idx: Game index (unused, for pool.map)
        fpu_reduction: First Play Urgency reduction relative to parent Q-value (for evaluation)
        max_moves: Plies after which the game is a draw (same limit as self-play)
        search_discount: MCTS backup discount per move (same as self-play)

    Returns:
        the game_record of the game, with network_side the side the network played
    """
    # Fresh seeds from the operating system for every game (see _play_self_play_game_worker)
    np.random.seed()
    torch.seed()

    # Set torch to use only 1 thread per worker to avoid conflicts
    torch.set_num_threads(1)
    # Compile in this process; the inherited Inductor compile thread pool can hang after fork. C++
    # wrapper for faster calls (both as in _play_self_play_game_worker)
    from torch._inductor import config as inductor_config
    inductor_config.compile_threads = 1
    inductor_config.cpp_wrapper = True

    # Set per-worker compilation cache to avoid lock contention
    # Each worker gets its own cache directory, so no delays needed
    import tempfile
    worker_cache_dir = os.path.join(tempfile.gettempdir(),
                                    f'torchinductor_{socket.gethostname()}_{os.getpid()}')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = worker_cache_dir
    
    # Network architecture and input planes from the config
    net_config = NetworkConfig.from_dict(network_config)

    from agent import Agent, RandomAgent, play_game

    # Reconstruct network on CPU and load from file
    network = TaflNet(net_config, rules.board_size)
    checkpoint = torch.load(network_path, map_location='cpu', weights_only=False)
    load_weights(network, checkpoint)
    network.to('cpu')
    network.eval()

    # Optimize for CPU inference with compilation for 2.5-2.75x speedup
    # Each worker has its own cache directory, so no contention
    network = network.optimize_for_inference(use_compile=True, compile_mode='default')

    # Determine which simulations to use based on NN's role
    num_simulations = num_sims_attacker if nn_plays_attacker else num_sims_defender

    # Create agents on CPU
    nn_agent = Agent(network, num_simulations=num_simulations, c_puct=c_puct, device='cpu',
                     fpu_reduction=fpu_reduction, feature_spec=net_config.features,
                     search_discount=search_discount)
    random_agent = RandomAgent()

    # Play game
    agents = (nn_agent, random_agent) if nn_plays_attacker else (random_agent, nn_agent)
    game = play_game(*agents, rules, display=False, max_moves=max_moves)
    return game_record(game, 'vs_random', 'attackers' if nn_plays_attacker else 'defenders')


def game_score(record: dict) -> float:
    """The score of the evaluated network in an evaluation game record: 1 win, 0.5 draw, 0 loss."""
    return 0.5 if record['winner'] is None else float(record['winner'] == record['network_side'])


def calculate_elo_difference(win_rate: float) -> float:
    """
    Calculate ELO rating difference from win rate / score.
    
    ELO formula: diff = -400 * log10((1/win_rate) - 1)
    
    For network evaluation, win_rate should be the score:
        score = (wins + 0.5 * draws) / total_games
    
    Args:
        win_rate: win rate / score between 0 and 1 (where draws count as 0.5)
    
    Returns:
        ELO difference (positive means stronger)
    """
    # Clamp win rate to avoid log(0)
    win_rate = np.clip(win_rate, 0.01, 0.99)
    
    # ELO difference from win rate
    elo_diff = -400 * np.log10((1 / win_rate) - 1)
    
    return elo_diff


def evaluate_vs_random(network: nn.Module, config: TrainingConfig,
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
        dict with 'total_wins', 'total_draws', 'total_losses', 'total_games',
        'win_rate' (score rate where draws = 0.5), 'elo_diff',
        'attacker_wins', 'attacker_draws', 'defender_wins', 'defender_draws', and 'games' (the
        game records)
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
    
    # Evaluation games use the same rules and draw limit as self-play
    try:
        # Create worker functions with partial for attacker and defender games
        network_config = config.network.to_dict()
        worker_func_attacker = partial(
            _evaluate_vs_random_worker,
            temp_network_path,
            network_config,
            config.rules,
            config.eval_mcts_sims_attacker,
            config.eval_mcts_sims_defender,
            config.eval_c_puct,
            True,  # nn_plays_attacker
            fpu_reduction=config.eval_fpu_reduction,
            max_moves=config.max_game_length,
            search_discount=config.search_discount
        )

        worker_func_defender = partial(
            _evaluate_vs_random_worker,
            temp_network_path,
            network_config,
            config.rules,
            config.eval_mcts_sims_attacker,
            config.eval_mcts_sims_defender,
            config.eval_c_puct,
            False,  # nn_plays_attacker
            fpu_reduction=config.eval_fpu_reduction,
            max_moves=config.max_game_length,
            search_discount=config.search_discount
        )
        
        # Play games in parallel
        # Both colours are submitted before waiting, so all workers stay busy instead of
        # running the two halves one after the other.
        if config.num_workers > 1:
            if pool is not None:
                # Use provided persistent pool
                attacker_async = pool.map_async(worker_func_attacker, range(num_games), chunksize=1)
                defender_async = pool.map_async(worker_func_defender, range(num_games), chunksize=1)
                attacker_results, defender_results = attacker_async.get(), defender_async.get()
            else:
                # Create temporary pool (for backward compatibility)
                with mp.Pool(processes=config.num_workers, maxtasksperchild=10) as temp_pool:
                    attacker_async = temp_pool.map_async(worker_func_attacker, range(num_games), chunksize=1)
                    defender_async = temp_pool.map_async(worker_func_defender, range(num_games), chunksize=1)
                    attacker_results, defender_results = attacker_async.get(), defender_async.get()
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
    
    # Process results: the network's score per game, 1.0 = win, 0.5 = draw, 0.0 = loss
    games = attacker_results + defender_results
    attacker_scores = [game_score(record) for record in attacker_results]
    defender_scores = [game_score(record) for record in defender_results]
    attacker_wins = attacker_scores.count(1.0)
    attacker_draws = attacker_scores.count(0.5)
    defender_wins = defender_scores.count(1.0)
    defender_draws = defender_scores.count(0.5)

    total_wins = attacker_wins + defender_wins
    total_draws = attacker_draws + defender_draws
    total_games = num_games * 2
    total_losses = total_games - total_wins - total_draws

    # Calculate score (wins + 0.5 * draws) for ELO and win rate
    total_score = sum(attacker_scores) + sum(defender_scores)
    score_rate = total_score / total_games
    elo_diff = calculate_elo_difference(score_rate)

    print(f"  As Attacker: {attacker_wins}/{num_games} wins, {attacker_draws} draws")
    print(f"  As Defender: {defender_wins}/{num_games} wins, {defender_draws} draws")
    print(f"  Total: {total_wins}W-{total_losses}L-{total_draws}D ({100*score_rate:.1f}% score)")
    print(f"  ELO vs Random: {elo_diff:+.0f}")
    print("  How the games ended:")
    print_endings({'network as attackers': attacker_results,
                   'network as defenders': defender_results}, config.rules)

    return {
        'games': games,
        'total_wins': total_wins,
        'total_draws': total_draws,
        'total_losses': total_losses,
        'total_games': total_games,
        'win_rate': score_rate,  # Note: now returns score rate (with draws as 0.5)
        'elo_diff': elo_diff,
        'attacker_wins': attacker_wins,
        'attacker_draws': attacker_draws,
        'defender_wins': defender_wins,
        'defender_draws': defender_draws
    }


def _evaluate_networks_worker(new_network_path, old_network_path,
                             network_config, rules, num_sims_attacker, num_sims_defender, c_puct,
                             new_plays_attacker,
                             game_idx, temperature, temperature_mode, temperature_threshold, temperature_decay_moves, fpu_reduction, temp_dir=None,
                             max_moves=500, search_discount=1.0):
    """
    Worker function for parallel network evaluation.
    Must be at module level for multiprocessing. Imports inside to avoid pickling issues.

    Args:
        new_network_path: Path to saved new network weights file
        old_network_path: Path to saved old network weights file
        network_config: NetworkConfig.to_dict() of both networks
        rules: the rule set (tafl.Rules, the same as in self-play)
        num_sims_attacker: MCTS simulations per move for attacker
        num_sims_defender: MCTS simulations per move for defender
        c_puct: MCTS exploration constant
        new_plays_attacker: Whether new network plays as attacker
        game_idx: Game index (unused, for pool.map)
        temperature: Sampling temperature for move selection
        temperature_mode: "fixed", "king", or "decay"
        temperature_threshold: Move number for temperature drop (for "fixed" mode)
        temperature_decay_moves: Number of moves for linear decay (for "decay" mode)
        fpu_reduction: First Play Urgency reduction relative to parent Q-value (for evaluation)
        max_moves: Plies after which the game is a draw (same limit as self-play)
        search_discount: MCTS backup discount per move (same as self-play)

    Returns:
        the game_record of the game, with network_side the side the new network played
    """
    # Fresh seeds from the operating system for every game (see _play_self_play_game_worker)
    np.random.seed()
    torch.seed()

    # Set torch to use only 1 thread per worker to avoid conflicts
    torch.set_num_threads(1)
    # Compile in this process; the inherited Inductor compile thread pool can hang after fork. C++
    # wrapper for faster calls (both as in _play_self_play_game_worker)
    from torch._inductor import config as inductor_config
    inductor_config.compile_threads = 1
    inductor_config.cpp_wrapper = True

    # Set up temp directory for this worker
    import tempfile
    if temp_dir is not None:
        # Use specified temp directory (e.g., RAM disk)
        os.environ['TMPDIR'] = temp_dir
        os.environ['TEMP'] = temp_dir
        os.environ['TMP'] = temp_dir
        base_temp_dir = temp_dir
    else:
        base_temp_dir = tempfile.gettempdir()
    
    # Set per-worker compilation cache to avoid lock contention
    # Each worker gets its own cache directory
    worker_cache_dir = os.path.join(base_temp_dir,
                                    f'torchinductor_{socket.gethostname()}_{os.getpid()}')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = worker_cache_dir
    
    # Network architecture and input planes from the config
    net_config = NetworkConfig.from_dict(network_config)

    from agent import Agent, play_game

    # Reconstruct networks on CPU and load from files
    new_network = TaflNet(net_config, rules.board_size)
    checkpoint = torch.load(new_network_path, map_location='cpu', weights_only=False)
    load_weights(new_network, checkpoint)
    new_network.to('cpu')
    new_network.eval()

    # Optimize for CPU inference with compilation for 2.5-2.75x speedup
    # Each worker has its own cache directory, so no contention
    new_network = new_network.optimize_for_inference(use_compile=True, compile_mode='default')

    old_network = TaflNet(net_config, rules.board_size)
    checkpoint = torch.load(old_network_path, map_location='cpu', weights_only=False)
    load_weights(old_network, checkpoint)
    old_network.to('cpu')
    old_network.eval()

    # Optimize for CPU inference with compilation for 2.5-2.75x speedup
    old_network = old_network.optimize_for_inference(use_compile=True, compile_mode='default')

    # Determine simulations based on roles
    # Both agents use role-appropriate simulation counts
    new_sims = num_sims_attacker if new_plays_attacker else num_sims_defender
    old_sims = num_sims_defender if new_plays_attacker else num_sims_attacker

    # Create agents on CPU, without Dirichlet noise: the match measures the real players, and the
    # evaluation temperature already varies the openings.
    new_agent = Agent(new_network, num_simulations=new_sims, c_puct=c_puct, device='cpu',
                      fpu_reduction=fpu_reduction, feature_spec=net_config.features,
                      search_discount=search_discount)
    old_agent = Agent(old_network, num_simulations=old_sims, c_puct=c_puct, device='cpu',
                      fpu_reduction=fpu_reduction, feature_spec=net_config.features,
                      search_discount=search_discount)

    # Play game
    agents = (new_agent, old_agent) if new_plays_attacker else (old_agent, new_agent)
    game = play_game(*agents, rules, display=False, temperature=temperature,
                     temperature_mode=temperature_mode, temperature_threshold=temperature_threshold,
                     temperature_decay_moves=temperature_decay_moves, max_moves=max_moves)
    return game_record(game, 'vs_best', 'attackers' if new_plays_attacker else 'defenders')


def evaluate_networks(new_network: nn.Module, old_network: nn.Module,
                     config: TrainingConfig, pool=None, temp_new_path=None,
                     temp_old_path=None) -> Tuple[float, List[dict]]:
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
        score_rate: score as fraction (wins + 0.5 * draws) / total_games
        games: the game records
    """
    print(f"\nEvaluating new network vs old network ({config.eval_games} games using {config.num_workers} workers)...")
    
    # Save networks to temporary files to avoid pickling issues with many workers
    import tempfile
    
    # Use configured temp directory if specified
    if config.temp_dir is not None:
        temp_base_dir = config.temp_dir
    else:
        temp_base_dir = tempfile.gettempdir()
    
    if temp_new_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_new.pth', delete=False, dir=temp_base_dir)
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
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_old.pth', delete=False, dir=temp_base_dir)
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

        # Evaluation games use the same rules and draw limit as self-play
        network_config = config.network.to_dict()
        worker_func_new_attacker = partial(
            _evaluate_networks_worker,
            temp_new_path,
            temp_old_path,
            network_config,
            config.rules,
            config.eval_mcts_sims_attacker,
            config.eval_mcts_sims_defender,
            config.eval_c_puct,
            True,  # new_plays_attacker
            # game_idx is provided by map()
            # temperature and mode parameters below
            temperature=config.eval_temperature,
            temperature_mode=config.eval_temperature_mode,
            temperature_threshold=config.eval_temperature_threshold,
            temperature_decay_moves=config.eval_temperature_decay_moves,
            fpu_reduction=config.eval_fpu_reduction,
            temp_dir=config.temp_dir,
            max_moves=config.max_game_length,
            search_discount=config.search_discount
        )

        worker_func_new_defender = partial(
            _evaluate_networks_worker,
            temp_new_path,
            temp_old_path,
            network_config,
            config.rules,
            config.eval_mcts_sims_attacker,
            config.eval_mcts_sims_defender,
            config.eval_c_puct,
            False,  # new_plays_attacker
            # game_idx is provided by map()
            # temperature and mode parameters below
            temperature=config.eval_temperature,
            temperature_mode=config.eval_temperature_mode,
            temperature_threshold=config.eval_temperature_threshold,
            temperature_decay_moves=config.eval_temperature_decay_moves,
            fpu_reduction=config.eval_fpu_reduction,
            temp_dir=config.temp_dir,
            max_moves=config.max_game_length,
            search_discount=config.search_discount
        )

        # Play games in parallel. Both colours are submitted before waiting, so all workers stay
        # busy instead of running the two halves one after the other.
        if config.num_workers > 1:
            if pool is not None:
                # Use provided persistent pool
                attacker_async = pool.map_async(worker_func_new_attacker, range(half_games + remainder), chunksize=1)
                defender_async = pool.map_async(worker_func_new_defender, range(half_games), chunksize=1)
                results = attacker_async.get() + defender_async.get()
            else:
                # Create temporary pool (for backward compatibility)
                with mp.Pool(processes=config.num_workers, maxtasksperchild=10) as temp_pool:
                    attacker_async = temp_pool.map_async(worker_func_new_attacker, range(half_games + remainder), chunksize=1)
                    defender_async = temp_pool.map_async(worker_func_new_defender, range(half_games), chunksize=1)
                    results = attacker_async.get() + defender_async.get()
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
    
    # Process results - split by attacker/defender. Scores: 1.0 = win, 0.5 = draw, 0.0 = loss
    attacker_results = results[:half_games + remainder]
    defender_results = results[half_games + remainder:]
    attacker_scores = [game_score(record) for record in attacker_results]
    defender_scores = [game_score(record) for record in defender_results]
    scores = attacker_scores + defender_scores

    # Count wins and draws separately
    attacker_wins = attacker_scores.count(1.0)
    attacker_draws = attacker_scores.count(0.5)
    defender_wins = defender_scores.count(1.0)
    defender_draws = defender_scores.count(0.5)

    total_wins = attacker_wins + defender_wins
    total_draws = attacker_draws + defender_draws
    total_losses = config.eval_games - total_wins - total_draws

    # Calculate score (wins + 0.5 * draws) for ELO
    total_score = sum(scores)
    score_rate = total_score / config.eval_games

    print(f"  As Attacker: {attacker_wins}/{half_games + remainder} wins, {attacker_draws} draws")
    print(f"  As Defender: {defender_wins}/{half_games} wins, {defender_draws} draws")
    print(f"New network results: {total_wins}W-{total_losses}L-{total_draws}D ({100*score_rate:.1f}% score)")
    print("  How the games ended:")
    print_endings({'new as attackers': attacker_results, 'new as defenders': defender_results},
                  config.rules)

    return score_rate, results


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


def save_checkpoint(network: nn.Module, optimizer: optim.Optimizer,
                   iteration: int, config: TrainingConfig,
                   metrics: Dict, filename: str = None):
    """Save model checkpoint. `metrics` is the training history so far (used on resume)."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_iter_{iteration}.pth"

    filepath = os.path.join(config.checkpoint_dir, filename)

    checkpoint = {
        'iteration': iteration,
        'model_state_dict': clean_state_dict(network.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        # All settings as one flat dict (incl. the old network keys, for older tools), the network
        # architecture (used to rebuild the network from the checkpoint) and the JSON run config
        'config': config.to_flat_dict(),
        'network_config': config.network.to_dict(),
        'run_config': config.to_dict(),
    }

    # Write to a temporary file and rename it, so a job killed while saving never leaves a broken
    # file
    torch.save(checkpoint, filepath + '.tmp')
    os.replace(filepath + '.tmp', filepath)
    print(f"Saved checkpoint: {filepath}")


def load_checkpoint(filepath: str, network: nn.Module,
                   optimizer: optim.Optimizer = None) -> Tuple[int, Dict]:
    """Load model checkpoint. Returns the iteration and the training history saved with it."""
    checkpoint = torch.load(filepath, weights_only=False)

    load_weights(network, checkpoint)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    iteration = checkpoint.get('iteration', 0)
    print(f"Loaded checkpoint from iteration {iteration}")

    return iteration, checkpoint.get('metrics')


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(config: TrainingConfig, resume_from: str = None, load_weights_from: str = None):
    """
    Main training loop.
    
    Args:
        config: TrainingConfig object
        resume_from: path to checkpoint to resume from (loads weights, optimizer, history)
        load_weights_from: path to checkpoint to load only weights from (starts fresh otherwise)
    """
    # Validate that resume_from and load_weights_from are not both specified
    if resume_from is not None and load_weights_from is not None:
        raise ValueError("Cannot specify both --resume and --load-weights. Use one or the other.")
    
    # Enable denormal (subnormal) flushing to zero for faster inference
    # This prevents tiny weights from causing 10-100x inference slowdown
    # Denormal numbers are extremely slow on CPUs, and flushing them to zero
    # has negligible impact on training quality but massive performance benefit
    torch.set_flush_denormal(True)
    print("Enabled denormal flushing (torch.set_flush_denormal(True)) for fast inference")
    
    # Suppress NumPy warnings about subnormals being zero (expected behavior)
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')
    
    # Set up temporary directory for PyTorch operations
    temp_dir_path = setup_temp_directory(config.temp_dir)
    if config.temp_dir is not None:
        print(f"Using custom temp directory: {temp_dir_path}")
    
    # Validate the network config (from_dict already checked the rule set name)
    config.network.validate()

    # Write the resolved configuration next to the checkpoints, so that a run's settings can be read
    # back later. A resumed run with other settings keeps the earlier file and adds a dated one.
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    config_path = os.path.join(config.checkpoint_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            previous = json.load(f)
        if previous != config.to_dict():
            config_path = os.path.join(config.checkpoint_dir,
                                       f"run_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            print(f"Settings differ from the saved run_config.json; writing them to {config_path}")
    save_config(config, config_path)

    print("=" * 70)
    print(f"Tafl AlphaZero Training ({config.ruleset})" + (f" - {config.name}" if config.name else ""))
    print("=" * 70)
    
    # Training configuration
    print("\n--- Training Configuration ---")
    print(f"Device: {config.device}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Games per iteration: {config.num_games_per_iteration}")
    print(f"Parallel workers: {config.num_workers}")
    
    # Game rules
    print("\n--- Game Rules ---")
    print(config.rules.describe())
    print(f"Draw after {config.max_game_length} plies")
    
    # Network architecture
    print("\n--- Network Architecture ---")
    print(f"Network: {config.network.describe()}")
    print(f"Blocks: {config.network.blocks}")
    print(f"Input: {config.network.features.num_planes} planes ({config.board_size}×{config.board_size}): "
          f"{', '.join(config.network.features.plane_names())}")
    print(f"Policy output: {config.policy_size} moves ({config.num_squares} squares × 4 directions × {config.board_size - 1} distances)")
    print("Value output: single scalar (win probability)")
    print(f"Extra outputs: how the game ends ({', '.join(config.rules.endings())}; loss weight "
          f"{config.ending_loss_weight}), plies left (loss weight {config.moves_left_loss_weight})")
    
    # MCTS parameters
    print("\n--- MCTS Parameters ---")
    print("Self-play simulations:")
    print(f"  Attacker: {config.num_mcts_sims_attacker}")
    print(f"  Defender: {config.num_mcts_sims_defender}")
    print("Evaluation simulations:")
    print(f"  Attacker: {config.eval_mcts_sims_attacker}")
    print(f"  Defender: {config.eval_mcts_sims_defender}")
    print(f"Exploration constant (c_puct):")
    print(f"  Self-play: {config.c_puct}")
    print(f"  Evaluation: {config.eval_c_puct}")
    print(f"First Play Urgency (FPU) reduction:")
    print(f"  Self-play: {config.fpu_reduction}")
    print(f"  Evaluation: {config.eval_fpu_reduction}")
    print(f"Temperature: {config.temperature}")
    print(f"Temperature mode: {config.temperature_mode}")
    if config.temperature_mode == "king":
        print(f"  Temperature drops when king leaves throne")
    elif config.temperature_mode == "decay":
        print(f"  Linear decay over {config.temperature_decay_moves} moves")
    else:  # "fixed"
        print(f"  Temperature drops at move {config.temperature_threshold}")
    
    print(f"Evaluation temperature: {config.eval_temperature}")
    print(f"Evaluation temperature mode: {config.eval_temperature_mode}")
    if config.eval_temperature_mode == "king":
        print(f"  Temperature drops when king leaves throne")
    elif config.eval_temperature_mode == "decay":
        print(f"  Linear decay over {config.eval_temperature_decay_moves} moves")
    else:  # "fixed"
        print(f"  Temperature drops at move {config.eval_temperature_threshold}")
    
    print(f"Dirichlet noise: {'enabled' if config.add_dirichlet_noise else 'disabled'}")
    if config.add_dirichlet_noise:
        print(f"  Alpha (concentration): {config.dirichlet_alpha}")
        print(f"  Epsilon (weight): {config.dirichlet_epsilon}")
    
    # Training parameters
    print("\n--- Training Parameters ---")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs per iteration: {config.num_epochs}")
    print(f"Batches per epoch: {config.batches_per_epoch}")
    print(f"Learning rate: {config.learning_rate}")
    if config.use_stepping_lr:
        print(f"Using stepping LR schedule:")
        print(f"  Step iterations: {config.lr_step_iterations}")
        print(f"  LR values: {config.lr_step_values}")
    else:
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
    network = TaflNet(config.network, config.board_size).to(config.device)

    # Calculate and display network size
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Network initialized: {describe_network(network)} ({trainable_params:,} trainable)")
    
    # Initialize optimizer with fused=True for faster CPU training
    # Note: fused AdamW is faster but only available on CPU/CUDA, not all devices
    try:
        optimizer = optim.AdamW(network.parameters(), 
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay,
                               fused=True)
        print("Using fused AdamW optimizer for better training performance")
    except Exception:
        # Fall back to standard AdamW if fused not available
        optimizer = optim.AdamW(network.parameters(), 
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
        print("Using standard AdamW optimizer (fused not available)")
    
    # Learning rate scheduler
    if config.use_stepping_lr:
        # Validate stepping LR configuration
        if len(config.lr_step_values) != len(config.lr_step_iterations) + 1:
            raise ValueError(
                f"lr_step_values must have exactly len(lr_step_iterations) + 1 elements. "
                f"Got {len(config.lr_step_values)} values for {len(config.lr_step_iterations)} step points."
            )
        # The LR is set at the start of each training step in the loop below (no scheduler)
        scheduler = None
        print(f"Using stepping LR schedule:")
        print(f"  Step iterations: {config.lr_step_iterations}")
        print(f"  LR values: {config.lr_step_values}")
    else:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    saved_history = None
    if resume_from is not None:
        start_iteration, saved_history = load_checkpoint(resume_from, network, optimizer)
        print("Note: the replay buffer is not saved; it starts empty and training waits until it "
              f"holds {config.min_buffer_size:,} samples")
    elif load_weights_from is not None:
        # Load only weights, not optimizer state or training history
        print(f"Loading weights from {load_weights_from}")
        checkpoint = torch.load(load_weights_from, weights_only=False)
        load_weights(network, checkpoint)
        print(f"Loaded weights from checkpoint (starting training from iteration 0)")

    # Note: We do NOT compile the network here because:
    # 1. Compiled networks can't be pickled for multiprocessing
    # 2. We only need compilation speedup during training phase
    # We'll compile a separate copy for training only

    # Initialize best network (for evaluation)
    best_network = TaflNet(config.network, config.board_size).to(config.device)

    # Load best network from best_model.pth if resuming and it exists, otherwise copy current network
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    if resume_from is not None and os.path.exists(best_model_path):
        print(f"Loading best network from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=False)
        load_weights(best_network, checkpoint)
    else:
        # Starting fresh, loading weights only, or no best model exists yet
        # Clean state dict to remove _orig_mod. prefix from compiled network
        best_network.load_state_dict(clean_state_dict(network.state_dict()))
        if load_weights_from is not None:
            print("Initialized best network with loaded weights")
    
    # Initialize replay buffer (each position is stored once and stands for its 8 symmetries)
    state_shape = (config.network.features.num_planes, config.board_size, config.board_size)
    replay_buffer = ReplayBuffer(config.replay_buffer_size, state_shape, config.policy_size,
                                 8 if config.use_data_augmentation else 1)
    
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
    
    # Initialize simulation balancer for dynamic MCTS simulation adjustment
    if config.use_dynamic_sim_balancing:
        sim_balancer = SimulationBalancer(
            initial_attacker_sims=config.num_mcts_sims_attacker,
            initial_defender_sims=config.num_mcts_sims_defender,
            adjustment_rate=config.dynamic_sim_adjustment_rate,
            min_win_rate=config.dynamic_sim_min_win_rate
        )
        print(f"Using dynamic simulation balancing (adjustment_rate={config.dynamic_sim_adjustment_rate}, "
              f"total_sims={config.num_mcts_sims_attacker + config.num_mcts_sims_defender}, "
              f"attacker score kept in [{config.dynamic_sim_min_win_rate}, {1 - config.dynamic_sim_min_win_rate}])")
    else:
        sim_balancer = None
        print(f"Using fixed simulation counts (attacker={config.num_mcts_sims_attacker}, "
              f"defender={config.num_mcts_sims_defender})")
    
    # ELO tracking - cumulative ELO relative to initial random network
    cumulative_elo = 0.0  # Start at 0 (random network baseline)
    
    # Track if 100% win rate vs random has been achieved
    achieved_100_percent_vs_random = False
    
    # Load or initialize training history. On resume, the history saved in the checkpoint is used,
    # so that resuming an older checkpoint does not pick up the balancer state etc. of later
    # iterations from training_history.json (the JSON file is overwritten from the next iteration
    # on).
    history_path = os.path.join(config.checkpoint_dir, "training_history.json")
    if resume_from is not None and (saved_history is not None or os.path.exists(history_path)):
        if saved_history is not None:
            print(f"Using the training history saved in {resume_from}")
            training_history = saved_history
        else:
            print(f"Loading training history from {history_path}")
            with open(history_path, 'r') as f:
                training_history = json.load(f)

        # Resume cumulative ELO if available
        if training_history.get('cumulative_elo'):
            cumulative_elo = training_history['cumulative_elo'][-1]
            print(f"Resuming with cumulative ELO: {cumulative_elo:+.1f}")
        
        # Check if 100% win rate was already achieved
        if training_history.get('vs_random_win_rate'):
            last_win_rate = training_history['vs_random_win_rate'][-1]
            if last_win_rate >= 1.0:
                achieved_100_percent_vs_random = True
                print("Note: 100% win rate vs random already achieved")

        # Resume the simulation balancer and the win-rate tracker where they were
        if sim_balancer is not None and training_history.get('sim_counts'):
            sim_balancer.attacker_sims, sim_balancer.defender_sims = training_history['sim_counts'][-1]
            print(f"Resuming with simulation counts: Attacker {sim_balancer.attacker_sims}, "
                  f"Defender {sim_balancer.defender_sims}")
        if win_rate_tracker is not None and training_history.get('win_rate_smoothed'):
            win_rate_tracker.attacker_wins_smooth, win_rate_tracker.defender_wins_smooth = \
                training_history['win_rate_smoothed'][-1]

        # Older histories do not record at which iterations the evaluations ran. They ran every
        # eval_frequency (eval_vs_random_frequency) iterations of the run's start settings, from
        # iteration 1 on, which gives these iterations unless the frequency was changed on a resume.
        eval_params = training_history['metadata']['evaluation_params']
        training_history.setdefault('eval_iterations', [
            eval_params['eval_frequency'] * (k + 1) for k in range(len(training_history['win_rates']))])
        training_history.setdefault('vs_random_iterations', [
            eval_params['eval_vs_random_frequency'] * (k + 1)
            for k in range(len(training_history['vs_random_win_rate']))])

        # Older histories do not record the simulation counts each iteration used, only
        # 'sim_counts': the counts set after each iteration (the last one after the resumed
        # iteration), which the next iteration used. Iteration 1 used the start settings. Iterations
        # whose counts are unknown get None.
        if 'simulations' not in training_history['selfplay_results']:
            mcts_params = training_history['metadata']['mcts_params']
            counts_after = {0: [mcts_params['selfplay_simulations_attacker'],
                                mcts_params['selfplay_simulations_defender']]}
            sim_counts = training_history.get('sim_counts', [])
            for k, counts in enumerate(sim_counts):
                counts_after[start_iteration - len(sim_counts) + 1 + k] = counts
            training_history['selfplay_results']['simulations'] = [
                counts_after.get(iteration - 1) for iteration in training_history['iterations']]
    else:
        # Initialize new training history with metadata
        training_history = {
            'metadata': {
                'start_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'run_name': config.name,
                'ruleset': config.ruleset,
                'board_size': config.board_size,
                'network_architecture': {
                    'type': type(network).__name__,
                    'description': config.network.describe(),
                    'num_res_blocks': config.network.num_blocks,
                    'num_channels': config.network.channels,
                    'policy_size': config.policy_size,
                    'num_parameters': total_params,
                    'config': config.network.to_dict(),
                },
                'rules': config.rules.to_dict(),
                'mcts_params': {
                    'selfplay_simulations_attacker': config.num_mcts_sims_attacker,
                    'selfplay_simulations_defender': config.num_mcts_sims_defender,
                    'eval_simulations_attacker': config.eval_mcts_sims_attacker,
                    'eval_simulations_defender': config.eval_mcts_sims_defender,
                    'c_puct': config.c_puct,
                    'temperature': config.temperature,
                    'temperature_mode': config.temperature_mode,
                    'temperature_threshold': config.temperature_threshold,
                    'temperature_decay_moves': config.temperature_decay_moves,
                    'add_dirichlet_noise': config.add_dirichlet_noise,
                    'dirichlet_alpha': config.dirichlet_alpha,
                    'dirichlet_epsilon': config.dirichlet_epsilon,
                },
                'training_params': {
                    'num_iterations': config.num_iterations,
                    'games_per_iteration': config.num_games_per_iteration,
                    'batch_size': config.batch_size,
                    'epochs_per_iteration': config.num_epochs,
                    'batches_per_epoch': config.batches_per_epoch,
                    'learning_rate_initial': config.learning_rate,
                    'lr_decay': config.lr_decay,
                    'weight_decay': config.weight_decay,
                    'value_loss_weight': config.value_loss_weight,
                    'use_data_augmentation': config.use_data_augmentation,
                    'replay_buffer_size': config.replay_buffer_size,
                    'min_buffer_size_for_training': config.min_buffer_size,
                },
                'boosting_params': {
                    'use_dynamic_boosting': config.use_dynamic_boosting,
                    'attacker_win_loss_boost': config.attacker_win_loss_boost,
                    'attacker_draw_penalty': config.draw_penalty_attacker,
                    'defender_draw_penalty': config.draw_penalty_defender,
                },
                'evaluation_params': {
                    'eval_frequency': config.eval_frequency,
                    'num_eval_games': config.eval_games,
                    'eval_win_rate_threshold': config.eval_win_rate,
                    'eval_vs_random_frequency': config.eval_vs_random_frequency,
                    'num_eval_vs_random_games': config.eval_vs_random_games,
                },
            },
        }

    # The lists of the training history, by what they have one entry for:
    # - each iteration (lines up with 'iterations'): the losses and gradient norms of LOSS_KEYS
    #   (None if training was skipped), the learning rate, buffer size, loss boosts [attacker,
    #   defender], 'selfplay_results' and 'timing'
    # - each new-vs-best evaluation (lines up with 'eval_iterations'): the score of the new network
    #   ('win_rates', draws count half), ELO gain, cumulative ELO vs iteration 0 and
    #   evaluation_summary ('eval_results')
    # - each evaluation vs random (lines up with 'vs_random_iterations')
    # A list that an older history lacks (train.py added it later) starts with one None per earlier
    # entry, so that the lists still line up.
    num_iterations = len(training_history.setdefault('iterations', []))
    num_evals = len(training_history.setdefault('eval_iterations', []))
    num_random_evals = len(training_history.setdefault('vs_random_iterations', []))
    selfplay_results = training_history.setdefault('selfplay_results', {})
    timing = training_history.setdefault('timing', {})
    for lists, keys, length in [
            (training_history, LOSS_KEYS + ['learning_rate', 'buffer_size', 'loss_boosts'], num_iterations),
            (selfplay_results, ['attacker_wins', 'defender_wins', 'draws', 'avg_game_length', 'endings',
                                'positions', 'simulations', 'unique_states'], num_iterations),
            (timing, ['selfplay_time', 'training_time', 'random_eval_time', 'network_eval_time',
                      'total_time', 'mcts'], num_iterations),
            (training_history, ['win_rates', 'elo_gain', 'cumulative_elo', 'eval_results'], num_evals),
            (training_history, ['vs_random_win_rate', 'vs_random_elo', 'vs_random_attacker_wins',
                                'vs_random_defender_wins', 'vs_random_total_draws', 'vs_random_results'],
             num_random_evals)]:
        for key in keys:
            lists.setdefault(key, [None] * length)

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
            print(f"Iteration {iteration + 1}/{config.num_iterations} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            
            # Remove the Inductor cache folders of worker processes that have exited (each worker
            # writes one folder named after its host and PID, see _play_self_play_game_worker);
            # otherwise they pile up in temp_dir. Only this host's folders are checked: /proc says
            # nothing about other hosts' PIDs.
            own_cache_dirs = os.path.join(temp_dir_path, f'torchinductor_{socket.gethostname()}_*')
            for cache_dir in glob.glob(own_cache_dirs):
                pid = cache_dir.rsplit('_', 1)[-1]
                if pid.isdigit() and not os.path.exists(f'/proc/{pid}'):
                    shutil.rmtree(cache_dir, ignore_errors=True)

            # Recreate worker pool (self-play terminates it at the end of each iteration)
            if config.num_workers > 1 and worker_pool is None:
                print(f"Recreating worker pool with {config.num_workers} workers...")
                worker_pool = mp.Pool(processes=config.num_workers, maxtasksperchild=50)

            # 1. Self-play
            print("\n[1/4] Self-play game generation...")
            selfplay_start = time.time()
            agent = Agent(network,
                         num_simulations=config.num_mcts_simulations,
                         c_puct=config.c_puct,
                         device=config.device)

            # Get current simulation counts (dynamic, or None for the fixed counts in config)
            if sim_balancer is not None:
                attacker_sims, defender_sims = sim_balancer.get_simulation_counts()
            else:
                attacker_sims, defender_sims = None, None
            # iteration_games collects all game records of this iteration, evaluation games included
            new_positions, iteration_games, selfplay_stats = generate_self_play_data(
                agent, config, pool=worker_pool, attacker_sims=attacker_sims, defender_sims=defender_sims)
            attacker_game_wins = sum(record['winner'] == 'attackers' for record in iteration_games)
            defender_game_wins = sum(record['winner'] == 'defenders' for record in iteration_games)
            avg_game_length = float(np.mean([record['plies'] for record in iteration_games]))
            selfplay_endings = {ending: sum(game['ending'] == ending for game in iteration_games)
                                for ending in config.rules.endings()}
            selfplay_time = time.time() - selfplay_start

            # generate_self_play_data terminated the pool (so its workers hold no memory during training).
            # It is recreated at the start of the next iteration; evaluation uses its own pools.
            worker_pool = None
            
            # Calculate draws from self-play
            total_games = config.num_games_per_iteration
            draws = total_games - attacker_game_wins - defender_game_wins
            
            # Update simulation balancer based on win rates
            if sim_balancer is not None:
                sim_balancer.update(attacker_game_wins, defender_game_wins, draws)
                new_attacker_sims, new_defender_sims = sim_balancer.get_simulation_counts()
                print(f"Simulation counts adjusted: Attacker {attacker_sims} -> {new_attacker_sims}, "
                      f"Defender {defender_sims} -> {new_defender_sims}")
                training_history.setdefault('sim_counts', []).append([new_attacker_sims, new_defender_sims])

            # Update win rate tracker with actual game results
            if config.use_dynamic_boosting:
                win_rate_tracker.update(attacker_game_wins, defender_game_wins)
                training_history.setdefault('win_rate_smoothed', []).append(
                    [win_rate_tracker.attacker_wins_smooth, win_rate_tracker.defender_wins_smooth])
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
            replay_buffer.add(*new_positions)
            
            print(f"Replay buffer size: {len(replay_buffer)}")
            print(f"Self-play time: {selfplay_time:.1f}s")
            
            # 2. Train network
            if len(replay_buffer) >= config.min_buffer_size:
                if config.use_stepping_lr:
                    # LR for this iteration: lr_step_values[k], with k the number of step iterations already
                    # completed. A step at 100 applies from iteration 101 on (as printed, counting from 1).
                    step_idx = sum(iteration >= step_iter for step_iter in config.lr_step_iterations)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = config.lr_step_values[step_idx]

                print(f"\n[2/4] Training network for {config.num_epochs} epochs...")
                training_start = time.time()
                losses = train_network(network, replay_buffer, optimizer, config,
                                     attacker_boost=attacker_boost,
                                     defender_boost=defender_boost)
                training_time = time.time() - training_start
                
                print(f"Training time: {training_time:.1f}s")
                print(f"\nFinal Summary (averaged over all epochs):")
                for prefix, name in (('', 'Training  '), ('val_', 'Validation')):
                    print(f"  {name} - Policy: {losses[prefix + 'policy_loss']:.4f}, "
                          f"Value: {losses[prefix + 'value_loss']:.4f}, "
                          f"Ending: {losses[prefix + 'ending_loss']:.4f}, "
                          f"Moves left: {losses[prefix + 'moves_left_loss']:.4f}, "
                          f"L2: {losses[prefix + 'l2_loss']:.6f}, "
                          f"Total: {losses[prefix + 'total_loss']:.4f}")
                print(f"  Gradients  - Policy: {losses['policy_grad']:.4f}, Value: {losses['value_grad']:.4f}")

                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\nLearning rate: {current_lr:.6f}")  # the one used in this iteration

                # Exponential learning rate decay for the next iteration (stepping LR is set before training)
                if not config.use_stepping_lr:
                    scheduler.step()
                    # Apply learning rate floor
                    if optimizer.param_groups[0]['lr'] < config.lr_floor:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = config.lr_floor
            else:
                print(f"\n[2/4] Skipping training: buffer size {len(replay_buffer)} < "
                      f"minimum {config.min_buffer_size}")
                training_time = 0
                losses = dict.fromkeys(LOSS_KEYS)  # None in the training history
                current_lr = optimizer.param_groups[0]['lr']
            
            # 3. Evaluate vs random player (only if 100% win rate not yet achieved)
            if not achieved_100_percent_vs_random and (iteration + 1) % config.eval_vs_random_frequency == 0:
                print("\n[3/4] Evaluating vs Random player...")
                eval_random_start = time.time()
                random_eval = evaluate_vs_random(network, config, 
                                                num_games=config.eval_vs_random_games,
                                                pool=worker_pool)
                eval_random_time = time.time() - eval_random_start
                iteration_games += random_eval['games']
                print(f"Random evaluation time: {eval_random_time:.1f}s")
                training_history['vs_random_iterations'].append(iteration + 1)
                training_history['vs_random_results'].append(
                    evaluation_summary(random_eval['games'], config.rules))
                training_history['vs_random_win_rate'].append(random_eval['win_rate'])
                training_history['vs_random_elo'].append(random_eval['elo_diff'])
                training_history['vs_random_attacker_wins'].append(random_eval['attacker_wins'])
                training_history['vs_random_defender_wins'].append(random_eval['defender_wins'])
                training_history['vs_random_total_draws'].append(random_eval['total_draws'])
                
                # Check if 100% score achieved (win_rate now includes draws as 0.5)
                # For 100% milestone, we want actual 100% wins (no draws or losses)
                perfect_score = (random_eval['total_wins'] == random_eval['total_games'])
                if perfect_score:
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
                print("\n[4/4] Evaluating new vs old network...")
                eval_network_start = time.time()
                score_rate, eval_games = evaluate_networks(network, best_network, config,
                                                           pool=worker_pool)
                iteration_games += eval_games
                eval_network_time = time.time() - eval_network_start
                print(f"Network evaluation time: {eval_network_time:.1f}s")
                training_history['eval_iterations'].append(iteration + 1)
                training_history['eval_results'].append(evaluation_summary(eval_games, config.rules))
                training_history['win_rates'].append(score_rate)
                
                # Calculate ELO gain from this evaluation (using score that counts draws as 0.5)
                elo_gain = calculate_elo_difference(score_rate)
                training_history['elo_gain'].append(elo_gain)
                
                # Check if score meets threshold (draws count as half a win)
                if score_rate >= config.eval_win_rate:
                    print(f"New network scores {100*score_rate:.1f}% - updating best network!")
                    print(f"ELO gain: {elo_gain:+.1f} (new network vs previous best)")
                    # Update cumulative ELO by adding this gain
                    cumulative_elo += elo_gain
                    # Clean state dict to remove _orig_mod. prefix from compiled network
                    best_network.load_state_dict(clean_state_dict(network.state_dict()))
                    # Save the new best model immediately
                    save_checkpoint(best_network, optimizer, iteration + 1, config,
                                  training_history, "best_model.pth")
                else:
                    print(f"New network scores {100*score_rate:.1f}% - keeping old network")
                    print(f"ELO difference: {elo_gain:+.1f} (new network vs previous best, not applied)")
                
                # Record cumulative ELO (relative to iteration 0)
                training_history['cumulative_elo'].append(cumulative_elo)
                print(f"Cumulative ELO (vs iteration 0): {cumulative_elo:+.1f}")
            else:
                print(f"\n[4/4] Skipping network evaluation (every {config.eval_frequency} iterations)")
                eval_network_time = 0
            
            # Calculate total iteration time
            iter_time = time.time() - iter_start_time

            # Record this iteration in all per-iteration lists of the training history
            training_history['iterations'].append(iteration + 1)
            for key, value in losses.items():
                training_history[key].append(value)
            training_history['learning_rate'].append(current_lr)
            training_history['buffer_size'].append(len(replay_buffer))
            training_history['loss_boosts'].append([attacker_boost, defender_boost])
            if attacker_sims is None:  # fixed simulation counts
                attacker_sims, defender_sims = config.num_mcts_sims_attacker, config.num_mcts_sims_defender
            for key, value in [('attacker_wins', attacker_game_wins), ('defender_wins', defender_game_wins),
                               ('draws', draws), ('avg_game_length', avg_game_length),
                               ('endings', selfplay_endings), ('positions', len(new_positions[0])),
                               ('simulations', [attacker_sims, defender_sims]),
                               ('unique_states', selfplay_stats['unique_states'])]:
                selfplay_results[key].append(value)
            for key, value in [('selfplay_time', selfplay_time), ('training_time', training_time),
                               ('random_eval_time', eval_random_time),
                               ('network_eval_time', eval_network_time), ('total_time', iter_time),
                               ('mcts', selfplay_stats['mcts_timing'])]:
                timing[key].append(value)

            # 5. Save checkpoint (only the current network, not best), after all records of this
            # iteration, since a resume continues from the training history stored in it
            if (iteration + 1) % config.save_frequency == 0:
                save_checkpoint(network, optimizer, iteration + 1, config,
                              training_history, f"checkpoint_iter_{iteration + 1}.pth")

            # 6. Save training history (via a temporary file, as in save_checkpoint)
            history_path = os.path.join(config.checkpoint_dir, "training_history.json")
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            with open(history_path + '.tmp', 'w') as f:
                json.dump(training_history, f, indent=2)
            os.replace(history_path + '.tmp', history_path)

            # 7. Append this iteration's games (self-play and evaluation) to games.jsonl, one JSON
            # object per line (see game_record). A resumed run appends the repeated iterations
            # again.
            with open(os.path.join(config.checkpoint_dir, "games.jsonl"), 'a') as f:
                for record in iteration_games:
                    f.write(json.dumps({'iteration': iteration + 1, **record}) + '\n')

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
    
    # Final save. best_model.pth is not saved again here: it was saved when the best network last
    # changed, together with the matching optimizer state and iteration.
    save_checkpoint(network, optimizer, config.num_iterations, config,
                   training_history, "final_model.pth")
    
    return network, best_network, training_history


# =============================================================================
# MAIN
# =============================================================================

# Runs are started with a JSON config:
#   python run_training.py configs/brandubh/baseline.json [--set section.key=value ...]
# See run_training.py, run_config.py and the configs/ directory.
