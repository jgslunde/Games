"""
Self-play training loop for Brandubh AlphaZero.

Implements the complete AlphaZero training pipeline:
- Self-play game generation with MCTS
- Replay buffer with experience storage
- Neural network training
- Model evaluation and checkpointing
- Data augmentation (symmetries)
"""

import os
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
from agent import BrandubhAgent


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

class TrainingConfig:
    """Configuration for training."""
    
    # Self-play
    num_iterations = 100              # Number of training iterations
    num_games_per_iteration = 100     # Self-play games per iteration
    num_mcts_simulations = 100        # MCTS simulations per move
    c_puct = 1.4                      # MCTS exploration constant
    num_workers = mp.cpu_count()      # Number of parallel workers (default: all CPUs)
    
    # Temperature (exploration during self-play)
    temperature = 1.0                 # Sampling temperature for moves
    temperature_threshold = 15        # Move number after which temperature = 0
    
    # Neural network
    num_res_blocks = 4                # Residual blocks in network
    num_channels = 64                 # Channels in convolutional layers
    
    # Training
    batch_size = 32                   # Training batch size
    num_epochs = 10                   # Epochs per iteration
    learning_rate = 0.001             # Initial learning rate
    lr_decay = 0.95                   # Learning rate decay per iteration
    weight_decay = 1e-4               # L2 regularization
    value_loss_weight = 1.0           # Weight for value loss (policy loss weight is always 1.0)
    
    # Replay buffer
    replay_buffer_size = 50000        # Maximum samples in replay buffer
    min_buffer_size = 1000            # Minimum samples before training
    
    # Evaluation
    eval_games = 20                   # Games for evaluation
    eval_win_rate = 0.55              # Win rate threshold to replace best model
    eval_frequency = 5                # Evaluate every N iterations
    
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
    Each sample is (state, policy, value).
    """
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add a sample to the buffer."""
        self.buffer.append((state, policy, value))
    
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
            
            self.add(state, policy, value)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch from the buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        policies = []
        values = []
        
        for idx in indices:
            state, policy, value = self.buffer[idx]
            states.append(state)
            policies.append(policy)
            values.append(value)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def augment_sample(state: np.ndarray, policy: np.ndarray, value: float) -> List[Tuple]:
    """
    Generate all 8 symmetric transformations of a sample.
    Returns list of (state, policy, value) tuples.
    
    The 7x7 board has 4-fold rotational symmetry and 2-fold reflective symmetry.
    This gives us 8 unique transformations total.
    """
    augmented = []
    
    # Helper to rotate a move in policy space
    def rotate_policy(policy_vec, k):
        """Rotate policy vector k times 90 degrees clockwise."""
        policy_arr = policy_vec.reshape(49, 24)  # (squares, directions*distances)
        
        # Rotate board positions
        rotated = np.zeros_like(policy_arr)
        for old_sq in range(49):
            old_r, old_c = old_sq // 7, old_sq % 7
            # Rotate position
            for _ in range(k):
                old_r, old_c = old_c, 6 - old_r
            new_sq = old_r * 7 + old_c
            
            # Rotate directions (up->right->down->left)
            for old_dir in range(4):
                new_dir = (old_dir + k) % 4
                for dist in range(6):
                    old_idx = old_dir * 6 + dist
                    new_idx = new_dir * 6 + dist
                    rotated[new_sq, new_idx] = policy_arr[old_sq, old_idx]
        
        return rotated.reshape(-1)
    
    def flip_policy(policy_vec, horizontal=True):
        """Flip policy vector horizontally or vertically."""
        policy_arr = policy_vec.reshape(49, 24)
        flipped = np.zeros_like(policy_arr)
        
        for old_sq in range(49):
            old_r, old_c = old_sq // 7, old_sq % 7
            # Flip position
            if horizontal:
                new_r, new_c = old_r, 6 - old_c
                dir_map = {0: 0, 1: 1, 2: 3, 3: 2}  # Swap left/right
            else:
                new_r, new_c = 6 - old_r, old_c
                dir_map = {0: 1, 1: 0, 2: 2, 3: 3}  # Swap up/down
            
            new_sq = new_r * 7 + new_c
            
            for old_dir in range(4):
                new_dir = dir_map[old_dir]
                for dist in range(6):
                    old_idx = old_dir * 6 + dist
                    new_idx = new_dir * 6 + dist
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
                                num_simulations, c_puct, temperature, temperature_threshold, game_idx):
    """
    Worker function for parallel self-play game generation.
    Must be at module level for multiprocessing. Imports torch inside to avoid pickling issues.
    
    Args:
        network_path: Path to saved network weights file
        num_res_blocks: Number of residual blocks in network
        num_channels: Number of channels in network
        num_simulations: MCTS simulations per move
        c_puct: MCTS exploration constant
        temperature: Sampling temperature
        temperature_threshold: Move number threshold for temperature
        game_idx: Game index (unused, for pool.map)
    
    Returns:
        dict with game data
    """
    # Set torch to use only 1 thread per worker to avoid conflicts
    torch.set_num_threads(1)
    
    # Import inside worker to avoid issues with multiprocessing
    from brandubh import Brandubh
    from network import BrandubhNet, MoveEncoder
    from mcts import MCTS
    
    # Reconstruct network on CPU and load from file
    network = BrandubhNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
    network.load_state_dict(torch.load(network_path, map_location='cpu'))
    network.to('cpu')
    network.eval()
    
    # Create MCTS instance
    mcts = MCTS(network, num_simulations=num_simulations, c_puct=c_puct, device='cpu')
    
    # Play game
    game = Brandubh()
    states = []
    policies = []
    players = []
    move_count = 0
    
    while not game.game_over:
        current_player = game.current_player
        
        # Determine temperature based on move count
        temp = temperature if move_count < temperature_threshold else 0.0
        
        # Get state
        state = game.get_state()
        
        # Run MCTS to get policy
        visit_probs = mcts.search(game)
        
        # Convert to policy vector
        policy = np.zeros(1176, dtype=np.float32)
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
    
    # Determine draw reason
    draw_reason = None
    if game.winner is None:
        draw_reason = 'repetition'
    elif not game.game_over and move_count > 500:
        # Hit move limit without natural game end
        draw_reason = 'move_limit'
    
    return {
        'states': states,
        'policies': policies,
        'winner': game.winner,  # Can be 0, 1, or None (draw)
        'players': players,
        'num_moves': move_count,
        'draw_reason': draw_reason  # 'repetition', 'move_limit', or None
    }


def play_self_play_game(agent: BrandubhAgent, config: TrainingConfig) -> Dict:
    """
    Play a single self-play game and collect training data.
    
    Returns:
        dict with 'states', 'policies', 'winner', 'players', 'num_moves'
    """
    game = Brandubh()
    states = []
    policies = []
    players = []
    
    move_count = 0
    
    while not game.game_over:
        current_player = game.current_player
        
        # Determine temperature based on move count
        if move_count < config.temperature_threshold:
            temperature = config.temperature
        else:
            temperature = 0.0  # Deterministic after threshold
        
        # Get state
        state = game.get_state()
        
        # Run MCTS to get policy
        visit_probs = agent.mcts.search(game)
        
        # Convert to policy vector
        policy = np.zeros(1176, dtype=np.float32)
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
    
    # Determine draw reason
    draw_reason = None
    if game.winner is None:
        draw_reason = 'repetition'
    elif not game.game_over and move_count > 500:
        # Hit move limit without natural game end
        draw_reason = 'move_limit'
    
    return {
        'states': states,
        'policies': policies,
        'winner': game.winner,  # Can be 0, 1, or None (draw)
        'players': players,
        'num_moves': move_count,
        'draw_reason': draw_reason  # 'repetition', 'move_limit', or None
    }


def generate_self_play_data(agent: BrandubhAgent, config: TrainingConfig, pool=None, temp_network_path=None) -> ReplayBuffer:
    """
    Generate self-play games and store in replay buffer.
    Uses multiprocessing to parallelize game generation.
    
    Args:
        agent: Agent to use for self-play
        config: Training configuration
        pool: Optional multiprocessing pool to use (if None, creates temporary pool)
        temp_network_path: Path to temporary network file (avoids pickling large tensors)
    
    Returns:
        ReplayBuffer with collected experience
    """
    buffer = ReplayBuffer(config.replay_buffer_size)
    
    print(f"Generating {config.num_games_per_iteration} self-play games using {config.num_workers} workers...")
    
    # Save network to temporary file to avoid pickling issues with many workers
    import tempfile
    if temp_network_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pth', delete=False)
        temp_network_path = temp_file.name
        temp_file.close()
        torch.save(agent.network.state_dict(), temp_network_path)
        cleanup_temp_file = True
    else:
        cleanup_temp_file = False
    
    try:
        # Create worker function with partial
        worker_func = partial(
            _play_self_play_game_worker,
            temp_network_path,
            config.num_res_blocks,
            config.num_channels,
            config.num_mcts_simulations,
            config.c_puct,
            config.temperature,
            config.temperature_threshold
        )
        
        # Play games in parallel
        if config.num_workers > 1:
            if pool is not None:
                # Use provided persistent pool
                game_results = pool.map(worker_func, range(config.num_games_per_iteration), chunksize=1)
            else:
                # Create temporary pool (for backward compatibility)
                with mp.Pool(processes=config.num_workers, maxtasksperchild=10) as temp_pool:
                    game_results = temp_pool.map(worker_func, range(config.num_games_per_iteration), chunksize=1)
        else:
            # Single-threaded fallback
            game_results = [worker_func(i) for i in range(config.num_games_per_iteration)]
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
    repetition_draws = 0
    move_limit_draws = 0
    
    for game_data in game_results:
        total_moves += game_data['num_moves']
        winner = game_data['winner']
        draw_reason = game_data.get('draw_reason', None)
        
        if winner == 0:
            attacker_wins += 1
        elif winner == 1:
            defender_wins += 1
        else:  # winner is None or move limit reached
            draws += 1
            if draw_reason == 'repetition':
                repetition_draws += 1
            elif draw_reason == 'move_limit':
                move_limit_draws += 1
        
        # Add to buffer
        for state, policy, player in zip(game_data['states'], 
                                         game_data['policies'], 
                                         game_data['players']):
            # Determine value from player's perspective
            if winner is None:
                value = 0.0  # Draw
            elif winner == player:
                value = 1.0
            elif winner == 1 - player:
                value = -1.0
            else:
                value = 0.0  # Shouldn't happen
            
            # Add original sample
            buffer.add(state, policy, value)
            
            # Optionally add augmented samples (can be memory intensive)
            # Uncomment to enable data augmentation
            # augmented = augment_sample(state, policy, value)
            # for aug_state, aug_policy, aug_value in augmented:
            #     buffer.add(aug_state, aug_policy, aug_value)
    
    print(f"Generated {len(buffer)} training samples")
    total_games = config.num_games_per_iteration
    print(f"Results: {attacker_wins} attacker wins ({100*attacker_wins/total_games:.1f}%), "
          f"{defender_wins} defender wins ({100*defender_wins/total_games:.1f}%), "
          f"{draws} draws ({100*draws/total_games:.1f}%)")
    if draws > 0:
        print(f"  Draw breakdown: {repetition_draws} by repetition, {move_limit_draws} by move limit (500+ moves)")
    print(f"Average game length: {total_moves/total_games:.1f} moves")
    
    return buffer


# =============================================================================
# TRAINING
# =============================================================================

def train_network(network: BrandubhNet, buffer: ReplayBuffer, 
                  optimizer: optim.Optimizer, config: TrainingConfig) -> Dict[str, float]:
    """
    Train the neural network on samples from the replay buffer.
    
    Returns:
        dict with 'policy_loss', 'value_loss', 'total_loss'
    """
    network.train()
    device = config.device
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    samples_per_epoch = min(len(buffer), config.batch_size * 100)
    batches_per_epoch = samples_per_epoch // config.batch_size
    
    for epoch in range(config.num_epochs):
        for batch_idx in range(batches_per_epoch):
            # Sample batch
            states, policies, values = buffer.sample(config.batch_size)
            
            # Convert to tensors
            states = torch.from_numpy(states).to(device)
            policies = torch.from_numpy(policies).to(device)
            values = torch.from_numpy(values).unsqueeze(1).to(device)
            
            # Forward pass
            pred_policies, pred_values = network(states)
            
            # Compute losses
            policy_loss = -torch.mean(torch.sum(policies * 
                                      torch.log_softmax(pred_policies, dim=1), dim=1))
            value_loss = torch.mean((pred_values - values) ** 2)
            loss = policy_loss + config.value_loss_weight * value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1
    
    return {
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
        'total_loss': total_loss / num_batches
    }


# =============================================================================
# EVALUATION
# =============================================================================

def _evaluate_vs_random_worker(network_path, num_res_blocks, num_channels,
                                num_simulations, c_puct, nn_plays_attacker, game_idx):
    """
    Worker function for parallel evaluation against random player.
    Must be at module level for multiprocessing. Imports inside to avoid pickling issues.
    
    Args:
        network_path: Path to saved network weights file
        num_res_blocks: Number of residual blocks
        num_channels: Number of channels
        num_simulations: MCTS simulations per move
        c_puct: MCTS exploration constant
        nn_plays_attacker: Whether NN plays as attacker
        game_idx: Game index (unused, for pool.map)
    
    Returns:
        1 if NN wins, 0 otherwise
    """
    # Set torch to use only 1 thread per worker to avoid conflicts
    torch.set_num_threads(1)
    
    # Import inside worker
    from network import BrandubhNet
    from agent import BrandubhAgent, RandomAgent, play_game
    
    # Reconstruct network on CPU and load from file
    network = BrandubhNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
    network.load_state_dict(torch.load(network_path, map_location='cpu'))
    network.to('cpu')
    network.eval()
    
    # Create agents on CPU
    nn_agent = BrandubhAgent(network, num_simulations=num_simulations,
                            c_puct=c_puct, device='cpu')
    random_agent = RandomAgent()
    
    # Play game
    if nn_plays_attacker:
        winner = play_game(nn_agent, random_agent, display=False)
        return 1 if winner == 0 else 0
    else:
        winner = play_game(random_agent, nn_agent, display=False)
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
        torch.save(network.state_dict(), temp_network_path)
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
            config.num_mcts_simulations,
            config.c_puct,
            True  # nn_plays_attacker
        )
        
        worker_func_defender = partial(
            _evaluate_vs_random_worker,
            temp_network_path,
            config.num_res_blocks,
            config.num_channels,
            config.num_mcts_simulations,
            config.c_puct,
            False  # nn_plays_attacker
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
                             num_res_blocks, num_channels, num_simulations, c_puct,
                             new_plays_attacker, game_idx):
    """
    Worker function for parallel network evaluation.
    Must be at module level for multiprocessing. Imports inside to avoid pickling issues.
    
    Args:
        new_network_path: Path to saved new network weights file
        old_network_path: Path to saved old network weights file
        num_res_blocks: Number of residual blocks
        num_channels: Number of channels
        num_simulations: MCTS simulations per move
        c_puct: MCTS exploration constant
        new_plays_attacker: Whether new network plays as attacker
        game_idx: Game index (unused, for pool.map)
    
    Returns:
        1 if new network wins, 0 otherwise
    """
    # Set torch to use only 1 thread per worker to avoid conflicts
    torch.set_num_threads(1)
    
    # Import inside worker
    from network import BrandubhNet
    from agent import BrandubhAgent, play_game
    
    # Reconstruct networks on CPU and load from files
    new_network = BrandubhNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
    new_network.load_state_dict(torch.load(new_network_path, map_location='cpu'))
    new_network.to('cpu')
    new_network.eval()
    
    old_network = BrandubhNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
    old_network.load_state_dict(torch.load(old_network_path, map_location='cpu'))
    old_network.to('cpu')
    old_network.eval()
    
    # Create agents on CPU
    new_agent = BrandubhAgent(new_network, num_simulations=num_simulations,
                             c_puct=c_puct, device='cpu')
    old_agent = BrandubhAgent(old_network, num_simulations=num_simulations,
                             c_puct=c_puct, device='cpu')
    
    # Play game
    if new_plays_attacker:
        winner = play_game(new_agent, old_agent, display=False)
        return 1 if winner == 0 else 0
    else:
        winner = play_game(old_agent, new_agent, display=False)
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
        torch.save(new_network.state_dict(), temp_new_path)
        cleanup_new = True
    else:
        cleanup_new = False
    
    if temp_old_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_old.pth', delete=False)
        temp_old_path = temp_file.name
        temp_file.close()
        torch.save(old_network.state_dict(), temp_old_path)
        cleanup_old = True
    else:
        cleanup_old = False
    
    try:
        # Create worker functions with partial, alternating who plays attacker
        work_items = []
        for i in range(config.eval_games):
            new_plays_attacker = (i % 2 == 0)
            worker_func = partial(
                _evaluate_networks_worker,
                temp_new_path,
                temp_old_path,
                config.num_res_blocks,
                config.num_channels,
                config.num_mcts_simulations,
                config.c_puct,
                new_plays_attacker
            )
            work_items.append(worker_func)
        
        # Play games in parallel
        if config.num_workers > 1:
            if pool is not None:
                # Use provided persistent pool
                results = [pool.apply(func, args=(i,)) for i, func in enumerate(work_items)]
            else:
                # Create temporary pool (for backward compatibility)
                with mp.Pool(processes=config.num_workers, maxtasksperchild=10) as temp_pool:
                    results = [temp_pool.apply(func, args=(i,)) for i, func in enumerate(work_items)]
        else:
            # Single-threaded fallback
            results = [func(i) for i, func in enumerate(work_items)]
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
    
    new_wins = sum(results)
    win_rate = new_wins / config.eval_games
    print(f"New network win rate: {100*win_rate:.1f}% ({new_wins}/{config.eval_games})")
    
    return win_rate


# =============================================================================
# CHECKPOINTING
# =============================================================================

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
    checkpoint = torch.load(filepath)
    
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
    print("=" * 70)
    print("Brandubh AlphaZero Training")
    print("=" * 70)
    print(f"\nDevice: {config.device}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Games per iteration: {config.num_games_per_iteration}")
    print(f"MCTS simulations: {config.num_mcts_simulations}")
    print(f"Parallel workers: {config.num_workers}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Replay buffer size: {config.replay_buffer_size}")
    print()
    
    # Initialize network
    network = BrandubhNet(num_res_blocks=config.num_res_blocks,
                         num_channels=config.num_channels).to(config.device)
    
    # Initialize optimizer
    optimizer = optim.Adam(network.parameters(), 
                          lr=config.learning_rate,
                          weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if resume_from is not None:
        start_iteration = load_checkpoint(resume_from, network, optimizer)
    
    # Initialize best network (for evaluation)
    best_network = BrandubhNet(num_res_blocks=config.num_res_blocks,
                              num_channels=config.num_channels).to(config.device)
    best_network.load_state_dict(network.state_dict())
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(config.replay_buffer_size)
    
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
        'vs_random_defender_wins': []
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
            agent = BrandubhAgent(network, 
                                 num_simulations=config.num_mcts_simulations,
                                 c_puct=config.c_puct,
                                 device=config.device)
            
            new_buffer = generate_self_play_data(agent, config, pool=worker_pool)
            
            # Add to main replay buffer
            for state, policy, value in new_buffer.buffer:
                replay_buffer.add(state, policy, value)
            
            print(f"Replay buffer size: {len(replay_buffer)}")
            
            # 2. Train network
            if len(replay_buffer) >= config.min_buffer_size:
                print(f"\nTraining network for {config.num_epochs} epochs...")
                losses = train_network(network, replay_buffer, optimizer, config)
                
                print(f"Policy loss: {losses['policy_loss']:.4f}")
                print(f"Value loss: {losses['value_loss']:.4f}")
                print(f"Total loss: {losses['total_loss']:.4f}")
                
                # Record metrics
                training_history['iterations'].append(iteration + 1)
                training_history['policy_loss'].append(losses['policy_loss'])
                training_history['value_loss'].append(losses['value_loss'])
                training_history['total_loss'].append(losses['total_loss'])
                training_history['buffer_size'].append(len(replay_buffer))
            else:
                print(f"\nSkipping training: buffer size {len(replay_buffer)} < "
                      f"minimum {config.min_buffer_size}")
            
            # 3. Evaluate vs random player
            if (iteration + 1) % config.eval_vs_random_frequency == 0:
                random_eval = evaluate_vs_random(network, config, 
                                                num_games=config.eval_vs_random_games,
                                                pool=worker_pool)
                training_history['vs_random_win_rate'].append(random_eval['win_rate'])
                training_history['vs_random_elo'].append(random_eval['elo_diff'])
                training_history['vs_random_attacker_wins'].append(random_eval['attacker_wins'])
                training_history['vs_random_defender_wins'].append(random_eval['defender_wins'])
            
            # 4. Evaluate and update best network
            if (iteration + 1) % config.eval_frequency == 0:
                win_rate = evaluate_networks(network, best_network, config, pool=worker_pool)
                training_history['win_rates'].append(win_rate)
                
                if win_rate >= config.eval_win_rate:
                    print(f"New network wins {100*win_rate:.1f}% - updating best network!")
                    best_network.load_state_dict(network.state_dict())
                else:
                    print(f"New network wins {100*win_rate:.1f}% - keeping old network")
            
            # 5. Learning rate decay
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nLearning rate: {current_lr:.6f}")
            
            # 6. Save checkpoint
            if (iteration + 1) % config.save_frequency == 0:
                save_checkpoint(network, optimizer, iteration + 1, config, 
                              training_history, f"checkpoint_iter_{iteration + 1}.pth")
                save_checkpoint(best_network, optimizer, iteration + 1, config,
                              training_history, "best_model.pth")
            
            # 7. Save training history
            history_path = os.path.join(config.checkpoint_dir, "training_history.json")
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
            
            iter_time = time.time() - iter_start_time
            print(f"\nIteration time: {iter_time:.1f}s")
    
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
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    # =============================================================================
    # DEFAULT COMMAND-LINE ARGUMENTS
    # Modify these values to change defaults without using command-line arguments
    # =============================================================================
    
    DEFAULT_ITERATIONS = 100
    DEFAULT_GAMES = 100
    DEFAULT_SIMULATIONS = 100
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_EPOCHS = 10
    DEFAULT_EVAL_VS_RANDOM = 10
    DEFAULT_NUM_WORKERS = mp.cpu_count()  # Use all available CPU cores
    DEFAULT_DEVICE = None  # None = auto-detect (cuda if available, else cpu)
    DEFAULT_RESUME = None  # Path to checkpoint file, or None to start fresh
    
    # Temperature parameters
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_TEMPERATURE_THRESHOLD = 15
    
    # Network architecture
    DEFAULT_RES_BLOCKS = 4
    DEFAULT_CHANNELS = 64
    
    # Replay buffer
    DEFAULT_REPLAY_BUFFER_SIZE = 50000
    DEFAULT_MIN_BUFFER_SIZE = 1000
    
    # Learning rate decay and regularization
    DEFAULT_LR_DECAY = 0.95
    DEFAULT_WEIGHT_DECAY = 1e-4
    DEFAULT_VALUE_LOSS_WEIGHT = 1.0
    
    # MCTS exploration
    DEFAULT_C_PUCT = 1.4
    
    # Evaluation
    DEFAULT_EVAL_GAMES = 20
    DEFAULT_EVAL_WIN_RATE = 0.55
    DEFAULT_EVAL_FREQUENCY = 5
    DEFAULT_EVAL_VS_RANDOM_FREQUENCY = 1
    
    # Checkpointing
    DEFAULT_SAVE_FREQUENCY = 5
    
    # =============================================================================
    # COMMAND-LINE ARGUMENT PARSER
    # =============================================================================
    
    parser = argparse.ArgumentParser(description="Train Brandubh AlphaZero")
    
    # Core training parameters
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                       help=f"Number of training iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                       help=f"Self-play games per iteration (default: {DEFAULT_GAMES})")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS,
                       help=f"MCTS simulations per move (default: {DEFAULT_SIMULATIONS})")
    
    # Neural network training
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE,
                       help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument("--lr-decay", type=float, default=DEFAULT_LR_DECAY,
                       help=f"Learning rate decay per iteration (default: {DEFAULT_LR_DECAY})")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY,
                       help=f"L2 regularization weight decay (default: {DEFAULT_WEIGHT_DECAY})")
    parser.add_argument("--value-loss-weight", type=float, default=DEFAULT_VALUE_LOSS_WEIGHT,
                       help=f"Weight for value loss relative to policy loss (default: {DEFAULT_VALUE_LOSS_WEIGHT})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                       help=f"Training epochs per iteration (default: {DEFAULT_EPOCHS})")
    
    # Network architecture
    parser.add_argument("--res-blocks", type=int, default=DEFAULT_RES_BLOCKS,
                       help=f"Number of residual blocks (default: {DEFAULT_RES_BLOCKS})")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS,
                       help=f"Number of channels in conv layers (default: {DEFAULT_CHANNELS})")
    
    # MCTS parameters
    parser.add_argument("--c-puct", type=float, default=DEFAULT_C_PUCT,
                       help=f"MCTS exploration constant (default: {DEFAULT_C_PUCT})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                       help=f"Sampling temperature for move selection (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--temperature-threshold", type=int, default=DEFAULT_TEMPERATURE_THRESHOLD,
                       help=f"Move number after which temperature=0 (default: {DEFAULT_TEMPERATURE_THRESHOLD})")
    
    # Replay buffer
    parser.add_argument("--replay-buffer-size", type=int, default=DEFAULT_REPLAY_BUFFER_SIZE,
                       help=f"Maximum replay buffer size (default: {DEFAULT_REPLAY_BUFFER_SIZE})")
    parser.add_argument("--min-buffer-size", type=int, default=DEFAULT_MIN_BUFFER_SIZE,
                       help=f"Minimum buffer size before training (default: {DEFAULT_MIN_BUFFER_SIZE})")
    
    # Evaluation
    parser.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES,
                       help=f"Games for network evaluation (default: {DEFAULT_EVAL_GAMES})")
    parser.add_argument("--eval-win-rate", type=float, default=DEFAULT_EVAL_WIN_RATE,
                       help=f"Win rate threshold to replace best model (default: {DEFAULT_EVAL_WIN_RATE})")
    parser.add_argument("--eval-frequency", type=int, default=DEFAULT_EVAL_FREQUENCY,
                       help=f"Evaluate every N iterations (default: {DEFAULT_EVAL_FREQUENCY})")
    parser.add_argument("--eval-vs-random", type=int, default=DEFAULT_EVAL_VS_RANDOM,
                       help=f"Games vs random per color (default: {DEFAULT_EVAL_VS_RANDOM})")
    parser.add_argument("--eval-vs-random-frequency", type=int, default=DEFAULT_EVAL_VS_RANDOM_FREQUENCY,
                       help=f"Evaluate vs random every N iterations (default: {DEFAULT_EVAL_VS_RANDOM_FREQUENCY})")
    
    # System
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                       help=f"Number of parallel workers (default: {DEFAULT_NUM_WORKERS} CPUs)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                       help="Device (cuda/cpu, default: auto-detect)")
    
    # Checkpointing
    parser.add_argument("--save-frequency", type=int, default=DEFAULT_SAVE_FREQUENCY,
                       help=f"Save checkpoint every N iterations (default: {DEFAULT_SAVE_FREQUENCY})")
    parser.add_argument("--resume", type=str, default=DEFAULT_RESUME,
                       help="Resume from checkpoint file")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    
    # Core training parameters
    config.num_iterations = args.iterations
    config.num_games_per_iteration = args.games
    config.num_mcts_simulations = args.simulations
    
    # Neural network training
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.lr_decay = args.lr_decay
    config.weight_decay = args.weight_decay
    config.value_loss_weight = args.value_loss_weight
    config.num_epochs = args.epochs
    
    # Network architecture
    config.num_res_blocks = args.res_blocks
    config.num_channels = args.channels
    
    # MCTS parameters
    config.c_puct = args.c_puct
    config.temperature = args.temperature
    config.temperature_threshold = args.temperature_threshold
    
    # Replay buffer
    config.replay_buffer_size = args.replay_buffer_size
    config.min_buffer_size = args.min_buffer_size
    
    # Evaluation
    config.eval_games = args.eval_games
    config.eval_win_rate = args.eval_win_rate
    config.eval_frequency = args.eval_frequency
    config.eval_vs_random_games = args.eval_vs_random
    config.eval_vs_random_frequency = args.eval_vs_random_frequency
    
    # System
    config.num_workers = args.num_workers
    if args.device is not None:
        config.device = args.device
    
    # Checkpointing
    config.save_frequency = args.save_frequency
    
    # Run training
    train(config, resume_from=args.resume)
