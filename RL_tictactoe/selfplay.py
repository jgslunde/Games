import numpy as np
import torch
from TicTacToe import TicTacToeGame
from MCTS import MCTS
import multiprocessing as mp
from functools import partial


def augment_tictactoe_data(board, policy):
    """
    Generate all 8 symmetries (4 rotations + 4 reflections) of a tic-tac-toe position.
    This increases training data 8x and helps the network learn symmetry-invariant patterns.
    
    Args:
        board: 9-element array representing board state
        policy: 9-element array representing move probabilities
        
    Returns:
        List of (augmented_board, augmented_policy) tuples
    """
    def rotate_90(arr):
        """Rotate a 3x3 array 90 degrees clockwise."""
        # Reshape to 3x3, rotate, flatten
        return np.rot90(arr.reshape(3, 3), k=-1).flatten()
    
    def reflect_horizontal(arr):
        """Reflect a 3x3 array horizontally."""
        return np.fliplr(arr.reshape(3, 3)).flatten()
    
    augmentations = []
    
    # Original
    augmentations.append((board.copy(), policy.copy()))
    
    # 3 rotations (90, 180, 270 degrees)
    b, p = board.copy(), policy.copy()
    for _ in range(3):
        b = rotate_90(b)
        p = rotate_90(p)
        augmentations.append((b.copy(), p.copy()))
    
    # Horizontal flip + 3 rotations
    b_flipped = reflect_horizontal(board)
    p_flipped = reflect_horizontal(policy)
    augmentations.append((b_flipped.copy(), p_flipped.copy()))
    
    for _ in range(3):
        b_flipped = rotate_90(b_flipped)
        p_flipped = rotate_90(p_flipped)
        augmentations.append((b_flipped.copy(), p_flipped.copy()))
    
    return augmentations


def _play_single_mcts_game(network_state_dict, num_simulations, c_puct, temperature, temperature_threshold, game_idx):
    """
    Worker function to play a single MCTS self-play game.
    This is a module-level function so it can be pickled for multiprocessing.
    
    Args:
        network_state_dict: State dict of the network (for reconstruction)
        num_simulations: MCTS simulations per move
        c_puct: MCTS exploration constant
        temperature: Temperature for move selection
        temperature_threshold: Move threshold for temperature
        game_idx: Game index (unused, for pool.imap)
    
    Returns:
        (training_data, winner) tuple
    """
    # Import here to avoid issues with multiprocessing
    from TTTnet import TicTacToeNet
    
    # Reconstruct network
    network = TicTacToeNet()
    network.load_state_dict(network_state_dict)
    network.eval()
    
    # Create SelfPlay instance
    selfplay = SelfPlay(network, num_simulations=num_simulations, c_puct=c_puct)
    
    # Play game
    return selfplay.play_game(temperature=temperature, temperature_threshold=temperature_threshold)


def _play_single_random_game(network_state_dict, num_simulations, c_puct, temperature, temperature_threshold, game_idx):
    """
    Worker function to play a single game against random opponent.
    
    Args:
        network_state_dict: State dict of the network
        num_simulations: MCTS simulations per move
        c_puct: MCTS exploration constant
        temperature: Temperature for move selection
        temperature_threshold: Move threshold for temperature
        game_idx: Game index (determines who plays first)
    
    Returns:
        (training_data, winner) tuple
    """
    from TTTnet import TicTacToeNet
    
    # Reconstruct network
    network = TicTacToeNet()
    network.load_state_dict(network_state_dict)
    network.eval()
    
    # Create SelfPlay instance
    selfplay = SelfPlay(network, num_simulations=num_simulations, c_puct=c_puct)
    
    # Alternate who plays first
    network_plays_first = (game_idx % 2 == 0)
    
    # Play game
    return selfplay.play_game_vs_random(
        temperature=temperature,
        temperature_threshold=temperature_threshold,
        network_plays_first=network_plays_first
    )


class SelfPlay:
    """
    Generates training data through self-play using MCTS.
    """
    def __init__(self, network, num_simulations=50, c_puct=1.0):
        self.network = network
        self.mcts = MCTS(network, num_simulations=num_simulations, c_puct=c_puct)
        
    def play_game(self, temperature=1.0, temperature_threshold=15):
        """
        Play one full game using MCTS for move selection.
        
        Args:
            temperature: Controls exploration during move selection.
                        Higher = more exploration. Set to 0 for greedy play.
            temperature_threshold: Move number after which temperature drops to 0
                                  (for deterministic endgame play)
        
        Returns:
            training_examples: List of (board_state, mcts_policy, player) tuples
            winner: Final game outcome (1, -1, or 0)
        """
        game = TicTacToeGame()
        training_examples = []
        move_count = 0
        
        while not game.is_game_over():
            move_count += 1
            
            # Record current player (before move is made)
            current_player = game.current_player
            
            # Get canonical board (always from current player's perspective)
            canonical_board = self._get_canonical_board(game.board, current_player)
            
            # Run MCTS to get action probabilities
            action_probs = self.mcts.search(game)
            
            # Store training example: (board, policy, player)
            training_examples.append((
                canonical_board.copy(),
                action_probs.copy(),
                current_player
            ))
            
            # Select move based on temperature
            temp = temperature if move_count < temperature_threshold else 0
            if temp == 0:
                # Greedy: pick best move
                action = action_probs.argmax()
            else:
                # Stochastic: sample from distribution
                # Apply temperature
                action_probs_temp = action_probs ** (1.0 / temp)
                action_probs_temp = action_probs_temp / action_probs_temp.sum()
                action = np.random.choice(9, p=action_probs_temp)
            
            # Make the move
            game.make_move(action)
        
        # Game is over, get winner
        winner = game.check_winner()
        
        # Convert training examples to include game outcome
        # outcome is from the perspective of the player who made the move
        training_data = []
        for board, policy, player in training_examples:
            # Value is game outcome from this player's perspective
            if winner == 0:
                value = 0  # Draw
            else:
                value = 1.0 if winner == player else -1.0
            
            training_data.append((board, policy, value))
        
        return training_data, winner
    
    def generate_training_data(self, num_games=100, verbose=True, use_augmentation=False, random_opponent_fraction=0.0, num_workers=None):
        """
        Generate training data by playing multiple self-play games in parallel.
        
        Args:
            num_games: Number of games to play
            verbose: Whether to print progress
            use_augmentation: Whether to use symmetry augmentation (8x data) - USE WITH CAUTION
            random_opponent_fraction: Fraction of games where opponent plays randomly (0.0-1.0)
            num_workers: Number of parallel workers (None = use CPU count)
        
        Returns:
            training_data: List of (board, policy, value) tuples
            game_stats: Dictionary with win/draw statistics
        """
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        all_training_data = []
        game_stats = {'X_wins': 0, 'O_wins': 0, 'draws': 0, 
                     'random_games': 0, 'mcts_games': 0}
        
        num_random_games = int(num_games * random_opponent_fraction)
        num_mcts_games = num_games - num_random_games
        
        # Get network state dict for workers
        network_state_dict = self.network.state_dict()
        
        if verbose:
            print(f"Using {num_workers} parallel workers for game generation...")
        
        # Play MCTS self-play games in parallel
        if num_mcts_games > 0:
            if verbose:
                print(f"\nGenerating {num_mcts_games} MCTS self-play games...")
            
            worker_func = partial(
                _play_single_mcts_game,
                network_state_dict,
                self.mcts.num_simulations,
                self.mcts.c_puct,
                1.0,  # temperature
                15    # temperature_threshold
            )
            
            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(worker_func, range(num_mcts_games))
            
            # Process results
            for training_data, winner in results:
                # Apply data augmentation if enabled
                if use_augmentation:
                    augmented_data = []
                    for board, policy, value in training_data:
                        augmentations = augment_tictactoe_data(board, policy)
                        for aug_board, aug_policy in augmentations:
                            augmented_data.append((aug_board, aug_policy, value))
                    training_data = augmented_data
                
                all_training_data.extend(training_data)
                game_stats['mcts_games'] += 1
                
                if winner == 1:
                    game_stats['X_wins'] += 1
                elif winner == -1:
                    game_stats['O_wins'] += 1
                else:
                    game_stats['draws'] += 1
            
            if verbose:
                print(f"Completed {num_mcts_games} MCTS games "
                      f"(X: {game_stats['X_wins']}, O: {game_stats['O_wins']}, "
                      f"Draw: {game_stats['draws']})")
        
        # Play against random opponent in parallel
        if num_random_games > 0:
            if verbose:
                print(f"\nGenerating {num_random_games} games against random opponent...")
            
            worker_func = partial(
                _play_single_random_game,
                network_state_dict,
                self.mcts.num_simulations,
                self.mcts.c_puct,
                1.0,  # temperature
                15    # temperature_threshold
            )
            
            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(worker_func, range(num_random_games))
            
            # Process results
            random_stats = {'wins': 0, 'losses': 0, 'draws': 0}
            for game_idx, (training_data, winner) in enumerate(results):
                if use_augmentation:
                    augmented_data = []
                    for board, policy, value in training_data:
                        augmentations = augment_tictactoe_data(board, policy)
                        for aug_board, aug_policy in augmentations:
                            augmented_data.append((aug_board, aug_policy, value))
                    training_data = augmented_data
                
                all_training_data.extend(training_data)
                game_stats['random_games'] += 1
                
                # Track wins from network's perspective
                network_plays_first = (game_idx % 2 == 0)
                if network_plays_first:
                    # Network played as X
                    if winner == 1:
                        random_stats['wins'] += 1
                    elif winner == -1:
                        random_stats['losses'] += 1
                    else:
                        random_stats['draws'] += 1
                else:
                    # Network played as O
                    if winner == -1:
                        random_stats['wins'] += 1
                    elif winner == 1:
                        random_stats['losses'] += 1
                    else:
                        random_stats['draws'] += 1
            
            if verbose:
                print(f"Completed {num_random_games} random opponent games "
                      f"(Wins: {random_stats['wins']}, Losses: {random_stats['losses']}, "
                      f"Draws: {random_stats['draws']})")
        
        if verbose:
            aug_note = " (with 8x augmentation)" if use_augmentation else ""
            print(f"\nTotal training examples: {len(all_training_data)}{aug_note}")
            print(f"MCTS games: {game_stats['mcts_games']}, Random opponent games: {game_stats['random_games']}")
        
        return all_training_data, game_stats
    
    def play_game_vs_random(self, temperature=1.0, temperature_threshold=15, network_plays_first=True):
        """
        Play a game where the network plays against a random opponent.
        Only the network's moves are recorded for training.
        
        Args:
            temperature: Temperature for move selection
            temperature_threshold: Move number after which temperature drops to 0
            network_plays_first: If True, network plays X (first), else O (second)
        
        Returns:
            training_examples: List of (board, policy, value) tuples (only network's turns)
            winner: Final game outcome (1, -1, or 0)
        """
        game = TicTacToeGame()
        training_examples = []
        move_count = 0
        
        # Determine which player the network controls
        network_player = 1 if network_plays_first else -1
        
        while not game.is_game_over():
            move_count += 1
            
            if game.current_player == network_player:
                # Network's turn (X) - use MCTS
                current_player = game.current_player
                canonical_board = self._get_canonical_board(game.board, current_player)
                action_probs = self.mcts.search(game)
                
                # Store training example
                training_examples.append((
                    canonical_board.copy(),
                    action_probs.copy(),
                    current_player
                ))
                
                # Select move
                temp = temperature if move_count < temperature_threshold else 0
                if temp == 0:
                    action = action_probs.argmax()
                else:
                    action_probs_temp = action_probs ** (1.0 / temp)
                    action_probs_temp = action_probs_temp / action_probs_temp.sum()
                    action = np.random.choice(9, p=action_probs_temp)
                
                game.make_move(action)
            else:
                # Random opponent's turn (O)
                legal_moves = game.get_legal_moves()
                action = np.random.choice(legal_moves)
                game.make_move(action)
        
        # Game is over, get winner
        winner = game.check_winner()
        
        # Convert training examples to include game outcome (from network's perspective)
        training_data = []
        for board, policy, player in training_examples:
            if winner == 0:
                value = 0  # Draw
            else:
                value = 1.0 if winner == player else -1.0
            
            training_data.append((board, policy, value))
        
        return training_data, winner
    
    def _get_canonical_board(self, board, player):
        """
        Get the canonical form of the board (from current player's perspective).
        For tic-tac-toe, we multiply the board by the player to normalize.
        This makes it so the current player is always represented by 1.
        """
        return board * player


class TrainingDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for training data.
    """
    def __init__(self, training_data):
        """
        Args:
            training_data: List of (board, policy, value) tuples
        """
        self.boards = []
        self.policies = []
        self.values = []
        
        for board, policy, value in training_data:
            self.boards.append(board)
            self.policies.append(policy)
            self.values.append(value)
        
        self.boards = np.array(self.boards, dtype=np.float32)
        self.policies = np.array(self.policies, dtype=np.float32)
        self.values = np.array(self.values, dtype=np.float32)
    
    def __len__(self):
        return len(self.boards)
    
    def __getitem__(self, idx):
        board = torch.tensor(self.boards[idx], dtype=torch.float32)
        policy = torch.tensor(self.policies[idx], dtype=torch.float32)
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return board, policy, value
