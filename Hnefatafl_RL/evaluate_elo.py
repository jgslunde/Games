#!/usr/bin/env python3
"""
Evaluate relative ELO ratings for all checkpoints in a directory.

This script plays matches between checkpoints and solves for their relative ELO ratings.
The ELO system is computed using a global optimization to best fit all match results.

Usage:
    python evaluate_elo.py <checkpoint_dir> [options]

Example:
    python evaluate_elo.py checkpoints_brandubh/fatman_nothrone/ --num-games 20 --checkpoint-distance 5
    python evaluate_elo.py checkpoints_tablut/jack_nothrone/ --game tablut --simulations 400
    python evaluate_elo.py checkpoints_brandubh/jack_nothrone/ --num-games 10 --max-checkpoints 20
"""

import argparse
import os
import glob
import re
import json
import numpy as np
import torch
from typing import List, Tuple, Optional
import time
from scipy.optimize import minimize
import multiprocessing as mp
from functools import partial

# Import game-specific modules
from brandubh import Brandubh
from tablut import Tablut
from hnefatafl import Hnefatafl
from network import BrandubhNet, MoveEncoder
from network_tablut import TablutNet, TablutMoveEncoder
from network_hnefatafl import HnefataflNet, HnefataflMoveEncoder
from agent import Agent


def extract_iteration_number(checkpoint_path: str) -> Optional[int]:
    """
    Extract iteration number from checkpoint filename.
    
    Expected format: checkpoint_iter_X.pth
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Iteration number or None if not found
    """
    basename = os.path.basename(checkpoint_path)
    
    # Try to match checkpoint_iter_X.pth pattern
    match = re.search(r'checkpoint_iter_(\d+)\.pth', basename)
    if match:
        return int(match.group(1))
    
    # Try other common patterns
    match = re.search(r'iter_(\d+)', basename)
    if match:
        return int(match.group(1))
    
    match = re.search(r'checkpoint_(\d+)', basename)
    if match:
        return int(match.group(1))
    
    return None


def load_checkpoint(checkpoint_path: str, network_class, move_encoder_class, device: str = 'cpu'):
    """
    Load a neural network from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        network_class: Network class to instantiate
        move_encoder_class: Move encoder class
        device: Device to load on
    
    Returns:
        Loaded network
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract config
        if 'config' in checkpoint:
            config = checkpoint['config']
            num_res_blocks = config.get('num_res_blocks', 4)
            num_channels = config.get('num_channels', 64)
        else:
            num_res_blocks = checkpoint.get('num_res_blocks', 4)
            num_channels = checkpoint.get('num_channels', 64)
        
        # Create and load network
        network = network_class(num_res_blocks=num_res_blocks, num_channels=num_channels)
        
        # Handle different checkpoint formats
        if 'network_state_dict' in checkpoint:
            network.load_state_dict(checkpoint['network_state_dict'])
        elif 'model_state_dict' in checkpoint:
            network.load_state_dict(checkpoint['model_state_dict'])
        else:
            network.load_state_dict(checkpoint)
        
        network.eval()
        return network
        
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint {checkpoint_path}: {e}")


def play_single_game(agent1_is_attacker: bool, game_class, game_rules: dict, network1_path: str, 
                    network2_path: str, network_class, move_encoder_class, num_simulations: int,
                    c_puct: float, device: str = 'cpu') -> int:
    """
    Play a single game between two checkpoints (worker function for multiprocessing).
    
    Args:
        agent1_is_attacker: Whether agent1 plays as attacker (FIRST argument for partial)
        game_class: Game class to instantiate
        game_rules: Dictionary of game rules
        network1_path: Path to first checkpoint
        network2_path: Path to second checkpoint
        network_class: Network class to instantiate
        move_encoder_class: Move encoder class
        num_simulations: MCTS simulations per move
        c_puct: MCTS exploration constant
        device: Device to use (cpu/cuda)
    
    Returns:
        0 if agent1 wins, 1 if agent2 wins, 2 if draw
    """
    # Load networks
    network1 = load_checkpoint(network1_path, network_class, move_encoder_class, device)
    network2 = load_checkpoint(network2_path, network_class, move_encoder_class, device)
    
    # Create agents
    agent1 = Agent(network1, num_simulations=num_simulations, c_puct=c_puct,
                  device=device, move_encoder_class=move_encoder_class)
    agent2 = Agent(network2, num_simulations=num_simulations, c_puct=c_puct,
                  device=device, move_encoder_class=move_encoder_class)
    
    # Set up agents based on colors
    if agent1_is_attacker:
        agents = [agent1, agent2]  # agent1 as attacker, agent2 as defender
    else:
        agents = [agent2, agent1]  # agent2 as attacker, agent1 as defender
    
    # Play the game
    game = game_class(**game_rules)
    move_count = 0
    
    while not game.game_over:
        agent = agents[game.current_player]
        move = agent.select_move(game, temperature=0.0)  # Deterministic play
        
        if move is None:
            game.game_over = True
            game.winner = 1 - game.current_player
            break
        
        game.make_move(move)
        move_count += 1
        
        # Prevent infinite games
        if move_count > 500:
            game.game_over = True
            game.winner = None
            break
    
    # Return result from agent1's perspective
    if game.winner is None:
        return 2  # Draw
    elif agent1_is_attacker:
        # agent1 was attacker
        return 0 if game.winner == 0 else 1
    else:
        # agent1 was defender
        return 0 if game.winner == 1 else 1


def play_match(checkpoint1_path: str, checkpoint2_path: str, game_class, game_rules: dict,
              network_class, move_encoder_class, num_simulations: int, c_puct: float,
              num_games: int = 10, num_workers: int = 1, device: str = 'cpu') -> Tuple[int, int, int]:
    """
    Play multiple games between two checkpoints, alternating colors.
    Uses multiprocessing to parallelize games.
    
    Args:
        checkpoint1_path: Path to first checkpoint
        checkpoint2_path: Path to second checkpoint
        game_class: Game class to instantiate
        game_rules: Dictionary of game rules
        network_class: Network class to instantiate
        move_encoder_class: Move encoder class
        num_simulations: MCTS simulations per move
        c_puct: MCTS exploration constant
        num_games: Number of games to play (each agent plays num_games//2 as each color)
        num_workers: Number of parallel workers
        device: Device to use (cpu/cuda)
    
    Returns:
        (checkpoint1_wins, checkpoint2_wins, draws)
    """
    # Create list of games to play
    game_configs = []
    for i in range(num_games):
        agent1_is_attacker = (i % 2 == 0)
        game_configs.append(agent1_is_attacker)
    
    # Create worker function with fixed parameters
    worker_func = partial(
        play_single_game,
        game_class=game_class,
        game_rules=game_rules,
        network1_path=checkpoint1_path,
        network2_path=checkpoint2_path,
        network_class=network_class,
        move_encoder_class=move_encoder_class,
        num_simulations=num_simulations,
        c_puct=c_puct,
        device=device
    )
    
    # Play games in parallel
    if num_workers > 1:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(worker_func, game_configs)
    else:
        # Sequential execution for debugging
        results = [worker_func(config) for config in game_configs]
    
    # Count results
    checkpoint1_wins = results.count(0)
    checkpoint2_wins = results.count(1)
    draws = results.count(2)
    
    return checkpoint1_wins, checkpoint2_wins, draws


def expected_score(elo_diff: float) -> float:
    """
    Calculate expected score given ELO difference.
    
    Args:
        elo_diff: ELO difference (player1 - player2)
    
    Returns:
        Expected score for player1 (0 to 1)
    """
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def compute_elo_ratings(match_results: List[Tuple[int, int, float]], 
                       num_checkpoints: int,
                       base_elo: float = 1500.0) -> np.ndarray:
    """
    Compute ELO ratings that best fit the match results.
    
    Uses maximum likelihood estimation to find ELO ratings that minimize
    the cross-entropy loss between predicted and observed match outcomes.
    
    Args:
        match_results: List of (checkpoint1_idx, checkpoint2_idx, score1) tuples
                      where score1 is the fraction of points scored by checkpoint1
        num_checkpoints: Total number of checkpoints
        base_elo: Base ELO rating (first checkpoint gets this rating)
    
    Returns:
        Array of ELO ratings for each checkpoint
    """
    if not match_results:
        return np.full(num_checkpoints, base_elo)
    
    # Initialize ELO ratings (relative to first checkpoint)
    # We'll optimize differences from the base
    initial_elos = np.zeros(num_checkpoints)
    
    def loss_function(elo_diffs):
        """
        Cross-entropy loss between predicted and observed scores.
        
        Args:
            elo_diffs: ELO differences from base (first checkpoint = 0)
        
        Returns:
            Total loss
        """
        total_loss = 0.0
        epsilon = 1e-10  # Small value to prevent log(0)
        
        for idx1, idx2, actual_score in match_results:
            # Calculate ELO difference
            elo_diff = elo_diffs[idx1] - elo_diffs[idx2]
            
            # Predicted score for checkpoint 1
            predicted_score = expected_score(elo_diff)
            
            # Cross-entropy loss
            # -[s * log(p) + (1-s) * log(1-p)]
            loss = -(actual_score * np.log(predicted_score + epsilon) + 
                    (1 - actual_score) * np.log(1 - predicted_score + epsilon))
            
            total_loss += loss
        
        return total_loss
    
    # Optimize ELO ratings
    # Constraint: first checkpoint has ELO = 0 (relative rating)
    def constraint_first_zero(elo_diffs):
        return elo_diffs[0]
    
    constraints = {'type': 'eq', 'fun': constraint_first_zero}
    
    result = minimize(
        loss_function,
        initial_elos,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")
    
    # Convert relative ratings to absolute ratings
    elo_ratings = result.x + base_elo
    
    return elo_ratings


def save_results(checkpoint_dir: str, 
                checkpoints: List[str], 
                elo_ratings: np.ndarray,
                match_results: List[Tuple[int, int, int, int, int]],
                args: argparse.Namespace):
    """
    Save ELO evaluation results to JSON file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        checkpoints: List of checkpoint paths
        elo_ratings: Array of ELO ratings
        match_results: List of (idx1, idx2, wins1, wins2, draws) tuples
        args: Command-line arguments
    """
    results = {
        'metadata': {
            'checkpoint_dir': checkpoint_dir,
            'game': args.game,
            'num_games_per_match': args.num_games,
            'checkpoint_distance': args.checkpoint_distance,
            'simulations': args.simulations,
            'base_elo': args.base_elo,
            'total_matches': len(match_results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'checkpoints': [],
        'matches': [],
    }
    
    # Add checkpoint information
    for i, checkpoint_path in enumerate(checkpoints):
        iteration = extract_iteration_number(checkpoint_path)
        results['checkpoints'].append({
            'index': i,
            'filename': os.path.basename(checkpoint_path),
            'iteration': iteration,
            'elo': float(elo_ratings[i]),
        })
    
    # Add match results
    for idx1, idx2, wins1, wins2, draws in match_results:
        results['matches'].append({
            'checkpoint1_index': idx1,
            'checkpoint2_index': idx2,
            'checkpoint1_wins': wins1,
            'checkpoint2_wins': wins2,
            'draws': draws,
            'total_games': wins1 + wins2 + draws,
        })
    
    # Save to JSON file
    output_path = os.path.join(checkpoint_dir, 'elo_evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_results_table(checkpoints: List[str], elo_ratings: np.ndarray):
    """
    Print a formatted table of ELO ratings.
    
    Args:
        checkpoints: List of checkpoint paths
        elo_ratings: Array of ELO ratings
    """
    # Sort by ELO rating (descending)
    sorted_indices = np.argsort(-elo_ratings)
    
    print("\n" + "=" * 80)
    print("ELO RATINGS")
    print("=" * 80)
    print(f"{'Rank':<6} {'Iteration':<12} {'ELO':<10} {'Checkpoint':<50}")
    print("-" * 80)
    
    for rank, idx in enumerate(sorted_indices, 1):
        checkpoint = os.path.basename(checkpoints[idx])
        iteration = extract_iteration_number(checkpoints[idx])
        iteration_str = str(iteration) if iteration is not None else 'N/A'
        elo = elo_ratings[idx]
        
        print(f"{rank:<6} {iteration_str:<12} {elo:<10.1f} {checkpoint:<50}")
    
    print("=" * 80)
    
    # Print statistics
    print("\nELO Statistics:")
    print(f"  Mean:   {np.mean(elo_ratings):.1f}")
    print(f"  Median: {np.median(elo_ratings):.1f}")
    print(f"  Std:    {np.std(elo_ratings):.1f}")
    print(f"  Range:  {np.min(elo_ratings):.1f} - {np.max(elo_ratings):.1f}")
    print(f"  Spread: {np.max(elo_ratings) - np.min(elo_ratings):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate relative ELO ratings for checkpoints")
    
    # Required arguments
    parser.add_argument("checkpoint_dir", type=str,
                       help="Directory containing checkpoint files")
    
    # Game settings
    parser.add_argument("--game", type=str, default="brandubh",
                       choices=["brandubh", "tablut", "hnefatafl"],
                       help="Game variant (default: brandubh)")
    parser.add_argument("--simulations", type=int, default=200,
                       help="MCTS simulations per move (default: 200)")
    parser.add_argument("--c-puct", type=float, default=1.4,
                       help="MCTS exploration constant (default: 1.4)")
    
    # Match settings
    parser.add_argument("--num-games", type=int, default=20,
                       help="Number of games per match (default: 20)")
    parser.add_argument("--checkpoint-distance", type=int, default=10,
                       help="Maximum distance between checkpoints to match (default: 10)")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count(),
                       help=f"Number of parallel workers (default: {mp.cpu_count()} CPUs)")
    
    # Checkpoint selection
    parser.add_argument("--pattern", type=str, default="checkpoint_iter_*.pth",
                       help="Checkpoint file pattern (default: checkpoint_iter_*.pth)")
    parser.add_argument("--max-checkpoints", type=int, default=None,
                       help="Maximum number of checkpoints to evaluate (default: all)")
    parser.add_argument("--skip-checkpoints", type=int, default=1,
                       help="Evaluate every Nth checkpoint (default: 1, i.e., all)")
    
    # ELO settings
    parser.add_argument("--base-elo", type=float, default=1500.0,
                       help="Base ELO rating for first checkpoint (default: 1500.0)")
    
    # Game rules (optional overrides)
    parser.add_argument("--king-capture-pieces", type=int, default=None, choices=[2, 3, 4],
                       help="Number of pieces to capture king (default: from checkpoint)")
    parser.add_argument("--throne-enabled", type=bool, default=None,
                       help="Whether throne is enabled (default: from checkpoint)")
    
    # System settings
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run on (default: cpu)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing results file if found")
    
    args = parser.parse_args()
    
    # Validate checkpoint directory
    if not os.path.isdir(args.checkpoint_dir):
        print(f"Error: Directory not found: {args.checkpoint_dir}")
        return
    
    # Select game configuration
    if args.game == "tablut":
        game_class = Tablut
        network_class = TablutNet
        move_encoder_class = TablutMoveEncoder
        board_size = 9
    elif args.game == "hnefatafl":
        game_class = Hnefatafl
        network_class = HnefataflNet
        move_encoder_class = HnefataflMoveEncoder
        board_size = 11
    else:  # brandubh
        game_class = Brandubh
        network_class = BrandubhNet
        move_encoder_class = MoveEncoder
        board_size = 7
    
    # Game rules (use defaults or checkpoint values)
    game_rules = {}
    if args.king_capture_pieces is not None:
        game_rules['king_capture_pieces'] = args.king_capture_pieces
    if args.throne_enabled is not None:
        game_rules['throne_enabled'] = args.throne_enabled
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(args.checkpoint_dir, args.pattern)
    all_checkpoints = sorted(glob.glob(checkpoint_pattern))
    
    if not all_checkpoints:
        print(f"Error: No checkpoint files found matching: {checkpoint_pattern}")
        return
    
    # Filter checkpoints
    # Sort by iteration number if available
    checkpoints_with_iter = []
    for cp in all_checkpoints:
        iteration = extract_iteration_number(cp)
        if iteration is not None:
            checkpoints_with_iter.append((iteration, cp))
    
    if checkpoints_with_iter:
        checkpoints_with_iter.sort(key=lambda x: x[0])
        all_checkpoints = [cp for _, cp in checkpoints_with_iter]
    
    # Apply skip and max filters
    if args.skip_checkpoints > 1:
        all_checkpoints = all_checkpoints[::args.skip_checkpoints]
    
    if args.max_checkpoints is not None:
        all_checkpoints = all_checkpoints[:args.max_checkpoints]
    
    checkpoints = all_checkpoints
    num_checkpoints = len(checkpoints)
    
    print("=" * 80)
    print(f"ELO Evaluation - {args.game.capitalize()} ({board_size}x{board_size})")
    print("=" * 80)
    print(f"Checkpoint directory:  {args.checkpoint_dir}")
    print(f"Number of checkpoints: {num_checkpoints}")
    print(f"Games per match:       {args.num_games}")
    print(f"Checkpoint distance:   {args.checkpoint_distance}")
    print(f"MCTS simulations:      {args.simulations}")
    print(f"Parallel workers:      {args.num_workers}")
    print(f"Device:                {args.device}")
    print("=" * 80)
    print()
    
    # Check for existing results
    results_path = os.path.join(args.checkpoint_dir, 'elo_evaluation_results.json')
    existing_matches = {}
    
    if args.resume and os.path.exists(results_path):
        print(f"Loading existing results from: {results_path}")
        try:
            with open(results_path, 'r') as f:
                existing_data = json.load(f)
            
            # Build map of existing matches
            for match in existing_data['matches']:
                idx1 = match['checkpoint1_index']
                idx2 = match['checkpoint2_index']
                key = (min(idx1, idx2), max(idx1, idx2))
                existing_matches[key] = (
                    match['checkpoint1_wins'],
                    match['checkpoint2_wins'],
                    match['draws']
                )
            
            print(f"Found {len(existing_matches)} existing matches")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
            existing_matches = {}
    
    # Determine which matches to play
    matches_to_play = []
    for i in range(num_checkpoints):
        for j in range(i + 1, min(i + 1 + args.checkpoint_distance, num_checkpoints)):
            key = (i, j)
            if key not in existing_matches:
                matches_to_play.append((i, j))
    
    total_matches = len(matches_to_play) + len(existing_matches)
    
    print(f"\nTotal matches needed:  {total_matches}")
    print(f"Existing matches:      {len(existing_matches)}")
    print(f"New matches to play:   {len(matches_to_play)}")
    print()
    
    # Play matches
    match_results_raw = []  # (idx1, idx2, wins1, wins2, draws)
    
    # Add existing matches
    for (idx1, idx2), (wins1, wins2, draws) in existing_matches.items():
        match_results_raw.append((idx1, idx2, wins1, wins2, draws))
    
    # Play new matches
    for match_num, (i, j) in enumerate(matches_to_play, 1):
        checkpoint1 = checkpoints[i]
        checkpoint2 = checkpoints[j]
        
        iter1 = extract_iteration_number(checkpoint1)
        iter2 = extract_iteration_number(checkpoint2)
        
        print(f"[{match_num}/{len(matches_to_play)}] Playing match: "
              f"iter_{iter1 if iter1 else '?'} vs iter_{iter2 if iter2 else '?'}... ", 
              end='', flush=True)
        
        try:
            # Play match using multiprocessing
            wins1, wins2, draws = play_match(
                checkpoint1_path=checkpoint1,
                checkpoint2_path=checkpoint2,
                game_class=game_class,
                game_rules=game_rules,
                network_class=network_class,
                move_encoder_class=move_encoder_class,
                num_simulations=args.simulations,
                c_puct=args.c_puct,
                num_games=args.num_games,
                num_workers=args.num_workers,
                device=args.device
            )
            
            match_results_raw.append((i, j, wins1, wins2, draws))
            
            print(f"Result: {wins1}-{wins2}-{draws} (W-L-D for checkpoint {i})")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("All matches completed. Computing ELO ratings...")
    print("=" * 80)
    
    # Convert match results to format for ELO computation
    # Each match result becomes a score: wins / total_games
    match_results_for_elo = []
    
    for idx1, idx2, wins1, wins2, draws in match_results_raw:
        total_games = wins1 + wins2 + draws
        if total_games > 0:
            # Score for checkpoint 1: (wins + 0.5 * draws) / total_games
            score1 = (wins1 + 0.5 * draws) / total_games
            match_results_for_elo.append((idx1, idx2, score1))
    
    # Compute ELO ratings
    elo_ratings = compute_elo_ratings(match_results_for_elo, num_checkpoints, args.base_elo)
    
    # Print results
    print_results_table(checkpoints, elo_ratings)
    
    # Save results
    save_results(args.checkpoint_dir, checkpoints, elo_ratings, match_results_raw, args)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    main()
