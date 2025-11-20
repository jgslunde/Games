"""
Play Tafl games between two AI agents loaded from checkpoints.
Displays the game state and move history.

Supports Brandubh (7x7), Tablut (9x9), and Hnefatafl (11x11) variants.

Usage:
    python play.py <checkpoint1> <checkpoint2> [options]
    
Example:
    python play.py checkpoints/best_model.pth checkpoints/checkpoint_iter_5.pth
    python play.py model1.pth model2.pth --game tablut --simulations 200
    python play.py model1.pth model2.pth --game hnefatafl --simulations 200
    python play.py model1.pth model2.pth --king-capture-pieces 4 --throne-is-hostile
    python play.py model1.pth model2.pth --num-games 10 --swap-colors  # 20 games total
    python play.py model1.pth model2.pth --num-games 50 --num-workers 16  # Parallel games
    python play.py model1.pth model2.pth --time-per-move 1.0  # 1 second per move
    python play.py model1.pth model2.pth --num-games 100 --num-workers 32 --time-per-move 0.5
"""

import argparse
import sys
import torch
import multiprocessing as mp
import numpy as np

from brandubh import Brandubh
from tablut import Tablut
from hnefatafl import Hnefatafl
from network import BrandubhNet
from network_tablut import TablutNet
from network_hnefatafl import HnefataflNet
from agent import Agent


def play_single_game_worker(game_idx, checkpoint1_path, checkpoint2_path, agent1_plays_attacker, 
                              game_name, simulations, king_capture_pieces, king_can_capture,
                              throne_is_hostile, throne_enabled, time_per_move, temperature):
    """
    Worker function for playing a single game in a separate process.
    Returns tuple: (agent1_won, winner_role, move_count)
    """
    try:
        # Set torch to use only 1 thread per worker to avoid conflicts
        # This MUST be done before creating any torch objects
        torch.set_num_threads(1)
        
        # Create game environment
        if game_name == 'brandubh':
            game_cls = Brandubh
            network_cls = BrandubhNet
        elif game_name == 'tablut':
            game_cls = Tablut
            network_cls = TablutNet
        elif game_name == 'hnefatafl':
            game_cls = Hnefatafl
            network_cls = HnefataflNet
        else:
            raise ValueError(f"Unknown game: {game_name}")
        
        game_state = game_cls(
            king_capture_pieces=king_capture_pieces,
            king_can_capture=king_can_capture,
            throne_is_hostile=throne_is_hostile,
            throne_enabled=throne_enabled
        )
        
        # Load models
        device = torch.device('cpu')
        
        # Load checkpoints
        checkpoint1 = torch.load(checkpoint1_path, map_location=device, weights_only=False)
        checkpoint2 = torch.load(checkpoint2_path, map_location=device, weights_only=False)
        
        # Extract architecture config from checkpoints
        if 'config' in checkpoint1:
            config1 = checkpoint1['config']
            num_res_blocks1 = config1.get('num_res_blocks', 4)
            num_channels1 = config1.get('num_channels', 64)
        else:
            num_res_blocks1 = checkpoint1.get('num_res_blocks', 4)
            num_channels1 = checkpoint1.get('num_channels', 64)
        
        if 'config' in checkpoint2:
            config2 = checkpoint2['config']
            num_res_blocks2 = config2.get('num_res_blocks', 4)
            num_channels2 = config2.get('num_channels', 64)
        else:
            num_res_blocks2 = checkpoint2.get('num_res_blocks', 4)
            num_channels2 = checkpoint2.get('num_channels', 64)
        
        # Create networks with correct architecture
        network1 = network_cls(num_res_blocks=num_res_blocks1, num_channels=num_channels1).to(device)
        network2 = network_cls(num_res_blocks=num_res_blocks2, num_channels=num_channels2).to(device)
        
        # Load state dicts - handle different checkpoint formats
        if 'network_state_dict' in checkpoint1:
            network1.load_state_dict(checkpoint1['network_state_dict'])
        elif 'model_state_dict' in checkpoint1:
            network1.load_state_dict(checkpoint1['model_state_dict'])
        else:
            network1.load_state_dict(checkpoint1)
        
        if 'network_state_dict' in checkpoint2:
            network2.load_state_dict(checkpoint2['network_state_dict'])
        elif 'model_state_dict' in checkpoint2:
            network2.load_state_dict(checkpoint2['model_state_dict'])
        else:
            network2.load_state_dict(checkpoint2)
        
        network1.eval()
        network2.eval()
        
        # Create agents
        agent1 = Agent(network1, num_simulations=simulations, device=device)
        agent2 = Agent(network2, num_simulations=simulations, device=device)
        
        # Play game
        move_count = 0
        max_moves = 200
        
        # Debug: Check initial state
        if game_state.game_over:
            # Game is already over before we start - this shouldn't happen
            return (None, 'error', 0)
        
        while not game_state.game_over and move_count < max_moves:
            current_player_is_attacker = (game_state.current_player == 0)
            
            # Determine which agent makes the move
            if current_player_is_attacker == agent1_plays_attacker:
                current_agent = agent1
            else:
                current_agent = agent2
            
            # Select move with time limit if specified
            if time_per_move is not None:
                move, value, visit_prob = current_agent.select_move_with_time_limit(game_state, time_per_move, temperature=temperature)
            else:
                move, value, visit_prob = current_agent.select_move_with_stats(game_state, temperature=temperature)
            
            if move is None:
                # No legal moves available - game should be over
                break
            
            game_state.make_move(move)
            move_count += 1
        
        # Determine winner
        if game_state.game_over:
            winner = game_state.winner
            if winner == 0:  # Attacker won
                winner_role = 'attacker'
                agent1_won = agent1_plays_attacker
            elif winner == 1:  # Defender won
                winner_role = 'defender'
                agent1_won = not agent1_plays_attacker
            else:  # Draw (winner is None)
                winner_role = 'draw'
                agent1_won = None
        else:
            # Max moves reached - consider it a draw
            winner_role = 'draw'
            agent1_won = None
        
        return (agent1_won, winner_role, move_count)
        
    except Exception as e:
        import traceback
        print(f"Error in worker {game_idx}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Return a recognizable error state
        return (None, 'error', -1)


def load_checkpoint_with_rules(checkpoint_path: str, force_rules: dict = None, game_class=None):
    """
    Load neural network from checkpoint and extract rules.
    
    Args:
        checkpoint_path: Path to checkpoint file
        force_rules: Optional dict of rules to force (overrides checkpoint)
        game_class: Game class (Brandubh, Tablut, or Hnefatafl) - determines network type
    
    Returns:
        (network, rules_dict) tuple
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            num_res_blocks = config.get('num_res_blocks', 4)
            num_channels = config.get('num_channels', 64)
            board_size = config.get('board_size', 7)  # Extract board size from config
            
            # Extract game rules from checkpoint (can be overridden)
            if force_rules is None:
                rules = {
                    'king_capture_pieces': config.get('king_capture_pieces', 2),
                    'king_can_capture': config.get('king_can_capture', True),
                    'throne_is_hostile': config.get('throne_is_hostile', False),
                    'throne_enabled': config.get('throne_enabled', True),
                }
            else:
                rules = force_rules
        else:
            # Default values
            num_res_blocks = 4
            num_channels = 64
            board_size = 7  # Default to Brandubh
            rules = force_rules if force_rules else {
                'king_capture_pieces': 2,
                'king_can_capture': True,
                'throne_is_hostile': False,
                'throne_enabled': True,
            }
        
        # Determine network type based on game_class or board_size
        if game_class == Tablut or board_size == 9:
            network = TablutNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
        elif game_class == Hnefatafl or board_size == 11:
            network = HnefataflNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
        else:
            network = BrandubhNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
        
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()
        
        print(f"Loaded network from {checkpoint_path}")
        print(f"  Architecture: {num_res_blocks} residual blocks, {num_channels} channels")
        
        return network, rules
        
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        sys.exit(1)


def play_game_between_agents(agent1, agent2, game_class, rules, display=True, temperature=0.0, time_per_move=None):
    """
    Play a game between two AI agents.
    
    Args:
        agent1: Agent playing as attacker
        agent2: Agent playing as defender
        game_class: Game class (Brandubh or Tablut)
        rules: Dictionary of game rules
        display: Whether to print the board state
        temperature: Temperature for move selection (0=deterministic)
        time_per_move: Optional time limit per move in seconds (if None, uses simulations)
    
    Returns:
        winner: 0 for Attackers, 1 for Defenders, None for draw
        num_moves: Number of moves in the game
    """
    game = game_class(**rules)
    move_count = 0
    
    if display:
        print("=" * 50)
        print(f"Starting new game of {game_class.__name__}")
        print("=" * 50)
        print(f"Rules: King capture pieces={rules['king_capture_pieces']}, "
              f"King can capture={rules['king_can_capture']}, "
              f"Throne hostile={rules['throne_is_hostile']}, "
              f"Throne enabled={rules['throne_enabled']}")
        if time_per_move is not None:
            print(f"Time per move: {time_per_move} seconds")
        print("\nInitial board:")
        print(game)
        print("\n")
    
    while not game.game_over:
        # Get current agent
        current_agent = agent1 if game.current_player == 0 else agent2
        
        # Get move from agent with statistics (using time limit if specified)
        if time_per_move is not None:
            move, value, visit_prob = current_agent.select_move_with_time_limit(game, time_per_move, temperature=temperature)
        else:
            move, value, visit_prob = current_agent.select_move_with_stats(game, temperature=temperature)
        
        if move is None:
            # No legal moves - opponent wins
            game.game_over = True
            game.winner = 1 - game.current_player
            break
        
        from_r, from_c, to_r, to_c = move
        
        if display:
            player_name = "Attackers" if game.current_player == 0 else "Defenders"
            # Convert value to attacker's perspective (value is from current player's perspective)
            attacker_value = value if game.current_player == 0 else -value
            print(f"Move {move_count + 1}: {player_name} move ({from_r},{from_c}) → ({to_r},{to_c}) "
                  f"[Eval: {attacker_value:+.3f}, Confidence: {visit_prob*100:.1f}%]")
        
        # Make move
        game.make_move(move)
        move_count += 1
        
        if display:
            print(game)
            print("\n")
        
        # Safety check to prevent infinite games
        if move_count > 500:
            if display:
                print("Game exceeded 500 moves - draw by move limit")
            game.game_over = True
            game.winner = None
            break
    
    if display:
        print("=" * 50)
        if game.winner is None:
            print("Game Over! Result: Draw")
        else:
            winner_name = "Attackers" if game.winner == 0 else "Defenders"
            print(f"Game Over! Winner: {winner_name}")
        print(f"Total moves: {move_count}")
        print("=" * 50)
    
    return game.winner, move_count


def play_multiple_games(agent1, agent2, game_class, rules, num_games=10, alternate_colors=True, 
                        display=False, temperature=0.0, time_per_move=None, num_workers=1,
                        checkpoint1_path=None, checkpoint2_path=None, game_name=None, simulations=None):
    """
    Play multiple games and collect statistics.
    
    Args:
        agent1: First agent
        agent2: Second agent
        game_class: Game class to use
        rules: Dictionary of game rules
        num_games: Number of games to play
        alternate_colors: If True, agents alternate colors each game. 
                         If False, each pairing is played twice (once per color)
        display: Whether to print board state for each game
        temperature: Temperature for move selection (0=deterministic)
        time_per_move: Optional time limit per move in seconds
        num_workers: Number of parallel workers (if > 1, uses multiprocessing)
        checkpoint1_path: Path to checkpoint1 (required for multiprocessing)
        checkpoint2_path: Path to checkpoint2 (required for multiprocessing)
        game_name: Name of game variant (required for multiprocessing)
        simulations: Number of MCTS simulations (required for multiprocessing)
    """
    # Track wins by role and agent
    agent1_attacker_wins = 0
    agent1_defender_wins = 0
    agent2_attacker_wins = 0
    agent2_defender_wins = 0
    draws = 0
    total_moves = 0
    
    # Track games played in each role
    agent1_attacker_games = 0
    agent1_defender_games = 0
    agent2_attacker_games = 0
    agent2_defender_games = 0
    
    if alternate_colors:
        print(f"\nPlaying {num_games} games (alternating colors)...\n")
        games_to_play = num_games
    else:
        print(f"\nPlaying {num_games * 2} games (each pairing played twice with colors swapped)...\n")
        games_to_play = num_games * 2
    
    # Use multiprocessing if num_workers > 1
    if num_workers > 1:
        if checkpoint1_path is None or checkpoint2_path is None or game_name is None or simulations is None:
            raise ValueError("For multiprocessing, checkpoint paths, game_name, and simulations must be provided")
        
        print(f"Using {num_workers} parallel workers...")

        
        # Extract rule parameters (only the ones that exist in game classes)
        king_capture_pieces = rules.get('king_capture_pieces', 2)
        king_can_capture = rules.get('king_can_capture', True)
        throne_is_hostile = rules.get('throne_is_hostile', False)
        throne_enabled = rules.get('throne_enabled', True)
        
        # Create list of game configurations
        game_configs = []
        for i in range(games_to_play):
            if alternate_colors:
                agent1_plays_attacker = (i % 2 == 0)
            else:
                agent1_plays_attacker = (i % 2 == 0)
            
            game_configs.append((
                i, checkpoint1_path, checkpoint2_path, agent1_plays_attacker,
                game_name, simulations, king_capture_pieces, king_can_capture,
                throne_is_hostile, throne_enabled, time_per_move, temperature
            ))
        
        # Play games in parallel using imap_unordered for better progress tracking
        # and to avoid spawning all workers at once
        with mp.Pool(processes=num_workers) as pool:
            # Use imap_unordered to process results as they complete
            results = list(pool.starmap(play_single_game_worker, game_configs, chunksize=1))
        
        # Process results
        for i, (agent1_won, winner_role, moves) in enumerate(results):
            if alternate_colors:
                agent1_plays_attacker = (i % 2 == 0)
            else:
                agent1_plays_attacker = (i % 2 == 0)
            
            # Update game counts
            if agent1_plays_attacker:
                agent1_attacker_games += 1
                agent2_defender_games += 1
            else:
                agent2_attacker_games += 1
                agent1_defender_games += 1
            
            # Update win counts
            if agent1_won is True:
                if agent1_plays_attacker:
                    agent1_attacker_wins += 1
                else:
                    agent1_defender_wins += 1
            elif agent1_won is False:
                if agent1_plays_attacker:
                    agent2_defender_wins += 1
                else:
                    agent2_attacker_wins += 1
            else:
                draws += 1
            
            total_moves += moves
            
            if (i + 1) % 10 == 0 or games_to_play <= 10:
                print(f"Completed {i + 1}/{games_to_play} games...")
    else:
        # Sequential game playing (original implementation)
        for i in range(games_to_play):
            # Determine which agent plays which color
            if alternate_colors:
                # Alternate who plays attacker/defender
                if i % 2 == 0:
                    # Agent1 as attacker, Agent2 as defender
                    agent1_attacker_games += 1
                    agent2_defender_games += 1
                    winner, moves = play_game_between_agents(agent1, agent2, game_class, rules, 
                                                             display=display, temperature=temperature, 
                                                             time_per_move=time_per_move)
                    if winner == 0:
                        agent1_attacker_wins += 1
                    elif winner == 1:
                        agent2_defender_wins += 1
                else:
                    # Agent2 as attacker, Agent1 as defender
                    agent2_attacker_games += 1
                    agent1_defender_games += 1
                    winner, moves = play_game_between_agents(agent2, agent1, game_class, rules, 
                                                             display=display, temperature=temperature,
                                                             time_per_move=time_per_move)
                    if winner == 0:
                        agent2_attacker_wins += 1
                    elif winner == 1:
                        agent1_defender_wins += 1
            else:
                # Play each pairing twice: once with agent1 as attacker, once with agent2 as attacker
                game_in_pair = i % 2
                if game_in_pair == 0:
                    # Agent1 as attacker, Agent2 as defender
                    agent1_attacker_games += 1
                    agent2_defender_games += 1
                    winner, moves = play_game_between_agents(agent1, agent2, game_class, rules, 
                                                             display=display, temperature=temperature,
                                                             time_per_move=time_per_move)
                    if winner == 0:
                        agent1_attacker_wins += 1
                    elif winner == 1:
                        agent2_defender_wins += 1
                else:
                    # Agent2 as attacker, Agent1 as defender
                    agent2_attacker_games += 1
                    agent1_defender_games += 1
                    winner, moves = play_game_between_agents(agent2, agent1, game_class, rules, 
                                                             display=display, temperature=temperature,
                                                             time_per_move=time_per_move)
                    if winner == 0:
                        agent2_attacker_wins += 1
                    elif winner == 1:
                        agent1_defender_wins += 1
            
            if winner is None:
                draws += 1
            
            total_moves += moves
            
            if (i + 1) % 10 == 0 or games_to_play <= 10:
                print(f"Completed {i + 1}/{games_to_play} games...")
    
    # Calculate overall statistics
    agent1_wins = agent1_attacker_wins + agent1_defender_wins
    agent2_wins = agent2_attacker_wins + agent2_defender_wins
    attacker_wins = agent1_attacker_wins + agent2_attacker_wins
    defender_wins = agent1_defender_wins + agent2_defender_wins
    
    print("\n" + "=" * 50)
    print("Statistics:")
    print("=" * 50)
    print(f"Total games: {games_to_play}")
    print(f"\nOverall Results:")
    print(f"  Agent 1 wins: {agent1_wins} ({100 * agent1_wins / games_to_play:.1f}%)")
    print(f"  Agent 2 wins: {agent2_wins} ({100 * agent2_wins / games_to_play:.1f}%)")
    print(f"  Draws: {draws} ({100 * draws / games_to_play:.1f}%)")
    
    print(f"\nBy Role:")
    print(f"  Attacker wins: {attacker_wins}/{games_to_play} ({100 * attacker_wins / games_to_play:.1f}%)")
    print(f"  Defender wins: {defender_wins}/{games_to_play} ({100 * defender_wins / games_to_play:.1f}%)")
    
    print(f"\nAgent 1 Performance:")
    if agent1_attacker_games > 0:
        print(f"  As Attacker: {agent1_attacker_wins}/{agent1_attacker_games} ({100 * agent1_attacker_wins / agent1_attacker_games:.1f}%)")
    if agent1_defender_games > 0:
        print(f"  As Defender: {agent1_defender_wins}/{agent1_defender_games} ({100 * agent1_defender_wins / agent1_defender_games:.1f}%)")
    
    print(f"\nAgent 2 Performance:")
    if agent2_attacker_games > 0:
        print(f"  As Attacker: {agent2_attacker_wins}/{agent2_attacker_games} ({100 * agent2_attacker_wins / agent2_attacker_games:.1f}%)")
    if agent2_defender_games > 0:
        print(f"  As Defender: {agent2_defender_wins}/{agent2_defender_games} ({100 * agent2_defender_wins / agent2_defender_games:.1f}%)")
    
    print(f"\nAverage game length: {total_moves / games_to_play:.1f} moves")
    
    # Calculate ELO difference using expected score formula
    # E(A) = 1 / (1 + 10^((R_B - R_A)/400))
    # If agent1 scored S out of N games, then S/N = E(A)
    # Solving for (R_A - R_B): ELO_diff = 400 * log10(S / (N - S))
    
    agent1_score = agent1_wins + 0.5 * draws  # Count draws as 0.5
    agent1_games = games_to_play
    
    if agent1_score == 0 or agent1_score == agent1_games:
        # Extreme cases - use approximation
        if agent1_score == 0:
            elo_diff = -800  # Effectively lost all games
            print(f"\nELO Difference: {elo_diff:.1f} (Agent 1 vs Agent 2)")
            print(f"  Agent 1 is approximately {abs(elo_diff):.0f} ELO points weaker")
        else:
            elo_diff = 800  # Effectively won all games
            print(f"\nELO Difference: {elo_diff:.1f} (Agent 1 vs Agent 2)")
            print(f"  Agent 1 is approximately {elo_diff:.0f} ELO points stronger")
    else:
        # Normal case
        expected_score = agent1_score / agent1_games
        elo_diff = 400 * np.log10(expected_score / (1 - expected_score))
        
        print(f"\nELO Difference: {elo_diff:.1f} (Agent 1 vs Agent 2)")
        if abs(elo_diff) < 5:
            print(f"  Agents are approximately equal in strength")
        elif elo_diff > 0:
            print(f"  Agent 1 is approximately {elo_diff:.0f} ELO points stronger")
        else:
            print(f"  Agent 1 is approximately {abs(elo_diff):.0f} ELO points weaker")
    
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Play Tafl games between two AI agents")
    parser.add_argument("checkpoint1", type=str,
                       help="Path to first checkpoint (.pth file) - plays as attacker")
    parser.add_argument("checkpoint2", type=str,
                       help="Path to second checkpoint (.pth file) - plays as defender")
    parser.add_argument("--game", type=str, default="brandubh", choices=["brandubh", "tablut", "hnefatafl"],
                       help="Game variant: brandubh (7x7), tablut (9x9), or hnefatafl (11x11) (default: brandubh)")
    parser.add_argument("--simulations", type=int, default=100,
                       help="Number of MCTS simulations per move (default: 100)")
    parser.add_argument("--c-puct", type=float, default=1.4,
                       help="MCTS exploration constant (default: 1.4)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for move selection (0=deterministic, higher=more random) (default: 0.0)")
    parser.add_argument("--dirichlet-noise", action="store_true",
                       help="Add Dirichlet noise to root node for variety (used during training)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
                       help="Dirichlet alpha parameter (default: 0.3)")
    parser.add_argument("--num-games", type=int, default=1,
                       help="Number of games to play (default: 1).")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of parallel workers for playing multiple games (default: 1, sequential)")
    parser.add_argument("--time-per-move", type=float, default=None,
                       help="Time limit per move in seconds (overrides --simulations if specified)")
    parser.add_argument("--display", action="store_true",
                       help="Display board state during play (default: shown for single game, hidden for multiple games)")
    parser.add_argument("--swap-colors", action="store_true",
                       help="Play each game twice with colors swapped. With --num-games N, "
                            "plays 2N games total (N pairings × 2 colors). "
                            "Default behavior alternates colors between games.")
    
    # Game rule arguments (optional - will use checkpoint config by default)
    parser.add_argument("--king-capture-pieces", type=int, default=None, choices=[2, 3, 4],
                       help="Number of pieces required to capture king: 2, 3, or 4. "
                            "If not specified, uses rules from checkpoint.")
    parser.add_argument("--king-can-capture", action="store_true", dest="king_can_capture_flag", default=None,
                       help="King CAN help capture enemy pieces (overrides checkpoint)")
    parser.add_argument("--king-cannot-capture", action="store_false", dest="king_can_capture_flag",
                       help="King CANNOT help capture enemy pieces (overrides checkpoint)")
    parser.add_argument("--throne-is-hostile", action="store_true", dest="throne_is_hostile_flag", default=None,
                       help="Throne IS hostile for captures (overrides checkpoint)")
    parser.add_argument("--throne-not-hostile", action="store_false", dest="throne_is_hostile_flag",
                       help="Throne is NOT hostile for captures (overrides checkpoint)")
    parser.add_argument("--throne-enabled", action="store_true", dest="throne_enabled_flag", default=None,
                       help="Throne IS enabled - blocks movement (overrides checkpoint)")
    parser.add_argument("--throne-disabled", action="store_false", dest="throne_enabled_flag",
                       help="Throne is DISABLED - center is normal square (overrides checkpoint)")
    parser.add_argument("--force-rules", action="store_true",
                       help="Use command-line rules for any unspecified options (instead of checkpoint defaults)")
    
    args = parser.parse_args()
    
    # Build forced rules dict if specified
    force_rules = None
    if args.force_rules or any([
        args.king_capture_pieces is not None,
        args.king_can_capture_flag is not None,
        args.throne_is_hostile_flag is not None,
        args.throne_enabled_flag is not None
    ]):
        # Start with defaults when forcing rules
        if args.force_rules:
            force_rules = {
                'king_capture_pieces': 2,
                'king_can_capture': True,
                'throne_is_hostile': False,
                'throne_enabled': True,
            }
        else:
            force_rules = {}
        
        # Override with any explicitly specified values
        if args.king_capture_pieces is not None:
            force_rules['king_capture_pieces'] = args.king_capture_pieces
        if args.king_can_capture_flag is not None:
            force_rules['king_can_capture'] = args.king_can_capture_flag
        if args.throne_is_hostile_flag is not None:
            force_rules['throne_is_hostile'] = args.throne_is_hostile_flag
        if args.throne_enabled_flag is not None:
            force_rules['throne_enabled'] = args.throne_enabled_flag
    
    # Select game class first (needed for network loading)
    if args.game.lower() == 'tablut':
        game_class = Tablut
    elif args.game.lower() == 'hnefatafl':
        game_class = Hnefatafl
    else:
        game_class = Brandubh
    
    # Load checkpoints with correct network type
    print("Loading checkpoints...\n")
    network1, rules1 = load_checkpoint_with_rules(args.checkpoint1, force_rules, game_class)
    network2, rules2 = load_checkpoint_with_rules(args.checkpoint2, force_rules, game_class)
    
    # Use rules from first checkpoint (or forced rules if specified)
    rules = rules1
    
    if not force_rules and rules1 != rules2:
        print("\nWarning: Checkpoints have different game rules!")
        print(f"Checkpoint 1 rules: {rules1}")
        print(f"Checkpoint 2 rules: {rules2}")
        print("Using rules from checkpoint 1. Use --force-rules to override.")
    
    print(f"\nUsing rules: {rules}\n")
    
    # Determine move encoder class based on game type
    if game_class == Tablut:
        from network_tablut import TablutMoveEncoder
        move_encoder_class = TablutMoveEncoder
    elif game_class == Hnefatafl:
        from network_hnefatafl import HnefataflMoveEncoder
        move_encoder_class = HnefataflMoveEncoder
    else:
        from network import MoveEncoder
        move_encoder_class = MoveEncoder
    
    # Create agents with optional Dirichlet noise
    agent1 = Agent(network1, num_simulations=args.simulations, c_puct=args.c_puct, device='cpu',
                   add_dirichlet_noise=args.dirichlet_noise, dirichlet_alpha=args.dirichlet_alpha,
                   move_encoder_class=move_encoder_class)
    agent2 = Agent(network2, num_simulations=args.simulations, c_puct=args.c_puct, device='cpu',
                   add_dirichlet_noise=args.dirichlet_noise, dirichlet_alpha=args.dirichlet_alpha,
                   move_encoder_class=move_encoder_class)
    
    # Play games
    if args.num_games == 1 and not args.swap_colors:
        # Single game - display by default (unless --display explicitly set to override)
        display = args.display if args.display else True
        play_game_between_agents(agent1, agent2, game_class, rules, display=display, 
                                temperature=args.temperature, time_per_move=args.time_per_move)
    else:
        # Multiple games - hide display by default (unless --display explicitly set)
        play_multiple_games(agent1, agent2, game_class, rules, 
                          num_games=args.num_games, 
                          alternate_colors=not args.swap_colors,
                          display=args.display,
                          temperature=args.temperature,
                          time_per_move=args.time_per_move,
                          num_workers=args.num_workers,
                          checkpoint1_path=args.checkpoint1,
                          checkpoint2_path=args.checkpoint2,
                          game_name=args.game,
                          simulations=args.simulations)


if __name__ == "__main__":
    main()

