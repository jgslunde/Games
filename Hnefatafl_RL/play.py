"""
Play Tafl games between two AI agents loaded from checkpoints.
Displays the game state and move history.

Supports both Brandubh (7x7) and Tablut (9x9) variants.

Usage:
    python play.py <checkpoint1> <checkpoint2> [options]
    
Example:
    python play.py checkpoints/best_model.pth checkpoints/checkpoint_iter_5.pth
    python play.py model1.pth model2.pth --game tablut --simulations 200
    python play.py model1.pth model2.pth --king-capture-pieces 4 --throne-is-hostile
    python play.py model1.pth model2.pth --num-games 10 --swap-colors  # 20 games total
"""

import argparse
import sys
import torch

from brandubh import Brandubh
from tablut import Tablut
from network import BrandubhNet
from agent import Agent


def load_checkpoint_with_rules(checkpoint_path: str, force_rules: dict = None):
    """
    Load neural network from checkpoint and extract rules.
    
    Args:
        checkpoint_path: Path to checkpoint file
        force_rules: Optional dict of rules to force (overrides checkpoint)
    
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
            rules = force_rules if force_rules else {
                'king_capture_pieces': 2,
                'king_can_capture': True,
                'throne_is_hostile': False,
                'throne_enabled': True,
            }
        
        # Create and load network
        network = BrandubhNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()
        
        print(f"Loaded network from {checkpoint_path}")
        print(f"  Architecture: {num_res_blocks} residual blocks, {num_channels} channels")
        
        return network, rules
        
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        sys.exit(1)


def play_game_between_agents(agent1, agent2, game_class, rules, display=True, temperature=0.0):
    """
    Play a game between two AI agents.
    
    Args:
        agent1: Agent playing as attacker
        agent2: Agent playing as defender
        game_class: Game class (Brandubh or Tablut)
        rules: Dictionary of game rules
        display: Whether to print the board state
        temperature: Temperature for move selection (0=deterministic)
    
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
        print("\nInitial board:")
        print(game)
        print("\n")
    
    while not game.game_over:
        # Get current agent
        current_agent = agent1 if game.current_player == 0 else agent2
        
        # Get move from agent
        move = current_agent.select_move(game, temperature=temperature)
        
        if move is None:
            # No legal moves - opponent wins
            game.game_over = True
            game.winner = 1 - game.current_player
            break
        
        from_r, from_c, to_r, to_c = move
        
        if display:
            player_name = "Attackers" if game.current_player == 0 else "Defenders"
            print(f"Move {move_count + 1}: {player_name} move ({from_r},{from_c}) → ({to_r},{to_c})")
        
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


def play_multiple_games(agent1, agent2, game_class, rules, num_games=10, alternate_colors=True, display=False, temperature=0.0):
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
    
    for i in range(games_to_play):
        # Determine which agent plays which color
        if alternate_colors:
            # Alternate who plays attacker/defender
            if i % 2 == 0:
                # Agent1 as attacker, Agent2 as defender
                agent1_attacker_games += 1
                agent2_defender_games += 1
                winner, moves = play_game_between_agents(agent1, agent2, game_class, rules, display=display, temperature=temperature)
                if winner == 0:
                    agent1_attacker_wins += 1
                elif winner == 1:
                    agent2_defender_wins += 1
            else:
                # Agent2 as attacker, Agent1 as defender
                agent2_attacker_games += 1
                agent1_defender_games += 1
                winner, moves = play_game_between_agents(agent2, agent1, game_class, rules, display=display, temperature=temperature)
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
                winner, moves = play_game_between_agents(agent1, agent2, game_class, rules, display=display, temperature=temperature)
                if winner == 0:
                    agent1_attacker_wins += 1
                elif winner == 1:
                    agent2_defender_wins += 1
            else:
                # Agent2 as attacker, Agent1 as defender
                agent2_attacker_games += 1
                agent1_defender_games += 1
                winner, moves = play_game_between_agents(agent2, agent1, game_class, rules, display=display, temperature=temperature)
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
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Play Tafl games between two AI agents")
    parser.add_argument("checkpoint1", type=str,
                       help="Path to first checkpoint (.pth file) - plays as attacker")
    parser.add_argument("checkpoint2", type=str,
                       help="Path to second checkpoint (.pth file) - plays as defender")
    parser.add_argument("--game", type=str, default="brandubh", choices=["brandubh", "tablut"],
                       help="Game variant: brandubh (7x7) or tablut (9x9) (default: brandubh)")
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
    
    # Load checkpoints
    print("Loading checkpoints...\n")
    network1, rules1 = load_checkpoint_with_rules(args.checkpoint1, force_rules)
    network2, rules2 = load_checkpoint_with_rules(args.checkpoint2, force_rules)
    
    # Use rules from first checkpoint (or forced rules if specified)
    rules = rules1
    
    if not force_rules and rules1 != rules2:
        print("\nWarning: Checkpoints have different game rules!")
        print(f"Checkpoint 1 rules: {rules1}")
        print(f"Checkpoint 2 rules: {rules2}")
        print("Using rules from checkpoint 1. Use --force-rules to override.")
    
    print(f"\nUsing rules: {rules}\n")
    
    # Create agents with optional Dirichlet noise
    agent1 = Agent(network1, num_simulations=args.simulations, c_puct=args.c_puct, device='cpu',
                   add_dirichlet_noise=args.dirichlet_noise, dirichlet_alpha=args.dirichlet_alpha)
    agent2 = Agent(network2, num_simulations=args.simulations, c_puct=args.c_puct, device='cpu',
                   add_dirichlet_noise=args.dirichlet_noise, dirichlet_alpha=args.dirichlet_alpha)
    
    # Select game class
    game_class = Tablut if args.game.lower() == 'tablut' else Brandubh
    
    # Play games
    if args.num_games == 1 and not args.swap_colors:
        # Single game - display by default (unless --display explicitly set to override)
        display = args.display if args.display else True
        play_game_between_agents(agent1, agent2, game_class, rules, display=display, temperature=args.temperature)
    else:
        # Multiple games - hide display by default (unless --display explicitly set)
        play_multiple_games(agent1, agent2, game_class, rules, 
                          num_games=args.num_games, 
                          alternate_colors=not args.swap_colors,
                          display=args.display,
                          temperature=args.temperature)


if __name__ == "__main__":
    main()

