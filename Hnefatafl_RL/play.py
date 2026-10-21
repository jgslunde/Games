"""
Play Tafl games between two AI agents loaded from checkpoints.
Displays the game state and move history.

The games are played under the rule set the first checkpoint was trained with (--ruleset names
another set of tafl.RULESETS on the same board).

Usage:
    python play.py <checkpoint1> <checkpoint2> [options]

Example:
    python play.py checkpoints/brandubh/baseline/best_model.pth checkpoints/brandubh/baseline/checkpoint_iter_5.pth
    python play.py model1.pth model2.pth --simulations 200
    python play.py model1.pth model2.pth --ruleset fetlar
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
import setproctitle

from run_training import PROCESS_TITLE
from agent import Agent
from tafl import Tafl, get_rules
from tafl_network import load_network, describe_network


def play_single_game_worker(game_idx, checkpoint1_path, checkpoint2_path, agent1_plays_attacker,
                              rules, simulations, time_per_move, temperature,
                              c_puct1, c_puct2, fpu_reduction1, fpu_reduction2, temperature_threshold,
                              max_moves, dirichlet_noise, dirichlet_alpha):
    """
    Worker function for playing a single game in a separate process.
    Each agent searches with the search_discount its network was trained with (from its checkpoint).
    The game is a draw once more than max_moves plies have been played.
    Returns tuple: (agent1_won, winner_role, move_count)
    """
    try:
        # Set torch to use only 1 thread per worker to avoid conflicts
        # This MUST be done before creating any torch objects
        torch.set_num_threads(1)
        # Fresh seed for every game: forked workers share the parent's random state once the parent
        # has drawn a random number, and would then play identical games
        np.random.seed()

        game_state = Tafl(rules)

        # Load the networks; their architecture and input planes come from the checkpoints, so two
        # different architectures can play each other
        device = torch.device('cpu')
        network1, checkpoint1 = load_network(checkpoint1_path)
        network2, checkpoint2 = load_network(checkpoint2_path)

        # Create agents with separate C_PUCT and FPU values (the input planes follow the network).
        # Each agent searches with the discount its network was trained with.
        agent1 = Agent(network1, num_simulations=simulations, device=device,
                      c_puct=c_puct1, fpu_reduction=fpu_reduction1,
                      add_dirichlet_noise=dirichlet_noise, dirichlet_alpha=dirichlet_alpha,
                      search_discount=checkpoint1['config']['search_discount'])
        agent2 = Agent(network2, num_simulations=simulations, device=device,
                      c_puct=c_puct2, fpu_reduction=fpu_reduction2,
                      add_dirichlet_noise=dirichlet_noise, dirichlet_alpha=dirichlet_alpha,
                      search_discount=checkpoint2['config']['search_discount'])

        # Play game
        move_count = 0

        # Debug: Check initial state
        if game_state.game_over:
            # Game is already over before we start - this shouldn't happen
            return (None, 'error', 0)
        
        while not game_state.game_over and move_count <= max_moves:
            current_player_is_attacker = (game_state.current_player == 0)
            
            # Determine which agent makes the move
            if current_player_is_attacker == agent1_plays_attacker:
                current_agent = agent1
            else:
                current_agent = agent2
            
            # Apply temperature threshold (drop to 0 after threshold)
            current_temp = temperature if move_count < temperature_threshold else 0.0
            
            # Select move (with a time limit per move if specified)
            move, value, visit_prob = current_agent.select_move_with_stats(game_state, temperature=current_temp,
                                                                           time_limit=time_per_move)

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


def load_checkpoint_with_rules(checkpoint_path: str):
    """
    Load a network from a training checkpoint, with the rules and settings it was trained with.

    Returns:
        (network, rules, search_discount, max_game_length): rules is a tafl.Rules, search_discount the
        MCTS discount and max_game_length the draw limit of the training run
    """
    try:
        network, checkpoint = load_network(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        sys.exit(1)
    config = checkpoint['config']
    rules = get_rules(config['rules'])
    print(f"Loaded network from {checkpoint_path}")
    print(f"  Architecture: {describe_network(network)}")
    print(f"  Rule set: {rules.name}; search discount: {config['search_discount']}")
    return network, rules, config['search_discount'], config['max_game_length']


def play_game_between_agents(agent1, agent2, rules, display=True, temperature=0.0, time_per_move=None,
                             temperature_threshold=999, max_moves=100):
    """
    Play a game between two AI agents.

    Args:
        agent1: Agent playing as attacker
        agent2: Agent playing as defender
        rules: the rule set (tafl.Rules)
        display: Whether to print the board state
        temperature: Temperature for move selection (0=deterministic)
        time_per_move: Optional time limit per move in seconds (if None, uses simulations)
        temperature_threshold: Move number after which temperature drops to 0 (default: 999)
        max_moves: The game is a draw once more than this many plies have been played
    
    Returns:
        winner: 0 for Attackers, 1 for Defenders, None for draw
        num_moves: Number of moves in the game
    """
    game = Tafl(rules)
    move_count = 0

    if display:
        print("=" * 50)
        print("Starting new game")
        print("=" * 50)
        print(rules.describe())
        if time_per_move is not None:
            print(f"Time per move: {time_per_move} seconds")
        print("\nInitial board:")
        print(game)
        print("\n")
    
    while not game.game_over:
        # Get current agent
        current_agent = agent1 if game.current_player == 0 else agent2
        
        # Apply temperature threshold (drop to 0 after threshold)
        current_temp = temperature if move_count < temperature_threshold else 0.0
        
        # Get move from agent with statistics (using time limit if specified)
        move, value, visit_prob = current_agent.select_move_with_stats(game, temperature=current_temp,
                                                                       time_limit=time_per_move)
        
        if move is None:
            # No legal moves - opponent wins
            blocked = ("attackers_blocked", "defenders_blocked")[game.current_player]
            game._end(1 - game.current_player, blocked)
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
        
        # Move limit (a win on the last allowed ply still counts as a win)
        if move_count > max_moves and not game.game_over:
            if display:
                print(f"Game exceeded {max_moves} moves - draw by move limit")
            game._end(None, "move_limit")
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


def play_multiple_games(agent1, agent2, rules, num_games=10, alternate_colors=True,
                        display=False, temperature=0.0, time_per_move=None, num_workers=1,
                        checkpoint1_path=None, checkpoint2_path=None, simulations=None,
                        c_puct1=1.4, c_puct2=1.4, fpu_reduction1=-0.5, fpu_reduction2=-0.5, temperature_threshold=999,
                        max_moves=100, dirichlet_noise=False, dirichlet_alpha=0.3):
    """
    Play multiple games and collect statistics.
    
    Args:
        agent1: First agent
        agent2: Second agent
        rules: the rule set (tafl.Rules)
        num_games: Number of games to play
        alternate_colors: If True, agents alternate colors each game. 
                         If False, each pairing is played twice (once per color)
        display: Whether to print board state for each game
        temperature: Temperature for move selection (0=deterministic)
        time_per_move: Optional time limit per move in seconds
        num_workers: Number of parallel workers (if > 1, uses multiprocessing)
        checkpoint1_path: Path to checkpoint1 (required for multiprocessing)
        checkpoint2_path: Path to checkpoint2 (required for multiprocessing)
        simulations: Number of MCTS simulations (required for multiprocessing)
        max_moves: The game is a draw once more than this many plies have been played
        dirichlet_noise, dirichlet_alpha: Root noise for the agents built in worker processes
                                          (the agents passed in already carry their own settings)
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
        if checkpoint1_path is None or checkpoint2_path is None or simulations is None:
            raise ValueError("For multiprocessing, checkpoint paths and simulations must be provided")

        print(f"Using {num_workers} parallel workers...")

        # Create list of game configurations
        game_configs = []
        for i in range(games_to_play):
            if alternate_colors:
                agent1_plays_attacker = (i % 2 == 0)
            else:
                agent1_plays_attacker = (i % 2 == 0)
            
            game_configs.append((
                i, checkpoint1_path, checkpoint2_path, agent1_plays_attacker,
                rules, simulations, time_per_move, temperature,
                c_puct1, c_puct2, fpu_reduction1, fpu_reduction2, temperature_threshold,
                max_moves, dirichlet_noise, dirichlet_alpha
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
            
            # Print progress every num_workers games (or at least every 10 games if num_workers is small)
            print_interval = max(num_workers, 10) if games_to_play > 20 else 1
            if (i + 1) % print_interval == 0 or (i + 1) == games_to_play:
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
                    winner, moves = play_game_between_agents(agent1, agent2, rules,
                                                             display=display, temperature=temperature,
                                                             time_per_move=time_per_move, temperature_threshold=temperature_threshold,
                                                             max_moves=max_moves)
                    if winner == 0:
                        agent1_attacker_wins += 1
                    elif winner == 1:
                        agent2_defender_wins += 1
                else:
                    # Agent2 as attacker, Agent1 as defender
                    agent2_attacker_games += 1
                    agent1_defender_games += 1
                    winner, moves = play_game_between_agents(agent2, agent1, rules,
                                                             display=display, temperature=temperature,
                                                             time_per_move=time_per_move, temperature_threshold=temperature_threshold,
                                                             max_moves=max_moves)
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
                    winner, moves = play_game_between_agents(agent1, agent2, rules,
                                                             display=display, temperature=temperature,
                                                             time_per_move=time_per_move, temperature_threshold=temperature_threshold,
                                                             max_moves=max_moves)
                    if winner == 0:
                        agent1_attacker_wins += 1
                    elif winner == 1:
                        agent2_defender_wins += 1
                else:
                    # Agent2 as attacker, Agent1 as defender
                    agent2_attacker_games += 1
                    agent1_defender_games += 1
                    winner, moves = play_game_between_agents(agent2, agent1, rules,
                                                             display=display, temperature=temperature,
                                                             time_per_move=time_per_move, temperature_threshold=temperature_threshold,
                                                             max_moves=max_moves)
                    if winner == 0:
                        agent2_attacker_wins += 1
                    elif winner == 1:
                        agent1_defender_wins += 1
            
            if winner is None:
                draws += 1
            
            total_moves += moves
            
            # Adaptive progress printing
            if games_to_play <= 10:
                print_interval = 1
            elif games_to_play <= 50:
                print_interval = 10
            else:
                print_interval = max(10, games_to_play // 10)
            
            if (i + 1) % print_interval == 0 or (i + 1) == games_to_play:
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
    parser.add_argument("--ruleset", type=str, default=None,
                       help="Play under this rule set of tafl.RULESETS (default: that of checkpoint 1)")
    parser.add_argument("--simulations", type=int, default=100,
                       help="Number of MCTS simulations per move (default: 100)")
    parser.add_argument("--c-puct", type=float, default=1.4,
                       help="MCTS exploration constant for both agents (default: 1.4)")
    parser.add_argument("--c-puct1", type=float, default=None,
                       help="MCTS exploration constant for agent 1 (overrides --c-puct)")
    parser.add_argument("--c-puct2", type=float, default=None,
                       help="MCTS exploration constant for agent 2 (overrides --c-puct)")
    parser.add_argument("--fpu-reduction", type=float, default=-0.5,
                       help="First Play Urgency reduction for both agents (default: -0.5)")
    parser.add_argument("--fpu-reduction1", type=float, default=None,
                       help="First Play Urgency reduction for agent 1 (overrides --fpu-reduction)")
    parser.add_argument("--fpu-reduction2", type=float, default=None,
                       help="First Play Urgency reduction for agent 2 (overrides --fpu-reduction)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for move selection (0=deterministic, higher=more random) (default: 0.0)")
    parser.add_argument("--temperature-threshold", type=int, default=999,
                       help="Move number (ply) after which temperature drops to 0 (default: 999, effectively disabled)")
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
    parser.add_argument("--max-moves", type=int, default=None,
                       help="A game is a draw once more than this many plies have been played "
                            "(default: the draw limit of checkpoint 1's training run)")
    parser.add_argument("--display", action="store_true",
                       help="Display board state during play (default: shown for single game, hidden for multiple games)")
    parser.add_argument("--swap-colors", action="store_true",
                       help="Play each game twice with colors swapped. With --num-games N, "
                            "plays 2N games total (N pairings × 2 colors). "
                            "Default behavior alternates colors between games.")

    args = parser.parse_args()
    setproctitle.setproctitle(PROCESS_TITLE)  # the name of this process and its workers in top/ps

    print("Loading checkpoints...\n")
    network1, rules1, search_discount1, max_game_length = load_checkpoint_with_rules(args.checkpoint1)
    network2, rules2, search_discount2, _ = load_checkpoint_with_rules(args.checkpoint2)
    if rules1 != rules2 and args.ruleset is None:
        print(f"\nWarning: the checkpoints were trained under different rules ({rules1.name} and "
              f"{rules2.name}); playing under those of checkpoint 1. Use --ruleset to choose.")
    rules = get_rules(args.ruleset) if args.ruleset is not None else rules1
    if rules.board_size != network1.board_size or rules.board_size != network2.board_size:
        sys.exit(f"Rule set {rules.name} is played on a different board than the networks")
    if args.max_moves is None:
        args.max_moves = max_game_length
    print(f"\n{rules.describe()}")

    # Determine C_PUCT and FPU values for each agent
    c_puct1 = args.c_puct1 if args.c_puct1 is not None else args.c_puct
    c_puct2 = args.c_puct2 if args.c_puct2 is not None else args.c_puct
    fpu_reduction1 = args.fpu_reduction1 if args.fpu_reduction1 is not None else args.fpu_reduction
    fpu_reduction2 = args.fpu_reduction2 if args.fpu_reduction2 is not None else args.fpu_reduction
    
    # Print comprehensive game parameters
    print("\n" + "=" * 50)
    print("Game Parameters:")
    print("=" * 50)
    print(f"Rule set: {rules.name}; draw after {args.max_moves} plies")
    print(f"Number of simulations: {args.simulations}")
    if args.time_per_move is not None:
        print(f"Time per move: {args.time_per_move}s (overrides simulations)")
    print(f"Temperature: {args.temperature}")
    if args.temperature > 0:
        print(f"Temperature threshold: {args.temperature_threshold} moves")
    if args.dirichlet_noise:
        print(f"Dirichlet noise: enabled (alpha={args.dirichlet_alpha})")
    
    print(f"\nAgent 1 MCTS: C_PUCT={c_puct1}, FPU_reduction={fpu_reduction1}")
    print(f"Agent 2 MCTS: C_PUCT={c_puct2}, FPU_reduction={fpu_reduction2}")
    
    if args.num_games > 1:
        print(f"\nNumber of games: {args.num_games}")
        if args.swap_colors:
            print(f"Mode: Swap colors (total {args.num_games * 2} games)")
        else:
            print(f"Mode: Alternate colors")
        if args.num_workers > 1:
            print(f"Parallel workers: {args.num_workers}")
    print("=" * 50 + "\n")
    
    # Create agents with optional Dirichlet noise and separate C_PUCT/FPU values. Each agent searches
    # with the discount its network was trained with.
    agent1 = Agent(network1, num_simulations=args.simulations, c_puct=c_puct1, device='cpu',
                   add_dirichlet_noise=args.dirichlet_noise, dirichlet_alpha=args.dirichlet_alpha,
                   fpu_reduction=fpu_reduction1, search_discount=search_discount1)
    agent2 = Agent(network2, num_simulations=args.simulations, c_puct=c_puct2, device='cpu',
                   add_dirichlet_noise=args.dirichlet_noise, dirichlet_alpha=args.dirichlet_alpha,
                   fpu_reduction=fpu_reduction2, search_discount=search_discount2)

    # Play games
    if args.num_games == 1 and not args.swap_colors:
        # Single game - display by default (unless --display explicitly set to override)
        display = args.display if args.display else True
        play_game_between_agents(agent1, agent2, rules, display=display,
                                temperature=args.temperature, time_per_move=args.time_per_move,
                                temperature_threshold=args.temperature_threshold, max_moves=args.max_moves)
    else:
        # Multiple games - hide display by default (unless --display explicitly set)
        play_multiple_games(agent1, agent2, rules,
                          num_games=args.num_games, 
                          alternate_colors=not args.swap_colors,
                          display=args.display,
                          temperature=args.temperature,
                          time_per_move=args.time_per_move,
                          num_workers=args.num_workers,
                          checkpoint1_path=args.checkpoint1,
                          checkpoint2_path=args.checkpoint2,
                          simulations=args.simulations,
                          c_puct1=c_puct1,
                          c_puct2=c_puct2,
                          fpu_reduction1=fpu_reduction1,
                          fpu_reduction2=fpu_reduction2,
                          temperature_threshold=args.temperature_threshold,
                          max_moves=args.max_moves,
                          dirichlet_noise=args.dirichlet_noise,
                          dirichlet_alpha=args.dirichlet_alpha)


if __name__ == "__main__":
    main()

