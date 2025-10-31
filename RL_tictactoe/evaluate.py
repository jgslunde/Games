import numpy as np
import multiprocessing as mp
from TicTacToe import TicTacToeGame
from MCTS import MCTS


class RandomPlayer:
    """
    A player that selects moves uniformly at random from legal moves.
    """
    def get_move(self, game_state):
        """
        Select a random legal move.
        
        Args:
            game_state: TicTacToeGame instance
            
        Returns:
            action: integer from 0-8 representing the move
        """
        legal_moves = game_state.get_legal_moves()
        return np.random.choice(legal_moves)


class MCTSPlayer:
    """
    A player that uses MCTS with a neural network for move selection.
    """
    def __init__(self, network, num_simulations=50, temperature=0):
        """
        Args:
            network: TicTacToeNet instance
            num_simulations: Number of MCTS simulations per move
            temperature: Temperature for move selection (0 = greedy)
        """
        self.network = network
        self.mcts = MCTS(network, num_simulations=num_simulations)
        self.temperature = temperature
        
    def get_move(self, game_state):
        """
        Select a move using MCTS.
        
        Args:
            game_state: TicTacToeGame instance
            
        Returns:
            action: integer from 0-8 representing the move
        """
        action_probs = self.mcts.search(game_state)
        
        if self.temperature == 0:
            # Greedy: pick best move
            action = action_probs.argmax()
        else:
            # Stochastic: sample from distribution
            action_probs_temp = action_probs ** (1.0 / self.temperature)
            action_probs_temp = action_probs_temp / action_probs_temp.sum()
            action = np.random.choice(9, p=action_probs_temp)
        
        return action


# --- Parallel evaluation helpers ---
def _eval_single_game_worker(network_state_dict, num_simulations, start_as_x, temperature):
    """
    Worker function to evaluate a single game in a separate process.
    Reconstructs the network from state_dict to avoid CUDA/IPC issues.
    """
    from TTTnet import TicTacToeNet  # local import for multiprocessing safety

    # Reconstruct network and players
    net = TicTacToeNet()
    net.load_state_dict(network_state_dict)
    net.eval()

    player_net = MCTSPlayer(net, num_simulations=num_simulations, temperature=temperature)
    random_player = RandomPlayer()

    # Play one game
    if start_as_x:
        winner = play_game(player_net, random_player, verbose=False)
    else:
        winner = play_game(random_player, player_net, verbose=False)

    return winner


def play_game(player1, player2, verbose=False):
    """
    Play a single game between two players.
    
    Args:
        player1: Player object with get_move() method (plays X)
        player2: Player object with get_move() method (plays O)
        verbose: Whether to print the game
        
    Returns:
        winner: 1 if player1 wins, -1 if player2 wins, 0 for draw
    """
    game = TicTacToeGame()
    
    if verbose:
        print("\n--- Starting Game ---")
        game.print_board()
    
    while not game.is_game_over():
        current_player = player1 if game.current_player == 1 else player2
        move = current_player.get_move(game)
        game.make_move(move)
        
        if verbose:
            player_symbol = "X" if game.current_player == -1 else "O"  # Flipped because move was just made
            print(f"\n{player_symbol} plays move: {move}")
            game.print_board()
    
    winner = game.check_winner()
    
    if verbose:
        if winner == 1:
            print("\n--- X (Player 1) WINS! ---")
        elif winner == -1:
            print("\n--- O (Player 2) WINS! ---")
        else:
            print("\n--- DRAW! ---")
    
    return winner


def evaluate_against_random(network, num_games=100, num_mcts_sims=50, verbose=True, temperature=0, num_workers=None):
    """
    Evaluate a network against a random player.
    
    Args:
        network: TicTacToeNet instance to evaluate
        num_games: Number of games to play
        num_mcts_sims: Number of MCTS simulations per move
        verbose: Whether to print progress
        temperature: Move selection temperature for MCTSPlayer (0 = greedy)
        num_workers: Number of parallel workers (None or 1 = sequential)
        
    Returns:
        results: Dictionary with win/loss/draw statistics
    """
    network.eval()  # Set to evaluation mode

    # Prepare results container
    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'games_as_X': num_games // 2 + (num_games % 2),
        'games_as_O': num_games // 2,
        'wins_as_X': 0,
        'wins_as_O': 0
    }

    if verbose:
        print(f"Evaluating against random player ({num_games} games)...")

    # Parallel path
    if num_workers is not None and num_workers != 1:
        # Snapshot model weights for workers
        state_dict = network.state_dict()
        # Build task list alternating starting player
        tasks = [(state_dict, num_mcts_sims, (i % 2 == 0), temperature) for i in range(num_games)]

        with mp.Pool(processes=(mp.cpu_count() if num_workers is None else num_workers)) as pool:
            winners = pool.starmap(_eval_single_game_worker, tasks)

        # Aggregate
        for i, winner in enumerate(winners):
            if i % 2 == 0:
                # network played as X
                if winner == 1:
                    results['wins'] += 1
                    results['wins_as_X'] += 1
                elif winner == -1:
                    results['losses'] += 1
                else:
                    results['draws'] += 1
            else:
                # network played as O
                if winner == -1:
                    results['wins'] += 1
                    results['wins_as_O'] += 1
                elif winner == 1:
                    results['losses'] += 1
                else:
                    results['draws'] += 1
    else:
        # Sequential fallback (with progress updates)
        mcts_player = MCTSPlayer(network, num_simulations=num_mcts_sims, temperature=temperature)
        random_player = RandomPlayer()

        for i in range(num_games):
            if i % 2 == 0:
                winner = play_game(mcts_player, random_player, verbose=False)
                if winner == 1:
                    results['wins'] += 1
                    results['wins_as_X'] += 1
                elif winner == -1:
                    results['losses'] += 1
                else:
                    results['draws'] += 1
            else:
                winner = play_game(random_player, mcts_player, verbose=False)
                if winner == -1:
                    results['wins'] += 1
                    results['wins_as_O'] += 1
                elif winner == 1:
                    results['losses'] += 1
                else:
                    results['draws'] += 1

            if verbose and (i + 1) % 20 == 0:
                win_rate = results['wins'] / (i + 1) * 100
                print(f"  Progress: {i+1}/{num_games} games, Win rate: {win_rate:.1f}%")

    # Final rates
    results['win_rate'] = results['wins'] / num_games
    results['loss_rate'] = results['losses'] / num_games
    results['draw_rate'] = results['draws'] / num_games

    if verbose:
        print("\nEvaluation Results:")
        print(f"  Wins:   {results['wins']}/{num_games} ({results['win_rate']*100:.1f}%)")
        print(f"  Losses: {results['losses']}/{num_games} ({results['loss_rate']*100:.1f}%)")
        print(f"  Draws:  {results['draws']}/{num_games} ({results['draw_rate']*100:.1f}%)")
        print(f"  As X: {results['wins_as_X']}/{results['games_as_X']} wins")
        print(f"  As O: {results['wins_as_O']}/{results['games_as_O']} wins")

    return results


def calculate_elo(win_rate, opponent_elo=1000):
    """
    Estimate ELO rating based on win rate against an opponent.
    
    Uses the ELO formula:
    Expected score = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
    
    Solving for player_elo:
    player_elo = opponent_elo - 400 * log10((1 / win_rate) - 1)
    
    Args:
        win_rate: Win rate against opponent (between 0 and 1)
        opponent_elo: ELO rating of the opponent
        
    Returns:
        estimated_elo: Estimated ELO rating
    """
    # Clamp win rate to avoid division by zero or log of negative
    win_rate = max(0.01, min(0.99, win_rate))
    
    # Calculate ELO difference
    elo_diff = -400 * np.log10((1 / win_rate) - 1)
    
    estimated_elo = opponent_elo + elo_diff
    
    return estimated_elo


def print_elo_report(results, opponent_elo=1000):
    """
    Print a formatted ELO report based on evaluation results.
    
    Args:
        results: Dictionary from evaluate_against_random()
        opponent_elo: ELO rating of the opponent (default 1000 for random player)
    """
    # Calculate win rate (treating draws as 0.5 wins for ELO purposes)
    adjusted_win_rate = results['wins'] / (results['wins'] + results['losses'] + results['draws'])
    elo_win_rate = (results['wins'] + 0.5 * results['draws']) / (results['wins'] + results['losses'] + results['draws'])
    
    estimated_elo = calculate_elo(elo_win_rate, opponent_elo)
    
    print(f"\n{'='*60}")
    print("ELO RATING ESTIMATE")
    print(f"{'='*60}")
    print(f"Opponent ELO:     {opponent_elo}")
    print(f"Win rate:         {adjusted_win_rate*100:.1f}% ({results['wins']}-{results['losses']}-{results['draws']})")
    print(f"Adjusted score:   {elo_win_rate*100:.1f}% (draws count as 0.5)")
    print(f"Estimated ELO:    {estimated_elo:.0f}")
    print(f"ELO difference:   {estimated_elo - opponent_elo:+.0f}")
    print(f"{'='*60}")
    
    return estimated_elo
