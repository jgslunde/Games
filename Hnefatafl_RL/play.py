"""
Play Brandubh with two random players.
Displays the game state and move history.
"""

import random
from brandubh import Brandubh


def play_random_game(display=True, delay=0):
    """
    Play a game with two random players.
    
    Args:
        display: Whether to print the board state
        delay: Delay between moves (in seconds)
    
    Returns:
        winner: 0 for Attackers, 1 for Defenders
        num_moves: Number of moves in the game
    """
    game = Brandubh()
    move_count = 0
    
    if display:
        print("=" * 50)
        print("Starting new game of Brandubh")
        print("=" * 50)
        print("\nInitial board:")
        print(game)
        print("\n")
    
    while not game.game_over:
        # Get legal moves
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            # No legal moves - opponent wins
            game.game_over = True
            game.winner = 1 - game.current_player
            break
        
        # Choose random move
        move = random.choice(legal_moves)
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
                print("Game exceeded 500 moves - calling it a draw (Attackers win by default)")
            game.game_over = True
            game.winner = 0
            break
    
    if display:
        print("=" * 50)
        winner_name = "Attackers" if game.winner == 0 else "Defenders"
        print(f"Game Over! Winner: {winner_name}")
        print(f"Total moves: {move_count}")
        print("=" * 50)
    
    return game.winner, move_count


def play_multiple_games(num_games=100):
    """
    Play multiple games and collect statistics.
    """
    attacker_wins = 0
    defender_wins = 0
    total_moves = 0
    
    print(f"Playing {num_games} games...\n")
    
    for i in range(num_games):
        winner, moves = play_random_game(display=False)
        
        if winner == 0:
            attacker_wins += 1
        else:
            defender_wins += 1
        
        total_moves += moves
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_games} games...")
    
    print("\n" + "=" * 50)
    print("Statistics:")
    print("=" * 50)
    print(f"Total games: {num_games}")
    print(f"Attacker wins: {attacker_wins} ({100 * attacker_wins / num_games:.1f}%)")
    print(f"Defender wins: {defender_wins} ({100 * defender_wins / num_games:.1f}%)")
    print(f"Average game length: {total_moves / num_games:.1f} moves")
    print("=" * 50)


if __name__ == "__main__":
    # Play a single game with display
    print("Playing a single game with visualization:\n")
    play_random_game(display=True)
    
    # Uncomment to play multiple games and see statistics
    # print("\n\n")
    # play_multiple_games(num_games=100)
