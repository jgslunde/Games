import numpy as np
import torch
from TicTacToe import TicTacToeGame
from TTTnet import TicTacToeNet

# --- --- --- --- --- --- --- ---
#  Example of two NNs playing a game
# --- --- --- --- --- --- --- ---
if __name__ == "__main__":
    
    # 1. Instantiate two (untrained) networks
    net1 = TicTacToeNet()
    net2 = TicTacToeNet()
    
    # 2. Instantiate the game environment
    game = TicTacToeGame()
    
    # 3. Set networks to evaluation mode
    net1.eval()
    net2.eval()
    
    print("--- Starting Game: Net 1 (X) vs Net 2 (O) ---")
    game.print_board()

    # 4. Main game loop
    while True:
        
        # 5. Determine the current player and network
        current_net = net1 if game.current_player == 1 else net2
        player_symbol = "X" if game.current_player == 1 else "O"
        
        # 6. Get the NN's move
        with torch.no_grad():
            board_tensor = game.get_nn_input()
            policy_output, value_output = current_net(board_tensor)
            
            # Get legal moves
            legal_moves = game.get_legal_moves()
            
            # Squeeze batch dim and convert to numpy
            policy_probs = policy_output.squeeze(0).numpy()
            
            # Create a mask to zero out illegal moves
            legal_mask = np.zeros(9)
            legal_mask[legal_moves] = 1
            
            # Apply mask
            masked_policy = policy_probs * legal_mask
            
            # Re-normalize probabilities
            if masked_policy.sum() > 0:
                masked_policy = masked_policy / masked_policy.sum()
            else:
                # This happens if the policy network (by chance)
                # assigns 0 probability to all legal moves.
                # We fall back to a uniform random choice.
                masked_policy[legal_moves] = 1.0 / len(legal_moves)
            
            # Choose the move with the highest probability
            move = masked_policy.argmax()

        # 7. Make the move
        game.make_move(move)
        
        print(f"\n{player_symbol} (Net {1 if player_symbol == 'X' else 2}) plays move: {move}")
        print(masked_policy)
        game.print_board()
        
        # 8. Check for game over
        winner = game.check_winner()
        if winner is not None:
            if winner == 1:
                print("\n--- GAME OVER: X (Net 1) WINS! ---")
            elif winner == -1:
                print("\n--- GAME OVER: O (Net 2) WINS! ---")
            elif winner == 0:
                print("\n--- GAME OVER: IT'S A DRAW! ---")
            break

