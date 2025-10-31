import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TicTacToeGame:
    """
    A minimalistic Tic-Tac-Toe game class.
    It handles game state, moves, and win conditions.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the game to an empty board."""
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1 # 1 for X, -1 for O

    def get_legal_moves(self):
        """Returns a numpy array of indices for empty squares."""
        return np.where(self.board == 0)[0]

    def is_game_over(self):
        """Checks if the game is over (win or draw)."""
        return self.check_winner() is not None or len(self.get_legal_moves()) == 0

    def check_winner(self):
        """
        Checks for a winner.
        Returns: 1 if X wins, -1 if O wins, 0 for draw, None if ongoing.
        """
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], # Cols
            [0, 4, 8], [2, 4, 6]             # Diagonals
        ]
        
        for condition in win_conditions:
            s = self.board[condition].sum()
            if s == 3:
                return 1 # X wins
            if s == -3:
                return -1 # O wins
        
        # Check for draw
        if len(self.get_legal_moves()) == 0:
            return 0 # Draw
            
        return None # Game is not over

    def make_move(self, action):
        """
        Makes a move on the board.
        'action' is an integer from 0 to 8.
        Returns True if the move was legal, False otherwise.
        """
        if action in self.get_legal_moves():
            self.board[action] = self.current_player
            self.current_player *= -1 # Switch player
            return True
        return False

    def get_nn_input(self):
        """
        Gets the board state as a [1, 9] tensor for the network.
        """
        # The network expects floats
        return torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)

    def print_board(self):
        """Prints a human-readable version of the board."""
        symbols = {1: 'X', -1: 'O', 0: '.'}
        b = [symbols[p] for p in self.board]
        print("\n" + "-" * 11)
        print(f" {b[0]} | {b[1]} | {b[2]}")
        print("---+---+---")
        print(f" {b[3]} | {b[4]} | {b[5]}")
        print("---+---+---")
        print(f" {b[6]} | {b[7]} | {b[8]}")
        print("-" * 11)