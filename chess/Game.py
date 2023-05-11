import random
import numpy as np
import time
import copy
from tqdm import trange
from Board import Board
from Player import width_search_player


if __name__ == "__main__":
    random.seed(3)
    board = Board()
    state = 0
    for iturn in range(400):
        if state != 0:
            break
        for player in [1, -1]:
            if state != 0:
                break
            board.display_board()
            if player == 1:
                move_idx = random.randint(0, len(board.legal_moves)-1)
            else:
                move_idx = width_search_player(board, 2)
            print("Moved", board.legal_moves[move_idx][0])
            state = board.make_move(player, move_idx)
    board.display_board()
    print(board.PGN)


    # random.seed(41)
    # board = Board()
    # state = 0
    # for iturn in range(400):
    #     if state != 0:
    #         break
    #     for player in [1, -1]:
    #         if state != 0:
    #             break
    #         board.display_board()
    #         if player == 1:
    #             was_legal = -99
    #             while was_legal == -99:
    #                 move_from = input("Move from:")
    #                 move_to = input("Move to:")
    #                 was_legal = board.make_string_move(player, move_from, move_to)
    #                 if was_legal == -99:
    #                     print("Illegal move, please try again.")
    #             state = was_legal
    #         elif player == -1:
    #             move_idx = random.randint(0, len(board.legal_moves)-1)
    #             state = board.make_move(player, move_idx)
    # board.display_board()
    # print(board.PGN)
    
    # random.seed(41)
    # board = Board()
    # state = 0
    # for iturn in range(400):
    #     if state != 0:
    #         break
    #     for player in [1, -1]:
    #         if state != 0:
    #             break
    #         board.display_board()
    #         width_search_player(board)
    #         move_idx = random.randint(0, len(board.legal_moves)-1)
    #         state = board.make_move(player, move_idx)
    # board.display_board()
    # print(board.PGN)
    