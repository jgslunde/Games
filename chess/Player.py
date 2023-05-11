import random
import numpy as np
import time
import copy
from tqdm import trange
from Board import Board
from multiprocessing import Pool

N_EVALS = 0

def return_best_move(board, depth, max_depth=3):
    move_scores = np.zeros(len(board.legal_moves))
    move_strings = ["" for i in range(len(board.legal_moves))]
    for i_move in range(len(board.legal_moves)):
        tmp_board = copy.deepcopy(board)
        state = tmp_board.make_move(tmp_board.current_player, i_move)
        global N_EVALS
        N_EVALS += 1
        if depth == max_depth:
            move_scores[i_move] = tmp_board.score
        else:
            if state != 0:  # Stalemate or check-mate. Do not continue.
                move_scores[i_move] = tmp_board.score
            else:
                move_scores[i_move], move_strings[i_move] = return_best_move(tmp_board, depth+1, max_depth)
    if board.current_player == 1:
        idx = np.argmax(move_scores)
        return move_scores[idx], " | " + board.legal_moves[idx][0] + move_strings[idx]
    else:
        idx = np.argmin(move_scores)
        return move_scores[idx], " | " + board.legal_moves[idx][0] + move_strings[idx]


def get_move_scores(board, max_depth=3):
    move_scores = np.zeros(len(board.legal_moves))
    move_strings = ["" for i in range(len(board.legal_moves))]
    for i_move in trange(len(board.legal_moves)):
        tmp_board = copy.deepcopy(board)
        state = tmp_board.make_move(tmp_board.current_player, i_move)
        global N_EVALS
        N_EVALS += 1
        if state != 0:
            move_scores[i_move] = tmp_board.score
        else:
            move_scores[i_move], move_strings[i_move] = return_best_move(tmp_board, 1, max_depth)
    return move_scores, move_strings


def width_search_player(board, max_depth=3):
    move_scores, move_strings = get_move_scores(board, max_depth)
    if board.current_player == 1:
        best_move_idx = np.argmax(move_scores)
    else:
        best_move_idx = np.argmin(move_scores)
    if move_scores[best_move_idx] == 0:  # If no moves produced benefit
        return random.randint(0, len(move_scores)-1)
    else:
        return best_move_idx
    
    
    

    
if __name__ == "__main__":
    board1 = np.array([
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0, -4,  0,  0,  0,  0,  0],
        [ 0,  0,  0, -6,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0, +3,  0,  0,  0],
        [+6,  0,  0,  0,  0,  0,  0,  0],
    ], dtype=np.int16,
    )

    board2 = np.array([
        [-6,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0, -4,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0, -4,  0,  0,  0,  0],
        [+6,  0,  0,  0,  0,  0,  0,  0],
    ], dtype=np.int16,
    )    



    random.seed(3)
    board = Board(board1)
    # board.make_move(1, 1)
    board.display_board()
    scores, strings = get_move_scores(board, 3)
    move_scores = scores - board.score
    best_move_idx = np.argmax(move_scores)
    print(f"Scores   Moves")
    for i in range(len(board.legal_moves)):
        print(f"{move_scores[i]:4.0f}   {board.legal_moves[i][0]:14s}  {strings[i]}")
        
    print(N_EVALS)