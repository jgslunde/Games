import numpy as np
from tqdm import trange
import chess.pgn

def move2matrix(move):
    x_from = move[0]
    y_from = move[1]
    x_to = move[2]
    y_to = move[3]

    y_from_idx = 8 - int(y_from)
    if x_from == "a":
        x_from_idx = 0
    elif x_from == "b":
        x_from_idx = 1
    elif x_from == "c":
        x_from_idx = 2
    elif x_from == "d":
        x_from_idx = 3
    elif x_from == "e":
        x_from_idx = 4
    elif x_from == "f":
        x_from_idx = 5
    elif x_from == "g":
        x_from_idx = 6
    elif x_from == "h":
        x_from_idx = 7

    y_to_idx = 8 - int(y_to)
    if x_to == "a":
        x_to_idx = 0
    elif x_to == "b":
        x_to_idx = 1
    elif x_to == "c":
        x_to_idx = 2
    elif x_to == "d":
        x_to_idx = 3
    elif x_to == "e":
        x_to_idx = 4
    elif x_to == "f":
        x_to_idx = 5
    elif x_to == "g":
        x_to_idx = 6
    elif x_to == "h":
        x_to_idx = 7
        
    matrix = np.zeros((128), dtype=bool)
    matrix[y_from_idx*8 + x_from_idx] = 1
    matrix[64 + y_to_idx*8 + x_to_idx] = 1
    
    return matrix
    



def board2matrix(board):
    matrix = np.zeros((768))
    idx = 0
    for item in "".join(board.split()):
        if idx < 64:
            if item == "\n":
                pass
            else:
                if item == "p":
                    matrix[0*64 + idx] = 1
                elif item == "n":
                    matrix[1*64 + idx] = 1
                elif item == "b":
                    matrix[2*64 + idx] = 1
                elif item == "r":
                    matrix[3*64 + idx] = 1
                elif item == "q":
                    matrix[4*64 + idx] = 1
                elif item == "k":
                    matrix[5*64 + idx] = 1
                elif item == "K":
                    matrix[6*64 + idx] = 1
                elif item == "Q":
                    matrix[7*64 + idx] = 1
                elif item == "R":
                    matrix[8*64 + idx] = 1
                elif item == "B":
                    matrix[9*64 + idx] = 1
                elif item == "N":
                    matrix[10*64 + idx] = 1
                elif item == "P":
                    matrix[11*64 + idx] = 1
                idx += 1
    return matrix

Nmoves = int(1e7)
all_boards = np.zeros((Nmoves, 768), dtype=bool)
all_moves = np.zeros((Nmoves, 128), dtype=bool)

pgn = open("/home/jonas/Downloads/lichess_db_standard_rated_2023-04.pgn")

imove = 0
for igame in trange(1000000):
    if imove < Nmoves:
        # if igame < 50000:
            # continue
        first_game = chess.pgn.read_game(pgn)
        board = first_game.board()
        board_matrix = board2matrix(str(board))
        white = True
        for move in first_game.mainline_moves():
            if imove < Nmoves:
                board.push(move)
                if white:
                    all_boards[imove] = board_matrix
                    move_matrix = move2matrix(str(move))
                    all_moves[imove] = move_matrix
                    imove += 1
                    board_matrix = board2matrix(str(board))
                    white = False
                else:
                    white = True
            else:
                break

np.save("all_boards_large.npy", all_boards)
np.save("all_moves_large.npy", all_moves)