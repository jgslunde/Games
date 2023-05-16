import numpy as np
import matplotlib.pyplot as plt
import random
import keras
from keras import Sequential
from keras.layers import Dense
import sys
sys.path.append("/home/jonas/github/Games/chess/")
from Board import Board

def board2nnboard(board):
    board_out = np.zeros((768))
    for idx, i in enumerate([-1, -2, -3, -4, -5, -6, 6, 5, 4, 3, 2, 1]):
        board_out[idx*64:(idx+1)*64] = board.flatten() == i
    return board_out
    
if __name__ == "__main__":
    model = keras.models.load_model("model")
    
    board = Board()
    nn_board = board2nnboard(board.board)
    prediction = model.predict(nn_board[None,:])
    move_from = np.unravel_index(np.argmax(prediction[0,:64].reshape((8,8))), (8,8))
    move_to = np.unravel_index(np.argmax(prediction[0,64:].reshape((8,8))), (8,8))

    state = 0
    tries_array = np.zeros((60))
    for iturn in range(60):
        if state != 0:
            break
        for player in [1, -1]:
            if state != 0:
                break
            board.display_board()
            if player == 1:
                nn_board = board2nnboard(board.board)
                prediction = model.predict(nn_board[None,:])
                move_from_list = np.argsort(prediction[:,:64], axis=-1)[0][::-1]
                move_to_list = np.argsort(prediction[:,64:], axis=-1)[0][::-1]
                # print(move_from_list)
                # print(move_to_list)
                tries = 0
                illegal = True
                for i_move_from in range(move_to_list.shape[0]):
                    for i_move_to in range(move_from_list.shape[0]):
                        if illegal:
                            move_from = np.unravel_index(move_from_list[i_move_from], (8,8))
                            move_to = np.unravel_index(move_to_list[i_move_to], (8,8))
                            move_idx = None
                            for imove, move in enumerate(board.legal_moves):
                                if move_from[0] == move[2][0] and move_from[1] == move[2][1] and move_to[0] == move[3][0] and move_to[1] == move[3][1]:
                                    move_idx = imove
                            tries += 1
                            if not move_idx is None:
                                illegal = False
                tries_array[iturn] = tries
                print(tries)
                if illegal:
                    raise ValueError()
            else:
                move_idx = random.randint(0, len(board.legal_moves)-1)
            state = board.make_move(player, move_idx)
            if state != 0:
                if state == 99:
                    print("STALEMATE")
                else:
                    print(f"VICTORY FOR PLAYER {np.sign(board.score)}.")
    
    print(np.mean(tries_array))
    plt.semilogy(tries_array)
    plt.xlabel("Move number")
    plt.ylabel("Attempts before legal move")
    # plt.yticks([1, 2, 3, 5, 10, 20, 40, 100])
    plt.show()
    
    plt.figure()
    plt.hist(tries_array, bins=51, range=(0, 50))
    plt.show()