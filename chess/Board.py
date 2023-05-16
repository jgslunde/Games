import random
import numpy as np
import time
import ctypes
from tqdm import trange

# Known bugs:
# - Rooks, queens and bishops can move through enemies.
# - Threat it not calculated for piece targeting friendly piece, but it should.

Clib = ctypes.cdll.LoadLibrary("/home/jonas/github/Games/chess/tools.so.1")
int16_array2 = np.ctypeslib.ndpointer(dtype=ctypes.c_short, ndim=2, flags="contiguous")
int16_array1 = np.ctypeslib.ndpointer(dtype=ctypes.c_short, ndim=1, flags="contiguous")
bool_array2 = np.ctypeslib.ndpointer(dtype=bool, ndim=2, flags="contiguous")
Clib.get_all_threat_moves.argtypes = [int16_array2, int16_array1, ctypes.c_long]
Clib.get_all_legal_moves.argtypes = [int16_array2, int16_array1, ctypes.c_long]
Clib.get_all_possible_moves.argtypes = [int16_array2, int16_array1, ctypes.c_long]
Clib.get_threat_board.argtypes = [bool_array2, int16_array2, ctypes.c_long]



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



board0 = np.array([
    [-4, -2, -3, -5, -6, -3, -2, -4],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [+1, +1, +1, +1, +1, +1, +1, +1],
    [+4, +2, +3, +5, +6, +3, +2, +4],
], dtype=np.int16,
)



pieces_dict = ["  ", "P+", "N+", "B+", "R+", "Q+", "K+", "K-", "Q-", "R-", "B-", "N-", "P-"]

knight_moves = np.array([[-2, -1], [-1, -2], [1, -2], [-2, 1], [2, -1], [-1, 2], [2, 1], [1, 2]])

def is_inside_board(y, x):
    return 0 <= x < 8 and 0 <= y < 8

def idx_to_board_pos(y, x):
    y_board = str(8 - y)
    if x == 0:
        x_board = "a"
    elif x == 1:
        x_board = "b"
    elif x == 2:
        x_board = "c"
    elif x == 3:
        x_board = "d"
    elif x == 4:
        x_board = "e"
    elif x == 5:
        x_board = "f"
    elif x == 6:
        x_board = "g"
    elif x == 7:
        x_board = "h"
    return x_board + y_board


def board_pos_to_idx(board_pos):
    x_board = board_pos[0]
    y_board = board_pos[1]
    y = 8 - int(y_board)
    if x_board == "a":
        x = 0
    elif x_board == "b":
        x = 1
    elif x_board == "c":
        x = 2
    elif x_board == "d":
        x = 3
    elif x_board == "e":
        x = 4
    elif x_board == "f":
        x = 5
    elif x_board == "g":
        x = 6
    elif x_board == "h":
        x = 7
    return y, x


def score_calc(piece):
    if piece == +2 or piece == -2:
        return piece*1.5
    elif piece == 4 or piece == -4:
        return piece*5/4
    elif piece == 5 or piece == -5:
        return piece*9/5
    elif piece == 6 or piece == -6:
        return piece*100
    else:
        return piece


def get_threat_board(board, player):
    threat_board = np.zeros((8, 8), dtype=bool)
    possible_moves = get_all_possible_moves(board, player)
    for move in possible_moves:
        string, piece, move_start, move_end, is_capture = move
        if is_capture:
            threat_board[*move_end] = True
    return threat_board


def get_all_possible_moves(board, player):
    possible_moves = []
    for board_y in range(8):
        for board_x in range(8):
            pos = np.array([board_y, board_x])
            piece = board[board_y, board_x]
            piece_str = pieces_dict[piece]
            if player*piece > 0:  # If there is a white/black piece at x/y.
                # print(piece_str, idx_to_board_pos(board_y, board_x), "(", piece, board_y, board_x, ")")

                # Pawn
                if piece == 1*player:
                    # 1 step forward
                    new_pos = (board_y-1*player, board_x)
                    if is_inside_board(*new_pos):
                        if board[*new_pos] == 0:  # Pawns can't capture forward, so no piece can be at new pos.
                            possible_moves.append([f"{piece_str}: {idx_to_board_pos(board_y, board_x)} ==> {idx_to_board_pos(*new_pos)}", piece, pos, new_pos, False])
                            # print(piece_str, idx_to_board_pos(board_y, board_x), "==>", idx_to_board_pos(*new_pos))
                    # The diagonal captures
                    for dx in [-1, 1]:
                        new_pos = (board_y-1*player, board_x+dx)
                        if is_inside_board(*new_pos):
                            possible_moves.append([f"{piece_str}: {idx_to_board_pos(board_y, board_x)} ==> {idx_to_board_pos(*new_pos)}", piece, pos, new_pos, True])
                            # print(piece_str, idx_to_board_pos(board_y, board_x), "==>", idx_to_board_pos(*new_pos))
                    # The double forward
                    if player == 1 and pos[0] == 6:
                        if board[pos[0]-1, pos[1]] == 0:
                            new_pos = pos + np.array([-2,0])
                            if is_inside_board(*new_pos):
                                if board[*new_pos] == 0:
                                    # print(piece_str, idx_to_board_pos(board_y, board_x), "==>", idx_to_board_pos(*new_pos))
                                    possible_moves.append([f"{piece_str}: {idx_to_board_pos(board_y, board_x)} ==> {idx_to_board_pos(*new_pos)}", piece, pos, new_pos, False])
                    elif player == -1 and pos[0] == 1:
                        if board[pos[0]+1, pos[1]] == 0:
                            new_pos = pos + np.array([+2,0])
                            if is_inside_board(*new_pos):
                                if board[*new_pos] == 0:
                                    possible_moves.append([f"{piece_str}: {idx_to_board_pos(board_y, board_x)} ==> {idx_to_board_pos(*new_pos)}", piece, pos, new_pos, False])
                                    # print(piece_str, idx_to_board_pos(board_y, board_x), "==>", idx_to_board_pos(*new_pos))
                        
                
                # Knight
                elif piece == 2*player:
                    for move in knight_moves:
                        new_pos = pos + move
                        if is_inside_board(*new_pos):
                            if board[*new_pos]*player <= 0:  # If no friendy piece at new pos.
                                possible_moves.append([f"{piece_str}: {idx_to_board_pos(board_y, board_x)} ==> {idx_to_board_pos(*new_pos)}", piece, pos, new_pos, True])
                                # print(piece_str, idx_to_board_pos(board_y, board_x), "==>", idx_to_board_pos(*new_pos))

                # Bishop
                elif piece == 3*player:
                    for dx in [-1, 1]:
                        for dy in [-1, 1]:
                            for length in range(1, 8):
                                move = np.array([dx, dy])*length
                                new_pos = pos + move
                                if is_inside_board(*new_pos):
                                    if board[*new_pos]*player <= 0:
                                        # print(piece_str, idx_to_board_pos(board_y, board_x), "==>", idx_to_board_pos(*new_pos))
                                        possible_moves.append([f"{piece_str}: {idx_to_board_pos(board_y, board_x)} ==> {idx_to_board_pos(*new_pos)}", piece, pos, new_pos, True])
                                    if board[*new_pos] != 0:
                                        break  # If any piece is encountered in this direction, the rest of the diagonal will be illegal.
                                else:
                                    break
                
                # Rook
                elif piece == 4*player:
                    for dir in np.array([[-1, 0], [1, 0], [0, -1], [0,1]]):
                        for length in range(1, 8):
                            move = dir*length
                            new_pos = pos + move
                            if is_inside_board(*new_pos):
                                if board[*new_pos]*player <= 0:
                                    possible_moves.append([f"{piece_str}: {idx_to_board_pos(board_y, board_x)} ==> {idx_to_board_pos(*new_pos)}", piece, pos, new_pos, True])
                                    # print(piece_str, idx_to_board_pos(board_y, board_x), "==>", idx_to_board_pos(*new_pos))
                                if board[*new_pos] != 0:
                                    break  # If any piece is encountered in this direction, the rest of the moves will be illegal.
                            else:
                                break
                
                # Queen
                elif piece == 5*player:
                    for dir in np.array([[-1, 0], [1, 0], [0, -1], [0,1]]):
                        for length in range(1, 8):
                            move = dir*length
                            new_pos = pos + move
                            if is_inside_board(*new_pos):
                                if board[*new_pos]*player <= 0:
                                    possible_moves.append([f"{piece_str}: {idx_to_board_pos(board_y, board_x)} ==> {idx_to_board_pos(*new_pos)}", piece, pos, new_pos, True])
                                    # print(piece_str, idx_to_board_pos(board_y, board_x), "==>", idx_to_board_pos(*new_pos))
                                if board[*new_pos] != 0:
                                    break  # If any piece is encountered in this direction, the rest of the moves will be illegal.
                            else:
                                break
                    for dx in [-1, 1]:
                        for dy in [-1, 1]:
                            for length in range(1, 8):
                                move = np.array([dx, dy])*length
                                new_pos = pos + move
                                if is_inside_board(*new_pos):
                                    if board[*new_pos]*player <= 0:
                                        possible_moves.append([f"{piece_str}: {idx_to_board_pos(board_y, board_x)} ==> {idx_to_board_pos(*new_pos)}", piece, pos, new_pos, True])
                                        # print(piece_str, idx_to_board_pos(board_y, board_x), "==>", idx_to_board_pos(*new_pos))
                                    if board[*new_pos] != 0:
                                        break  # If any piece is encountered in this direction, the rest of the diagonal will be illegal.
                                else:
                                    break
                
                # King
                elif piece == 6*player:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                move = np.array([dy, dx])
                                new_pos = pos + move
                                if is_inside_board(*new_pos):
                                    if board[*new_pos]*player <= 0:
                                        # print(piece_str, idx_to_board_pos(board_y, board_x), "==>", idx_to_board_pos(*new_pos))
                                        possible_moves.append([f"{piece_str}: {idx_to_board_pos(board_y, board_x)} ==> {idx_to_board_pos(*new_pos)}", piece, pos, new_pos, True])
    return possible_moves



def C_get_all_possible_moves(board, player):
    moves = np.zeros((600), dtype=np.int16)
    num_moves = Clib.get_all_possible_moves(board, moves, player)
    moves = moves.reshape(100,6)[:num_moves]
    possible_moves = []
    for move in moves:
        string = f"{pieces_dict[move[0]]}: {idx_to_board_pos(move[2], move[3])} => {idx_to_board_pos(move[4], move[5])}"
        possible_moves.append([string, move[0], (move[2], move[3]), (move[4], move[5]), move[1]])
    # print([legal_move[0] for legal_move in legal_moves])
    return possible_moves
    


def get_all_legal_moves(board, player):
    if True:
        moves = np.zeros((600), dtype=np.int16)
        num_moves = Clib.get_all_legal_moves(board, moves, player)
        moves = moves.reshape(100,6)[:num_moves]
        legal_moves = []
        for move in moves:
            string = f"{pieces_dict[move[0]]}: {idx_to_board_pos(move[2], move[3])} => {idx_to_board_pos(move[4], move[5])}"
            legal_moves.append([string, move[0], (move[2], move[3]), (move[4], move[5]), move[1]])
        # print([legal_move[0] for legal_move in legal_moves])
        return legal_moves

    else:
        enemy_threat_board = get_threat_board(board, -player)  # Get board of all squares opponent can attack (for king movement).
        possible_moves = get_all_possible_moves(board, player)
        legal_moves = []
        king_pos = np.argwhere(board == player*6)[0]
        if enemy_threat_board[*king_pos]:
            check = True
        else:
            check = False
        for move in possible_moves:
            string, piece, move_start, move_end, is_capture = move
            if piece == player*1 and move_end[1] != move_start[1]:  # The diagonal pawn moves are only legal if there is an enemy piece there.
                if board[*move_end]*player < 0:                       # We sort out the diagonal moves specifically by looking for a change in x-direction.
                    potential_board = board.copy()
                    potential_board[*move[3]] = potential_board[*move[2]]
                    potential_board[*move[2]] = 0
                    potential_enemy_threat_board = get_threat_board(potential_board, -player)  # Get board of all squares opponent can attack (for king movement).
                    potential_king_pos = np.argwhere(potential_board == player*6)[0]
                    if not potential_enemy_threat_board[*potential_king_pos]:
                        legal_moves.append(move)
            else:
                potential_board = board.copy()
                potential_board[*move[3]] = potential_board[*move[2]]
                potential_board[*move[2]] = 0
                potential_enemy_threat_board = get_threat_board(potential_board, -player)  # Get board of all squares opponent can attack (for king movement).
                potential_king_pos = np.argwhere(potential_board == player*6)[0]
                if not potential_enemy_threat_board[*potential_king_pos]:
                    legal_moves.append(move)

        return legal_moves



class Board:
    def __init__(self, board=board0):
        self.board = board.copy()
        self.white_can_castle = True
        self.black_can_castle = True
        self.turn = 1
        self.full_turn = 1
        self.current_player = 1
        self.legal_moves = get_all_legal_moves(self.board, self.current_player)
        self.PGN = ""
        self.score = 0
        for row in range(8):
            for col in range(8):
                self.score += score_calc(self.board[row,col])
        


    def display_board(self):
        printstring = f"TURN {self.full_turn}, SCORE {self.score} "
        if self.current_player == 1:
            printstring += "WHITE TO MOVE:\n"
        else:
            printstring += "BLACK TO MOVE:\n"
        printstring += f"{bcolors.OKBLUE}  a  b  c  d  e  f  g  h{bcolors.ENDC}\n"
        for y in range(8):
            printstring += f"{bcolors.OKBLUE}{8 - y}{bcolors.ENDC} "
            for x in range(8):
                printstring += pieces_dict[self.board[y,x]] + " "
            printstring += "\n"
        printstring += f"{bcolors.OKBLUE}  a  b  c  d  e  f  g  h{bcolors.ENDC}"
        print(printstring)
    
    
    def eval_sufficient_material(self):
        white_material = [np.sum(self.board == i) for i in range(1, 7)]
        black_material = [np.sum(self.board == i) for i in range(-1, -7, -1)]

        if white_material[0] > 0 or black_material[0] > 0:  # Pawns
            return True
        elif white_material[4] > 0 or black_material[4] > 0:  # Queens
           return True
        elif white_material[3] > 0 or black_material[3] > 0:  # Rooks
           return True
        elif white_material[1] + white_material[2] + black_material[1] + black_material[2] > 1:
            return True
        else:
            return False


    def make_move(self, player, legal_move_idx):
        if player != self.current_player:
            raise ValueError()
        if player == 1:
            self.PGN += f"{self.full_turn}. "

        move = self.legal_moves[legal_move_idx]
        string, piece, move_start, move_end, is_capture = move
        if self.board[*move_end] != 0:
            self.score -= score_calc(self.board[*move_end])
        self.board[*move_end] = self.board[*move_start]
        self.board[*move_start] = 0
        if move_end[0] == 0 or move_end[0] == 7:
            if abs(self.board[*move_end]) == 1:
                self.score += 8*player
                self.board[*move_end] = player*5
        self.PGN += f"{string[0]}{string[4:6]}{string[-2:]} "
        
        self.current_player *= -1
        
        self.legal_moves = get_all_legal_moves(self.board, self.current_player)
        if len(self.legal_moves) == 0:
            threat_board = get_threat_board(self.board, -self.current_player)
            king_pos = np.argwhere(self.board == self.current_player*6)[0]
            if threat_board[*king_pos]:
                self.score = -1000*self.current_player
                # print(f"VICTORY FOR {-self.current_player}.")
                return -self.current_player
            else:
                self.score = 0
                # print(f"STALEMATE")
                return 99
                
        
        if not self.eval_sufficient_material():
            self.score = 0
            # print(f"STALEMATE")
            return 99

        self.turn += 1
        if player == -1:
            self.full_turn += 1
        return 0


    def make_string_move(self, player, move_from, move_to):
        state = -99
        y_start, x_start = board_pos_to_idx(move_from)
        y_stop, x_stop = board_pos_to_idx(move_to)
        for i_move, move in enumerate(self.legal_moves):
            string, piece, move_start, move_end, is_capture = move
            # print(y_start, move_start[0], x_start, move_start[1], y_stop, move_end[0], x_stop, move_end[1])
            if move_start[0] == y_start and move_start[1] == x_start:
                if move_end[0] == y_stop and move_end[1] == x_stop:
                    state = self.make_move(player, i_move)
                    break
        return state


if __name__ == "__main__":
    random.seed(41)
    board = Board()
    state = 0
    for iturn in trange(10000):
        get_all_legal_moves(board.board, 1)
        # if state != 0:
        #     break
        # for player in [1, -1]:
        #     if state != 0:
        #         break
        #     board.display_board()
        #     move_idx = random.randint(0, len(board.legal_moves)-1)
        #     state = board.make_move(player, move_idx)
        #     if state != 0:
        #         if state == 99:
        #             print("STALEMATE")
        #         else:
        #             print(f"VICTORY FOR PLAYER {np.sign(board.score)}.")
    board.display_board()
    print(board.PGN)