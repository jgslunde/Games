import numpy as np
from random import randint, choice

EMPTY = 0
ATTACKER = 1
DEFENDER = -1
KING = -2

corners = np.array([[0, 0], [10, 0], [0, 10], [10,10]])
offsets = np.array([[-1,0], [0,-1], [1,0], [0,1]])

piece_names = {0: "·", 1: "♖", -1: "♜", -2: "♚"}
player_names = {0: "None", 1: "Attacker", -1: "Defender"}


class Board:
    def __init__(self):
        self.board = np.array(
            [[ 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [ 0 ,0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1],
            [ 1, 0, 0, 0,-1,-1,-1, 0, 0, 0, 1],
            [ 1, 1, 0,-1,-1,-2,-1,-1, 0, 1, 1],
            [ 1, 0, 0, 0,-1,-1,-1, 0, 0, 0, 1],
            [ 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]],
            dtype=np.int8)
        
        self.current_player = ATTACKER
        self.turn = 1
        self._legal_moves = None


    def print_board(self):
        board = self.board
        print(f"Turn {self.turn}, {player_names[self.current_player]} to move.")
        print(f"Current score: {self.get_current_score()}")
        printstring = ""
        printstring += f"   A B C D E F G H I J K\n"
        for x in range(11):
            printstring += f"{x+1:2d} "
            for y in range(11):
                if x == 5 and y == 5 and board [x,y] == EMPTY:
                    printstring += "○ "
                else:
                    printstring += piece_names[board[x,y]] + " "
            printstring += f" {x+1:d}"
            printstring += "\n"
        printstring += f"   A B C D E F G H I J K"
        print(printstring)


    @property
    def legal_moves(self):
        if self._legal_moves is None:
            player = self.current_player
            board = self.board
            self._legal_moves = []
            for x in range(11):
                for y in range(11):
                    if (player == ATTACKER and board[x,y] == ATTACKER) or (player == DEFENDER and (board[x,y] == DEFENDER or board[x,y] == KING)):
                        for x_new in range(x+1, 11, 1):
                            if board[x_new, y] == EMPTY:
                                self._legal_moves.append([[x,y],[x_new,y]])
                            else:
                                break
                        for x_new in range(x-1, -1, -1):
                            if board[x_new, y] == EMPTY:
                                self._legal_moves.append([[x,y],[x_new,y]])
                            else:
                                break
                        for y_new in range(y+1, 11, 1):
                            if board[x, y_new] == EMPTY:
                                self._legal_moves.append([[x,y],[x,y_new]])
                            else:
                                break
                        for y_new in range(y-1, -1, -1):
                            if board[x, y_new] == EMPTY:
                                self._legal_moves.append([[x,y],[x,y_new]])
                            else:
                                break

        return self._legal_moves


    def make_move(self, move):
        player = self.current_player
        board = self.board
        move_from, move_to = move
        if (player == ATTACKER and board[*move_from] != 1) or (player == DEFENDER and board[*move_from] not in [DEFENDER, KING]):
            raise ValueError(f"Player {player} tried moving piece {piece_names[board[*move_from]]}")
        if move not in self.legal_moves:
            raise ValueError(f"Move {move} is not a legal move.")
        board[*move_to] = board[*move_from]
        board[*move_from] = EMPTY

        for offset in offsets:  # Consider all neighboring squares for potential captures.
            if 0 <= move_to[0]+offset[0] <= 10 and 0 <= move_to[1]+offset[1] <= 10:  # If neighboring square is inside board.
                if board[move_to[0]+offset[0], move_to[1]+offset[1]]*player < 0 and board[move_to[0]+offset[0], move_to[1]+offset[1]] != KING:  # If neighboring piece is enemy but not king.
                    # For a enemy neighboring piece, look at the square one further in the same direction.
                    double_neighbor_x = move_to[0]+offset[0]*2
                    double_neighbor_y = move_to[1]+offset[1]*2
                    # Check if that square is "hostile", aka either: inside the board and has allied piece, or a corner square. 
                    if (0 <= double_neighbor_x <= 10 and 0 <= double_neighbor_y <= 10 and board[double_neighbor_x,double_neighbor_y]*player > 0)\
                        or (double_neighbor_x in [0, 10] and double_neighbor_y in [0,10]):
                        capture_x, capture_y = move_to[0]+offset[0], move_to[1]+offset[1]
                        print(f"##### Piece captured on {capture_x}, {capture_y}. #####")
                        board[capture_x,capture_y] = EMPTY
        print(f"Making move {move[0]} to {move[1]}.")

        self.current_player *= -1  # Switching current player.
        self._legal_moves = None  # Resetting legal moves, as they don't apply to new board state.


    def evaluate_win(self):
        board = self.board
        king_pos = np.argwhere(board == KING)[0]
        if (king_pos == corners[0]).all() or (king_pos == corners[1]).all() or (king_pos == corners[2]).all() or (king_pos == corners[3]).all():
            print("King in corner.")
            return DEFENDER
        king_surrounded = True
        for offset in offsets:
            pos = king_pos+offset
            if not ((0 < pos)*(pos < 11)).all():
                king_surrounded = False
            else:
                if board[*pos] != ATTACKER:
                    king_surrounded = False
        if king_surrounded:
            print("King surrounded.")
            return ATTACKER
        if np.sum(board == ATTACKER) == 0:
            print("No .")
            return DEFENDER
        if np.sum(board == DEFENDER) == 0:
            return ATTACKER
        return 0


    def get_current_score(self):
        board = self.board
        return np.sum(board == 1) - np.sum(board==-1)*2


    def play_random_move(self):
        # Plays a random move for the current player.
        legal_moves = self.legal_moves
        chosen_move = choice(legal_moves)
        self.make_move(chosen_move)