import numpy as np
from random import randint

EMPTY = 0
ATTACKER = 1
DEFENDER = -1
KING = -2

corners = np.array([[0, 0], [10, 0], [0, 10], [10,10]])
offsets = np.array([[-1,0], [0,-1], [1,0], [0,1]])

# piece_names = {0: "o", 1: "A", 2: "D", 3: "K"}
piece_names = {0: "·", 1: "♖", -1: "♜", -2: "♚"} # ♔
player_names = {0: "None", 1: "Attacker", -1: "Defender"}

start_board = np.array(
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
     [ 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]]
    )


def get_legal_moves(board, player):
    legal_moves = []
    for x in range(11):
        for y in range(11):
            if (player == ATTACKER and board[x,y] == ATTACKER) or (player == DEFENDER and (board[x,y] == DEFENDER or board[x,y] == KING)):
                for x_new in range(x+1, 11, 1):
                    if board[x_new, y] == EMPTY:
                        legal_moves.append([[x,y],[x_new,y]])
                    else:
                        break
                for x_new in range(x-1, -1, -1):
                    if board[x_new, y] == EMPTY:
                        legal_moves.append([[x,y],[x_new,y]])
                    else:
                        break
                for y_new in range(y+1, 11, 1):
                    if board[x, y_new] == EMPTY:
                        legal_moves.append([[x,y],[x,y_new]])
                    else:
                        break
                for y_new in range(y-1, -1, -1):
                    if board[x, y_new] == EMPTY:
                        legal_moves.append([[x,y],[x,y_new]])
                    else:
                        break
    return legal_moves

def print_board(board):
    printstring = ""
    printstring += f"    A B C D E F G H I J K\n\n"
    for x in range(11):
        printstring += f"{x+1:2d}  "
        for y in range(11):
            printstring += piece_names[board[x,y]] + " "
        printstring += f"  {x+1:2d}"
        printstring += "\n"
    printstring += f"\n    A B C D E F G H I J K"
    print(printstring)


def make_move_on_board(board, player, move):
    move_from, move_to = move
    if (player == ATTACKER and board[*move_from] != 1) or (player == DEFENDER and board[*move_from] not in [DEFENDER, KING]):
        raise ValueError(f"Player {player} tried moving piece {piece_names[board[*move_from]]}")
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


def evaluate_win(board):
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


def get_current_score(board):
    return np.sum(board == 1) - np.sum(board==-1)*2


board = start_board.copy()
player = ATTACKER
for turn in range(1, 2001):
    legal_moves = get_legal_moves(board, player)
    print(f"Turn {turn}, {player_names[player]} to move.")
    print(f"Current score: {get_current_score(board)}")
    print(f"Num legal moves: {len(legal_moves)}.")
    chosen_move = legal_moves[randint(0, len(legal_moves)-1)]
    print(f"Making move {chosen_move[0]} to {chosen_move[1]}.")
    make_move_on_board(board, player, chosen_move)
    print_board(board)
    win = evaluate_win(board)
    if win != 0:
        print(f"PLAYER {player_names[player]} WON THE GAME!")
        break
    player *= -1