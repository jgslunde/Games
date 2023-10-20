from HnefataflBoard7x7 import Board

player_names = {0: "None", 1: "Attacker", -1: "Defender"}

if __name__ == "__main__":
    board = Board()
    for turn in range(100):
        if board.current_player == 1:
            board.play_AI_1()
        else:
            board.play_AI_1()
        board.print_board()
        win = board.evaluate_win()
        if win != 0:
            print(f"PLAYER {player_names[-board.current_player]} WON THE GAME!")
            break