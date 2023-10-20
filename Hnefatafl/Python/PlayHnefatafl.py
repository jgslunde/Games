from HnefataflBoard import Board

player_names = {0: "None", 1: "Attacker", -1: "Defender"}

if __name__ == "__main__":
    board = Board()
    for turn in range(2000):
        board.play_random_move()
        board.print_board()
        win = board.evaluate_win()
        if win != 0:
            print(f"PLAYER {player_names[-board.current_player]} WON THE GAME!")
            break