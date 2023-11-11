#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <random>
#include <thread>
#include <map>
#include <random>
#include "Tools.h"
#include "Board.h"
#include "Heuristics.h"
#include "AI.h"


int main(){
    HeuristicsConfig config;
    AI ai(config);
    
    Board board;
    board.reset_board();
    
    Move move = ai.get_preffered_move(board, 7);
    cout << move.eval << endl;
    board.make_move(move.move_from, move.move_to);

    print_bitgame(board.atk_bb, board.def_bb, board.king_bb);

}