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
#include "Optimization.h"


int main(){
    // HeuristicsConfig config;
    // AI ai(config);
    
    // Board board;
    // board.reset_board();
    
    // Move move = ai.get_preffered_move(board, 7);
    // cout << move.eval << endl;
    // board.make_move(move.move_from, move.move_to);

    // print_bitgame(board.atk_bb, board.def_bb, board.king_bb);

    // AI_vs_AI_tournament(2, 0, 2, 2, &config, &config);

    vector<HeuristicsConfig> random_opponents_arr(1000);
    for(int i=0; i<500; i++){
        random_opponents_arr[i].def_pieces_weight = thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].king_free_moves_weight = thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].king_neighboring_enemies_weight = thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].king_neighboring_allies_weight = thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].atk_pieces_on_edges_weight = thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].atk_pieces_diag_to_corners_weight = thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].atk_pieces_next_to_corners_weight = thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].def_pieces_next_to_corners_weight = thread_safe_rand_float(-0.5, 2.5);
    }
    for(int i=500; i<1000; i++){
        random_opponents_arr[i].def_pieces_weight = thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].king_free_moves_weight = thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].king_neighboring_enemies_weight = thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].king_neighboring_allies_weight = thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].atk_pieces_on_edges_weight = thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].atk_pieces_diag_to_corners_weight = thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].atk_pieces_next_to_corners_weight = thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].def_pieces_next_to_corners_weight = thread_safe_rand_float(0.5, 1.5);
    }

    int N1 = 31;
    int N2 = 41;
    vector<HeuristicsConfig> config_arr(N1*N2);
    for(int i=0; i<N1; i++){
        for(int j=0; j<N2; j++){
            int idx = i*N2+j;
            HeuristicsConfig current_config;
            current_config.atk_pieces_diag_to_corners_weight = -0.5 + i*0.1;
            current_config.atk_pieces_on_edges_weight = -0.5 + j*0.1;
            config_arr[idx] = current_config;
        }
    }

    // grid_search(config_arr, random_opponents_arr, 4, "data_new/grid_31x41_atk_diagcorners_onedges.txt");
    HeuristicsConfig config;
    SPSA_optimization(config, random_opponents_arr, 20.0, 0.1, 1000, 4, "data_new/SPSA_1000_alpha20_sigma01.txt");
}