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
    // HeuristicsConfig config;
    // SPSA_optimization(config, random_opponents_arr, 20.0, 0.1, 2000, 4, "data_new/SPSA_2000_alpha20_sigma01.txt");


    vector<HeuristicsConfig> random_opponents_arr(2000);
    for(int i=0; i<1000; i++){
        random_opponents_arr[i].def_pieces_weight                   *= thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].king_free_moves_weight              *= thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].king_neighboring_enemies_weight     *= thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].king_neighboring_allies_weight      *= thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].atk_pieces_on_edges_weight          *= thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].atk_pieces_diag_to_corners_weight   *= thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].atk_pieces_next_to_corners_weight   *= thread_safe_rand_float(-0.5, 2.5);
        random_opponents_arr[i].def_pieces_next_to_corners_weight   *= thread_safe_rand_float(-0.5, 2.5);
    }
    for(int i=1000; i<2000; i++){
        random_opponents_arr[i].def_pieces_weight                   *= thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].king_free_moves_weight              *= thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].king_neighboring_enemies_weight     *= thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].king_neighboring_allies_weight      *= thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].atk_pieces_on_edges_weight          *= thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].atk_pieces_diag_to_corners_weight   *= thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].atk_pieces_next_to_corners_weight   *= thread_safe_rand_float(0.5, 1.5);
        random_opponents_arr[i].def_pieces_next_to_corners_weight   *= thread_safe_rand_float(0.5, 1.5);
    }

    int N1 = 61;
    int N2 = 62;
    vector<HeuristicsConfig> config_arr(N1*N2);
    for(int i=0; i<N1; i++){
        for(int j=0; j<N2; j++){
            int idx = i*N2+j;
            HeuristicsConfig current_config;
            current_config.def_pieces_weight = -2.5 + i*0.05;
            current_config.king_free_moves_weight = -0.5 + j*0.01;
            config_arr[idx] = current_config;
        }
    }
    grid_search(config_arr, random_opponents_arr, 3, "data/grid_61x62_defpieces_vs_kingfreedom_N2000.txt");


    int N1 = 61;
    int N2 = 62;
    vector<HeuristicsConfig> config_arr(N1*N2);
    for(int i=0; i<N1; i++){
        for(int j=0; j<N2; j++){
            int idx = i*N2+j;
            HeuristicsConfig current_config;
            current_config.king_neighboring_enemies_weight = -0.1 + i*0.02;
            current_config.king_neighboring_allies_weight = -1.0 + j*0.02;
            config_arr[idx] = current_config;
        }
    }
    grid_search(config_arr, random_opponents_arr, 3, "data/grid_61x62_kingenemies_vs_kingallies_N2000.txt");


    int N1 = 61;
    int N2 = 62;
    vector<HeuristicsConfig> config_arr(N1*N2);
    for(int i=0; i<N1; i++){
        for(int j=0; j<N2; j++){
            int idx = i*N2+j;
            HeuristicsConfig current_config;
            current_config.atk_pieces_on_edges_weight = -0.1 + i*0.01;
            current_config.atk_pieces_diag_to_corners_weight = -0.1 + j*0.01;
            config_arr[idx] = current_config;
        }
    }
    grid_search(config_arr, random_opponents_arr, 3, "data/grid_61x62_atkonedges_vs_atkdiag2corners_N2000.txt");


    int N1 = 61;
    int N2 = 62;
    vector<HeuristicsConfig> config_arr(N1*N2);
    for(int i=0; i<N1; i++){
        for(int j=0; j<N2; j++){
            int idx = i*N2+j;
            HeuristicsConfig current_config;
            current_config.atk_pieces_next_to_corners_weight = -0.9 + i*0.02;
            current_config.def_pieces_next_to_corners_weight = -0.3 + j*0.02;
            config_arr[idx] = current_config;
        }
    }
    grid_search(config_arr, random_opponents_arr, 3, "data/grid_61x62_atknext2corners_vs_defnext2corners_N2000.txt");

}