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



void perform_grid_search(){
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



void perform_SPSA_optimization(){
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


    HeuristicsConfig config1{
          1.0, // atk_pieces_weight
         -0.3, // def_pieces_weight
         -0.1, // king_free_moves_weight
          0.2, // king_neighboring_enemies_weight
        -0.08, // king_neighboring_allies_weight
         0.06, // atk_pieces_on_edges_weight
          0.1, // atk_pieces_diag_to_corners_weight
        -0.15, // atk_pieces_next_to_corners_weight
         0.14, // def_pieces_next_to_corners_weight
    };
    SPSA_optimization(config1, random_opponents_arr, 10.0, 0.01, 2000, 3, "data/SPSA_2000_alpha10_sigma001_run1.txt");

    HeuristicsConfig config2{
          1.0, // atk_pieces_weight
         -0.2, // def_pieces_weight
         -0.15, // king_free_moves_weight
          0.3, // king_neighboring_enemies_weight
        -0.02, // king_neighboring_allies_weight
         0.1, // atk_pieces_on_edges_weight
          0.15, // atk_pieces_diag_to_corners_weight
        -0.25, // atk_pieces_next_to_corners_weight
         -0.02, // def_pieces_next_to_corners_weight
    };
    SPSA_optimization(config2, random_opponents_arr, 10.0, 0.01, 2000, 3, "data/SPSA_2000_alpha10_sigma001_run2.txt");

    HeuristicsConfig config3{
          1.0, // atk_pieces_weight
         -0.6, // def_pieces_weight
         -0.4, // king_free_moves_weight
          0.1, // king_neighboring_enemies_weight
        -0.2, // king_neighboring_allies_weight
         -0.01, // atk_pieces_on_edges_weight
          -0.01, // atk_pieces_diag_to_corners_weight
        0.1, // atk_pieces_next_to_corners_weight
         0.3, // def_pieces_next_to_corners_weight
    };
    SPSA_optimization(config3, random_opponents_arr, 10.0, 0.01, 2000, 3, "data/SPSA_2000_alpha10_sigma001_run3.txt");

    HeuristicsConfig config4{
          1.0, // atk_pieces_weight
          0.0, // def_pieces_weight
          0.0, // king_free_moves_weight
          0.0, // king_neighboring_enemies_weight
          0.0, // king_neighboring_allies_weight
          0.0, // atk_pieces_on_edges_weight
          0.0, // atk_pieces_diag_to_corners_weight
          0.0, // atk_pieces_next_to_corners_weight
          0.0, // def_pieces_next_to_corners_weight
    };
    SPSA_optimization(config4, random_opponents_arr, 10.0, 0.01, 2000, 3, "data/SPSA_2000_alpha10_sigma001_run4.txt");


}


int main(){

    // perform_SPSA_optimization();

    // AI ai(config);
    
    // Board board;
    // board.reset_board();
    
    // Move move = ai.get_preffered_move(board, 7);
    // cout << move.eval << endl;
    // board.make_move(move.move_from, move.move_to);

    // print_bitgame(board.atk_bb, board.def_bb, board.king_bb);

    HeuristicsConfig config1;
    HeuristicsConfig config2;
    // config1.king_on_open_edge_weight = -99.0;
    TournamentResults results = AI_vs_AI_tournament(1000, 2, 2, config1, config2);
    cout << results.AI_1_score << endl;
    cout << results.AI_1_rel_elo << endl;

    cout << results.AI_1_wins_atk << endl;
    cout << results.AI_1_wins_def << endl;
    cout << results.AI_2_wins_atk << endl;
    cout << results.AI_2_wins_def << endl;
    cout << results.AI_1_ties_atk << endl;
    cout << results.AI_1_ties_def << endl;
    }