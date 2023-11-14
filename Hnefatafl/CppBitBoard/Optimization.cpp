#include <iostream>
#include <string>
#include <fstream>
#include "Heuristics.h"
#include "AI.h"
#include "Tools.h"
#include "Optimization.h"

using namespace std;

void grid_search(vector<HeuristicsConfig> search_config_arr, vector<HeuristicsConfig> opponent_config_arr, int num_battles, string outfile){
    int Nconfigs = search_config_arr.size();
    int Nopponentconfigs = search_config_arr.size();
    vector<float> AI_scores(Nconfigs);
    HeuristicsConfig initial_config;

    #pragma omp parallel for schedule(static, 1)
    for(int idx=0; idx<Nconfigs; idx++){
        cout << idx << "/" << Nconfigs << endl;
        HeuristicsConfig current_config;
        current_config = search_config_arr[idx];
        float AI_score = 0;
        for(int idx2=0; idx2<Nopponentconfigs; idx2++){
            HeuristicsConfig opponent_config = opponent_config_arr[idx2];
            TournamentResults results = AI_vs_AI_tournament(num_battles, 2, 2, 2, &current_config, &opponent_config);
            AI_score += results.AI_1_score;
        }
        AI_scores[idx] = AI_score;
        // AI_scores[idx] = modified_AI_vs_AI_tournament(4000, 2, 2, &current_config, &initial_config, false, false);
    }
    ofstream myfile;
    myfile.open(outfile);
    for(int idx=0; idx<Nconfigs; idx++){
        float AI_score = AI_scores[idx];
        HeuristicsConfig current_config = search_config_arr[idx];
        myfile << AI_score << " ";
        myfile << current_config.atk_pieces_weight << " ";
        myfile << current_config.def_pieces_weight << " ";
        myfile << current_config.king_free_moves_weight << " ";
        myfile << current_config.king_neighboring_enemies_weight << " ";
        myfile << current_config.king_neighboring_allies_weight << " ";
        myfile << current_config.atk_pieces_on_edges_weight << " ";
        myfile << current_config.atk_pieces_diag_to_corners_weight << " ";
        myfile << current_config.atk_pieces_next_to_corners_weight << " ";
        myfile << current_config.def_pieces_next_to_corners_weight << " ";
        myfile << endl;
    }
}