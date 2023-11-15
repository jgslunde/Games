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
        // float AI_score = 0;
        // for(int idx2=0; idx2<Nopponentconfigs; idx2++){
        //     HeuristicsConfig opponent_config = opponent_config_arr[idx2];
        //     TournamentResults results = AI_vs_AI_tournament(num_battles, 2, 2, 2, &current_config, &opponent_config);
        //     AI_score += results.AI_1_score;
        // }
        TournamentResults results = one_vs_many_tournament(num_battles, 2, 2, current_config, opponent_config_arr);
        AI_scores[idx] = results.AI_1_score;
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


void SPSA_optimization(HeuristicsConfig initial_config, vector<HeuristicsConfig> opponent_config_arr, double alpha, double sigma, int num_iter, int num_battles, string outfile){
    vector<HeuristicsConfig> all_configs;
    HeuristicsConfig current_config = initial_config;
    all_configs.reserve(num_iter+1);
    all_configs.push_back(initial_config);
    ofstream myfile;
    myfile.open(outfile);
    for(int i=0; i<num_iter; i++){
        cout << i << " / " << num_iter << endl;
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::normal_distribution<float> distribution(0.0, sigma);
        HeuristicsConfig delta_weights = {
            0,
            distribution(generator),
            distribution(generator),
            distribution(generator),
            distribution(generator),
            distribution(generator),
            distribution(generator),
            distribution(generator),
            distribution(generator),
        };

        // cout << "Proposal step from " << current_config.def_pieces_weight << "  " << current_config.king_free_moves_weight << endl;
        // cout << "Step size " << delta_weights.def_pieces_weight << "  " << delta_weights.king_free_moves_weight << endl;

        HeuristicsConfig config_plus = current_config;
        config_plus.def_pieces_weight += delta_weights.def_pieces_weight;
        config_plus.king_free_moves_weight += delta_weights.king_free_moves_weight;
        config_plus.king_neighboring_enemies_weight += delta_weights.king_neighboring_enemies_weight;
        config_plus.king_neighboring_allies_weight += delta_weights.king_neighboring_allies_weight;
        config_plus.atk_pieces_on_edges_weight += delta_weights.atk_pieces_on_edges_weight;
        config_plus.atk_pieces_diag_to_corners_weight += delta_weights.atk_pieces_diag_to_corners_weight;
        config_plus.atk_pieces_next_to_corners_weight += delta_weights.atk_pieces_next_to_corners_weight;
        config_plus.def_pieces_next_to_corners_weight += delta_weights.def_pieces_next_to_corners_weight;

        HeuristicsConfig config_minus = current_config;
        config_minus.def_pieces_weight -= delta_weights.def_pieces_weight;
        config_minus.king_free_moves_weight -= delta_weights.king_free_moves_weight;
        config_minus.king_neighboring_enemies_weight - delta_weights.king_neighboring_enemies_weight;
        config_minus.king_neighboring_allies_weight - delta_weights.king_neighboring_allies_weight;
        config_minus.atk_pieces_on_edges_weight - delta_weights.atk_pieces_on_edges_weight;
        config_minus.atk_pieces_diag_to_corners_weight - delta_weights.atk_pieces_diag_to_corners_weight;
        config_minus.atk_pieces_next_to_corners_weight - delta_weights.atk_pieces_next_to_corners_weight;
        config_minus.def_pieces_next_to_corners_weight - delta_weights.def_pieces_next_to_corners_weight;

        float AI_plus_score = 0;  // The win performance score of the "plus delta" AI. Number between -1 and 1.
        
        // Then they both play against the random AIs. The plus AI lose points if the minus AI wins.
        for(int idx2=0; idx2<opponent_config_arr.size(); idx2++){    
            TournamentResults result_plus = AI_vs_AI_tournament(num_battles, 2, 2, 2, &config_plus, &opponent_config_arr[idx2]);
            TournamentResults result_minus = AI_vs_AI_tournament(num_battles, 2, 2, 2, &config_minus, &opponent_config_arr[idx2]);
            AI_plus_score += result_plus.AI_1_score;
            AI_plus_score -= result_minus.AI_1_score;
        }
        AI_plus_score /= opponent_config_arr.size();

        cout << "AI plus win rate score = " << AI_plus_score << endl;
        cout << "Step size = " << alpha*AI_plus_score << endl;

        // current_config.atk_pieces_weight += alpha*AI_plus_score*delta_weights.atk_pieces_weight;
        current_config.def_pieces_weight += alpha*AI_plus_score*delta_weights.def_pieces_weight;
        current_config.king_free_moves_weight += alpha*AI_plus_score*delta_weights.king_free_moves_weight;
        current_config.king_neighboring_enemies_weight += alpha*AI_plus_score*delta_weights.king_neighboring_enemies_weight;
        current_config.king_neighboring_allies_weight += alpha*AI_plus_score*delta_weights.king_neighboring_allies_weight;
        current_config.atk_pieces_on_edges_weight += alpha*AI_plus_score*delta_weights.atk_pieces_on_edges_weight;
        current_config.atk_pieces_diag_to_corners_weight += alpha*AI_plus_score*delta_weights.atk_pieces_diag_to_corners_weight;
        current_config.atk_pieces_next_to_corners_weight += alpha*AI_plus_score*delta_weights.atk_pieces_next_to_corners_weight;
        current_config.def_pieces_next_to_corners_weight += alpha*AI_plus_score*delta_weights.def_pieces_next_to_corners_weight;

        all_configs.push_back(current_config);
        myfile << current_config.atk_pieces_weight << " ";
        myfile << current_config.def_pieces_weight << " ";
        myfile << current_config.king_free_moves_weight << " ";
        myfile << current_config.king_neighboring_enemies_weight << " ";
        myfile << current_config.king_neighboring_allies_weight << " ";
        myfile << current_config.atk_pieces_on_edges_weight << " ";
        myfile << current_config.atk_pieces_diag_to_corners_weight << " ";
        myfile << current_config.atk_pieces_next_to_corners_weight << " ";
        myfile << current_config.def_pieces_next_to_corners_weight << " ";
        myfile << AI_plus_score << " ";
        myfile << endl;
    }
    myfile.close();
}