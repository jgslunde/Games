#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <random>
#include <thread>
#include <map>
#include "Tools.h"
#include "Board.h"
#include "Heuristics.h"
#include "AI.h"

using namespace std;

vector<uint64_t> NUM_NODES(12);


AI::AI(HeuristicsConfig config){
    // Set the class attribute "config" to the provided config.
    this->config = config;
};


Move AI::get_preffered_move(Board board, ushort max_depth){
    int player = board.get_current_player();
    vector<uint64_t> legal_moves = board.get_all_legal_moves();
    int num_legal_moves = legal_moves.size()/2;
    if(num_legal_moves == 0){
        return Move{0, 0, 0.0};
    }
    int preffered_move_idx = 0;
    vector<float> move_scores(num_legal_moves);

    NUM_NODES[0] = num_legal_moves;

    // #pragma omp parallel for
    for(int i=0; i<num_legal_moves; i++){
        uint64_t atk_bb_new, def_bb_new, king_bb_new;
        // atk_bb_new = atk_bb;
        // def_bb_new = def_bb;
        // king_bb_new = king_bb;
        Board board_new = board;
        board_new.make_move(legal_moves[2*i], legal_moves[2*i+1]);
        float move_score;
        move_score = get_board_score_by_alpha_beta_search(board_new, 1, max_depth, -INFINITY, INFINITY);
        move_scores[i] = move_score;
    }

    // Finding the preffered (highest or lowest, depending on player) score among the options.
    float best_move_score = -9999999.0*player;
    for(int i=0; i<num_legal_moves; i++){
        if(move_scores[i]*player > best_move_score*player){
            best_move_score = move_scores[i];
        }
    }
    // Finding all the moves with the preffered score.
    vector<int> preffered_move_indices;
    for(int i=0; i<num_legal_moves; i++){
        if(move_scores[i] == best_move_score){
            preffered_move_indices.push_back(i);
        }
    }
    // Chosing a random among those.
    uint64_t chosen_move_index = preffered_move_indices[thread_safe_rand()%preffered_move_indices.size()];

    // if(verbose){
    //     cout << "Player: " << player << ". Board eval: " << best_move_score << endl;
    // }
    // *eval = best_move_score;
    for(int i=0; i<12; i++){
        cout << i << " " << NUM_NODES[i] << endl;
    }
    Move chosen_move;
    chosen_move.move_from = legal_moves[2*chosen_move_index];
    chosen_move.move_to = legal_moves[2*chosen_move_index+1];
    chosen_move.eval = best_move_score;
    return chosen_move;
}



float AI::get_board_score_by_alpha_beta_search(Board board, unsigned short int depth, unsigned short int max_depth, float alpha, float beta){
    int player = board.get_current_player();
    float board_wins = board.get_board_wins();
    if((depth >= max_depth) || (abs(board_wins) > 100)){ // Base case: terminal depth or leaf node
        return (combined_board_heuristics(board, &config) + board_wins)*(1 - 0.01*(depth-1));
    }

    vector<uint64_t> legal_moves = board.get_all_legal_moves();
    int num_legal_moves = legal_moves.size() / 2;
    if(num_legal_moves == 0){
        return (float) -1000*player;
    }
    NUM_NODES[depth] += num_legal_moves;

    if(player == PLAYER_ATK){
        float best_score = -999999;
        for(int imove = 0; imove < num_legal_moves; imove++){
            // uint64_t atk_bb_new = atk_bb;
            // uint64_t def_bb_new = def_bb;
            // uint64_t king_bb_new = king_bb;
            Board board_new = board;
            board_new.make_move(legal_moves[2*imove], legal_moves[2*imove + 1]);
            float score = get_board_score_by_alpha_beta_search(board_new, depth + 1, max_depth, alpha, beta);
            best_score = max(best_score, score);
            alpha = max(alpha, best_score);
            if(beta <= alpha){ // Beta cut-off
                break;
            }
        }
        return best_score;
    }
    else{
        float best_score = 999999;
        for(int imove = 0; imove < num_legal_moves; imove++){
            // uint64_t atk_bb_new = atk_bb;
            // uint64_t def_bb_new = def_bb;
            // uint64_t king_bb_new = king_bb;
            Board board_new = board;
            board_new.make_move(legal_moves[2*imove], legal_moves[2*imove + 1]);
            float score = get_board_score_by_alpha_beta_search(board_new, depth + 1, max_depth, alpha, beta);
            best_score = min(best_score, score);
            beta = min(beta, best_score);
            if(beta <= alpha){ // Alpha cut-off
                break;
            }
        }
        return best_score;
    }
}
