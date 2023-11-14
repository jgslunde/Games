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

    #pragma omp parallel for
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
    // for(int i=0; i<12; i++){
        // cout << i << " " << NUM_NODES[i] << endl;
    // }
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


TournamentResults AI_vs_AI_tournament(int num_games, int max_premoves, int depth1, int depth2, HeuristicsConfig *config1, HeuristicsConfig *config2){
    TournamentResults results;
    results.num_games = num_games;
    // uint64_t initial_atk_bb = 0x8080063000808;
    // uint64_t initial_def_bb = 0x814080000;
    // uint64_t initial_king_bb = 0x8000000;

    // int num_AI_1_wins_atk = 0;
    // int num_AI_1_wins_def = 0;
    // int num_AI_1_ties_atk = 0;
    // int num_AI_1_ties_def = 0;
    // int num_AI_2_wins_atk = 0;
    // int num_AI_2_wins_def = 0;
    // int num_ties = 0;
    int AI_1_playing_as = 1;
    // uint64_t starting_atk_bb = initial_atk_bb;
    // uint64_t starting_def_bb = initial_def_bb;
    // uint64_t starting_king_bb = initial_king_bb;
    AI ai_1(*config1);
    AI ai_2(*config2);
    Board starting_board;
    for(int game=0; game<num_games; game++){
        if(AI_1_playing_as == 1){  // Only create new board after both players had a chance to play both sides.
            int num_premoves = (max_premoves*game)/num_games;
            // if(vverbose){
            // cout << "Num premoves: " << num_premoves << endl;
            // }
            // starting_atk_bb = initial_atk_bb;
            // starting_def_bb = initial_def_bb;
            // starting_king_bb = initial_king_bb;
            uint64_t move_from, move_to, move_idx;
            vector<uint64_t> legal_moves;
            starting_board.reset_board();
            for(int ipremove=0; ipremove<num_premoves; ipremove++){
                // Attacker move.
                legal_moves = starting_board.get_all_legal_moves();
                move_idx = thread_safe_rand()%(legal_moves.size()/2);
                move_from = legal_moves[2*move_idx];
                move_to = legal_moves[2*move_idx+1];
                starting_board.make_move(move_from, move_to);
                // Defender move.
                legal_moves = starting_board.get_all_legal_moves();
                move_idx = thread_safe_rand()%(legal_moves.size()/2);
                move_from = legal_moves[2*move_idx];
                move_to = legal_moves[2*move_idx+1];
                starting_board.make_move(move_from, move_to);
            }
        }
        Board board = starting_board;
        // if(vverbose)
        // cout << "Game: " << game << endl;
        int current_player = 1;
        int score = 0;
        unsigned short int depth = 4;
        int iturn = 0;
        while (true){
            if(current_player == 1){
                depth=depth1;
            }else{
                depth=depth2;
            }
            Move preffered_move;
            if(current_player*AI_1_playing_as == 1)
                preffered_move = ai_1.get_preffered_move(board, depth);
            else
                preffered_move = ai_2.get_preffered_move(board, depth);
            // cout << preffered_move.move_from << " " << preffered_move.move_to << endl;
            if((preffered_move.move_from != 0) && (preffered_move.move_to != 0)){
                board.make_move(preffered_move.move_from, preffered_move.move_to);
                score = board.get_board_wins();
            }
            else{ // No legal moves.
                score = -1000*current_player;
            }
            // print_bitgame(board.atk_bb, board.def_bb, board.king_bb);
            if(abs(score) > 800){
                if(score < 0){
                    if(AI_1_playing_as == -1)
                        results.AI_1_wins_def++;
                    else
                        results.AI_2_wins_def++;
                    // if(vverbose){
                    //     cout << "DEFENDER WINS" << endl;
                    //     print_bitgame(atk_bb, def_bb, king_bb);
                    // }
                }else{
                    if(AI_1_playing_as == 1)
                        results.AI_1_wins_atk++;
                    else
                        results.AI_2_wins_atk++;
                    // if(vverbose){
                    //     cout << "ATTACKER WINS" << endl;
                    //     print_bitgame(atk_bb, def_bb, king_bb);
                    // }
                }
                break;
            }
            current_player *= -1;
            iturn++;
            if(iturn >= 100){
                if(AI_1_playing_as == 1)
                    results.AI_1_ties_atk++;
                else
                    results.AI_1_ties_def++;
                // if(vverbose){
                //     cout << "100 TURNS REACHED" << endl;
                //     print_bitgame(atk_bb, def_bb, king_bb);
                // }
                break;
            }
        }
        AI_1_playing_as *= -1;
    }
    // if(verbose){
    //     cout << "Total number of atk wins for AI_1: " << num_AI_1_wins_atk << endl;
    //     cout << "Total number of atk wins for AI_2: " << num_AI_2_wins_atk << endl;
    //     cout << "Total number of def wins for AI_1: " << num_AI_1_wins_def << endl;
    //     cout << "Total number of def wins for AI_2: " << num_AI_2_wins_def << endl;
    //     cout << "Total number of draws with AI_1 as atk: " << num_AI_1_ties_atk << endl;
    //     cout << "Total number of draws with AI_1 as def: " << num_AI_1_ties_def << endl;
    //     cout << "AI_1/AI_2 win ratio as atk: " << (float) num_AI_1_wins_atk / (float)num_AI_2_wins_atk << endl;
    //     cout << "AI_1/AI_2 win ratio as def: " << (float) num_AI_1_wins_def / (float)num_AI_2_wins_def << endl;
    //     cout << "AI_1_score: " << AI_1_score << endl;
    // }

    results.AI_1_score = (float) ((results.AI_1_wins_atk + results.AI_1_wins_def) - (results.AI_2_wins_atk + results.AI_2_wins_def))/(float) num_games;

    return results;
}