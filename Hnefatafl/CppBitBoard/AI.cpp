#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <random>
#include <thread>
#include <map>
#include <fstream>
#include <sstream>
#include <cmath>
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


int AI_vs_AI_game(Board board, int depth1, int depth2, HeuristicsConfig *config1, HeuristicsConfig *config2){
    // Given a starting board, two AI configs, and two AI depths, play a game between the two AIs and return who won (1=atk, 0=tie, -1=def).
    AI AI1(*config1);
    AI AI2(*config2);
    for(int iturn=0; iturn<100; iturn++){
        int current_player = 1;
        int score = 0;
        Move preffered_move;
        if(current_player == 1)
            preffered_move = AI1.get_preffered_move(board, depth1);
        else
            preffered_move = AI2.get_preffered_move(board, depth2);
        if((preffered_move.move_from != 0) && (preffered_move.move_to != 0)){
            board.make_move(preffered_move.move_from, preffered_move.move_to);
            score = board.get_board_wins();
        }
        else{
            return -current_player;
        }
        if(score > 800){
            return 1;
        }
        else if(score < -800){
            return -1;
        }
    }
    return 0;
}


TournamentResults one_vs_many_tournament(int num_games, int depth1, int depth2, HeuristicsConfig config, vector<HeuristicsConfig> configs_opponent_arr){
    TournamentResults results;

    // Reading in 2 move and 4 move boards from file.
    vector<Board> move2_boards;
    vector<Board> move4_boards;
    ifstream myfile;
    string line;
    myfile.open("openings/openings_even_2.txt");
    while(getline(myfile, line)){
        istringstream iss(line);
        Board board;
        iss >> board.atk_bb >> board.def_bb >> board.king_bb;
        move2_boards.push_back(board);
    }
    myfile.close();
    myfile.open("openings/openings_even_4.txt");
    while(getline(myfile, line)){
        istringstream iss(line);
        Board board;
        iss >> board.atk_bb >> board.def_bb >> board.king_bb;
        move4_boards.push_back(board);
    }
    myfile.close();

    #pragma omp parallel for
    for(int iopponent=0; iopponent<configs_opponent_arr.size(); iopponent++){  // Looping over all given opponents.
        for(int igame=0; igame<num_games; igame++){  // Looping over number of games for each opponent.
            Board starting_board;  // The default is a normal starting board (1/3 of games).
            if(igame%3 == 1){  // 1/3 of the games played with 2 random premoves.
                starting_board = move2_boards[thread_safe_rand()%move2_boards.size()];
            }
            else if(igame%3 == 2){  // 1/3 of the games played with 4 random premoves.
                starting_board = move4_boards[thread_safe_rand()%move4_boards.size()];
            }

            int AI_1_won_as_atk = AI_vs_AI_game(starting_board, depth1, depth2, &config, &configs_opponent_arr[iopponent]);  // Playing as attacker.
            int AI_2_won_as_atk = AI_vs_AI_game(starting_board, depth2, depth1, &configs_opponent_arr[iopponent], &config);  // Playing as defender.
            #pragma omp critical
            {
                results.AI_1_wins_atk += AI_1_won_as_atk == 1;
                results.AI_1_wins_def += AI_2_won_as_atk == -1;
                results.AI_2_wins_atk += AI_2_won_as_atk == 1;
                results.AI_2_wins_def += AI_1_won_as_atk == -1;
                results.AI_1_ties_atk += AI_1_won_as_atk == 0;
                results.AI_1_ties_def += AI_2_won_as_atk == 0;
            }
        }
    }
    results.num_games = 2*num_games*configs_opponent_arr.size();  // Both black and white, games per opponent, and num of opponents.
    results.AI_1_score = (float) ((results.AI_1_wins_atk + results.AI_1_wins_def) - (results.AI_2_wins_atk + results.AI_2_wins_def))/(float) results.num_games;
    return results;
}


TournamentResults AI_vs_AI_tournament(int num_games, int depth1, int depth2, HeuristicsConfig config1, HeuristicsConfig config2){
    TournamentResults results;

    // Reading in 2 move and 4 move boards from file.
    vector<Board> move2_boards;
    vector<Board> move4_boards;
    ifstream myfile;
    string line;
    myfile.open("openings/openings_even_2.txt");
    while(getline(myfile, line)){
        istringstream iss(line);
        Board board;
        iss >> board.atk_bb >> board.def_bb >> board.king_bb;
        move2_boards.push_back(board);
    }
    myfile.close();
    myfile.open("openings/openings_even_4.txt");
    while(getline(myfile, line)){
        istringstream iss(line);
        Board board;
        iss >> board.atk_bb >> board.def_bb >> board.king_bb;
        move4_boards.push_back(board);
    }
    myfile.close();

    #pragma omp parallel for
    for(int igame=0; igame<num_games; igame++){  // Looping over number of games for each opponent.
        Board starting_board;  // The default is a normal starting board (1/3 of games).
        // if(igame%3 == 1){  // 1/3 of the games played with 2 random premoves.
        //     starting_board = move2_boards[thread_safe_rand()%move2_boards.size()];
        // }
        // else if(igame%3 == 2){  // 1/3 of the games played with 4 random premoves.
        //     starting_board = move4_boards[thread_safe_rand()%move4_boards.size()];
        // }

        int AI_1_won_as_atk = AI_vs_AI_game(starting_board, depth1, depth2, &config1, &config2);  // Playing as attacker.
        int AI_2_won_as_atk = AI_vs_AI_game(starting_board, depth2, depth1, &config2, &config1);  // Playing as defender.
        #pragma omp critical
        {
            results.AI_1_wins_atk += AI_1_won_as_atk == 1;
            results.AI_1_wins_def += AI_2_won_as_atk == -1;
            results.AI_2_wins_atk += AI_2_won_as_atk == 1;
            results.AI_2_wins_def += AI_1_won_as_atk == -1;
            results.AI_1_ties_atk += AI_1_won_as_atk == 0;
            results.AI_1_ties_def += AI_2_won_as_atk == 0;
        }
    }
    results.num_games = 2*num_games;  // Both black and white, games per opponent, and num of opponents.
    results.AI_1_score = (float) ((results.AI_1_wins_atk + results.AI_1_wins_def) - (results.AI_2_wins_atk + results.AI_2_wins_def))/(float) results.num_games;

    // float P_win_atk = 2.0*results.AI_1_wins_atk/(float) results.num_games;
    // float P_win_def = 2.0*results.AI_1_wins_def/(float) results.num_games;
    float P_win = (results.AI_1_wins_atk + results.AI_1_wins_def)/(float) results.num_games;
    // float P_draw_atk = 2.0*results.AI_1_ties_atk/(float) results.num_games;
    // float P_draw_def = 2.0*results.AI_1_ties_def/(float) results.num_games;
    float P_draw = (results.AI_1_ties_atk + results.AI_1_ties_def)/(float) results.num_games;
    // results.AI_1_atk_rel_elo = 400.0 * log10(1.0/(P_win_atk + 0.5*P_draw_atk) - 1.0);
    // results.AI_1_def_rel_elo = 400.0 * log10(1.0/(P_win_def + 0.5*P_draw_def) - 1.0);
    results.AI_1_rel_elo = -400.0*log10(1.0/(P_win + 0.5*P_draw) - 1.0);
    return results;
}



void print_tournament_results(TournamentResults results){
    printf("ELO diff     W/D/L (as atk)   W/D/L (as def)\n");
    printf("%6.1f       %.0f%%/%.0f%%/%.0f%%       %.0f%%/%.0f%%/%.0f%%\n", results.AI_1_rel_elo, 100.0*results.AI_1_wins_atk/(0.5*results.num_games), 100.0*results.AI_1_ties_atk/(0.5*results.num_games), 100.0*results.AI_2_wins_def/(0.5*results.num_games), 100.0*results.AI_1_wins_def/(0.5*results.num_games), 100.0*results.AI_1_ties_def/(0.5*results.num_games), 100.0*results.AI_2_wins_atk/(0.5*results.num_games));
}