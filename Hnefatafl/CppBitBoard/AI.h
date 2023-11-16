#pragma once

#include <vector>
#include "Board.h"
#include "Heuristics.h"


class AI{
    private:
        HeuristicsConfig config;
    public:
        AI(HeuristicsConfig config);
        Move get_preffered_move(Board board, unsigned short max_depth);
        float get_board_score_by_alpha_beta_search(Board board, unsigned short int depth, unsigned short int max_depth, float alpha, float beta);
};


struct TournamentResults{
    int num_games = 0;
    int AI_1_wins_atk = 0;
    int AI_1_wins_def = 0;
    int AI_1_ties_atk = 0;
    int AI_1_ties_def = 0;
    int AI_2_wins_atk = 0;
    int AI_2_wins_def = 0;
    float AI_1_score = 0.0;
    float AI_1_rel_elo = 0.0;
    // float AI_1_atk_rel_elo = 0.0;
    // float AI_1_def_rel_elo = 0.0;
};

int AI_vs_AI_game(Board board, int depth1, int depth2, HeuristicsConfig *config1, HeuristicsConfig *config2);
TournamentResults one_vs_many_tournament(int num_games, int depth1, int depth2, HeuristicsConfig config, vector<HeuristicsConfig> configs_opponent_arr);
TournamentResults AI_vs_AI_tournament(int num_games, int depth1, int depth2, HeuristicsConfig config1, HeuristicsConfig config2);