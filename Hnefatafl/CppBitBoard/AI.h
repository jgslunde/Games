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