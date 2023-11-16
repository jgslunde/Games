#pragma once

#include "Board.h"

struct HeuristicsConfig {
    float atk_pieces_weight                 =  1.0;
    float def_pieces_weight                 = -0.3;
    float king_free_moves_weight            = -0.1;
    float king_neighboring_enemies_weight   =  0.2;
    float king_neighboring_allies_weight    = -0.08;
    float atk_pieces_on_edges_weight        =  0.06;
    float atk_pieces_diag_to_corners_weight =  0.1;
    float atk_pieces_next_to_corners_weight = -0.15;
    float def_pieces_next_to_corners_weight =  0.14;
    float king_on_open_edge_weight          =  0.0;
};

float combined_board_heuristics(Board &board, HeuristicsConfig *config);