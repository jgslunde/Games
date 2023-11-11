#pragma once

struct HeuristicsConfig {
    float atk_pieces_weight = 1.0;
    float def_pieces_weight = 1.0;
    float king_free_moves_weight = 1.0;
    float king_neighboring_enemies_weight = 1.0;
    float king_neighboring_allies_weight = 1.0;
    float atk_pieces_on_edges_weight = 1.0;
    float atk_pieces_diag_to_corners_weight = 1.0;
    float atk_pieces_next_to_corners_weight = 1.0;
    float def_pieces_next_to_corners_weight = 1.0;
};

float combined_board_heuristics(Board &board, HeuristicsConfig *config);