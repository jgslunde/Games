#include <iostream>
#include <vector>
#include <cstdlib>
#include "Tools.h"
#include "Board.h"
#include "Heuristics.h"


inline float board_heuristic_king_free_moves(Board &board){
    uint64_t blocker_bb = board.atk_bb | edge_bb;
    float score = 0.0;
    for(int i=1; i<6; i++){
        if((board.king_bb<<i) & (~blocker_bb)){
            score += 1.0;
        }else{
            break;
        }
    }
    for(int i=1; i<6; i++){
        if((board.king_bb>>i) & (~blocker_bb)){
            score += 1.0;
        }else{
            break;
        }
    }
    for(int i=1; i<6; i++){
        if((board.king_bb<<8*i) & (~blocker_bb)){
            score += 1.0;
        }else{
            break;
        }
    }
    for(int i=1; i<6; i++){
        if((board.king_bb>>8*i) & (~blocker_bb)){
            score += 1.0;
        }else{
            break;
        }
    }
    return score;
}

inline float board_heuristic_king_neighboring_enemies(Board &board){
    return  (float) ((board.king_bb<<1 & board.atk_bb) != 0)
          + (float) ((board.king_bb>>1 & board.atk_bb) != 0)
          + (float) ((board.king_bb<<8 & board.atk_bb) != 0)
          + (float) ((board.king_bb>>8 & board.atk_bb) != 0);
}

inline float board_heuristic_king_neighboring_allies(Board &board){
    return  (float) ((board.king_bb<<1 & board.atk_bb) != 0)
          + (float) ((board.king_bb>>1 & board.atk_bb) != 0)
          + (float) ((board.king_bb<<8 & board.atk_bb) != 0)
          + (float) ((board.king_bb>>8 & board.atk_bb) != 0);
}

inline float board_heuristic_attacker_on_edges(Board &board){
    return    (float) ((right_smalledge_bb & board.atk_bb) != 0)
            + (float) ((left_smalledge_bb & board.atk_bb) != 0)
            + (float) ((top_smalledge_bb & board.atk_bb) != 0)
            + (float) ((bottom_smalledge_bb & board.atk_bb) != 0);
}

inline float board_heuristic_atk_diag_to_corners(Board &board){
    return __builtin_popcountll(diag2corner_bb & board.atk_bb);
}

inline float board_heuristic_atk_next_to_corners(Board &board){
    // Standing next to the corners is often a bad idea, as you're easy to capture.
    return (float) __builtin_popcountll(corner_neighbors_bb & board.atk_bb);
}
inline float board_heuristic_def_next_to_corners(Board &board){
    // Standing next to the corners is often a bad idea, as you're easy to capture.
    return (float) __builtin_popcountll(corner_neighbors_bb & board.def_bb);
}

inline float board_heuristic_attacker_pieces(Board &board){
    return __builtin_popcountll(board.atk_bb);
}

inline float board_heuristic_defender_pieces(Board &board){
    return __builtin_popcountll(board.def_bb);
}

// inline float board_heuristic_king_on_open_edge(Board &board){
//     return (float) (board.king_bb & right_smalledge) && 
// }


float combined_board_heuristics(Board &board, HeuristicsConfig *config){
    return (float)  config->atk_pieces_weight                 * board_heuristic_attacker_pieces(board)
                  + config->def_pieces_weight                 * board_heuristic_defender_pieces(board)
                  + config->king_free_moves_weight            * board_heuristic_king_free_moves(board)
                  + config->king_neighboring_enemies_weight   * board_heuristic_king_neighboring_enemies(board)
                  + config->king_neighboring_allies_weight    * board_heuristic_king_neighboring_allies(board)
                  + config->atk_pieces_on_edges_weight        * board_heuristic_attacker_on_edges(board)
                  + config->atk_pieces_diag_to_corners_weight * board_heuristic_atk_diag_to_corners(board)
                  + config->atk_pieces_next_to_corners_weight * board_heuristic_atk_next_to_corners(board)
                  + config->def_pieces_next_to_corners_weight * board_heuristic_def_next_to_corners(board);
}
