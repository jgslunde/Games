#include <iostream>
#include <vector>
#include <cstdlib>
#include "Tools.h"
#include "Board.h"
#include "Heuristics.h"


inline float board_heuristic_king_free_moves(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    uint64_t blocker_bb = atk_bb | edge_bb;
    float score = 0.0;
    for(int i=1; i<6; i++){
        if((king_bb<<i) & (~blocker_bb)){
            score += 1.0;
        }else{
            break;
        }
    }
    for(int i=1; i<6; i++){
        if((king_bb>>i) & (~blocker_bb)){
            score += 1.0;
        }else{
            break;
        }
    }
    for(int i=1; i<6; i++){
        if((king_bb<<8*i) & (~blocker_bb)){
            score += 1.0;
        }else{
            break;
        }
    }
    for(int i=1; i<6; i++){
        if((king_bb>>8*i) & (~blocker_bb)){
            score += 1.0;
        }else{
            break;
        }
    }
    return score;
}

inline float board_heuristic_king_neighboring_enemies(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  (float) ((king_bb<<1 & atk_bb) != 0)
          + (float) ((king_bb>>1 & atk_bb) != 0)
          + (float) ((king_bb<<8 & atk_bb) != 0)
          + (float) ((king_bb>>8 & atk_bb) != 0);
}

inline float board_heuristic_king_neighboring_allies(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  (float) ((king_bb<<1 & atk_bb) != 0)
          + (float) ((king_bb>>1 & atk_bb) != 0)
          + (float) ((king_bb<<8 & atk_bb) != 0)
          + (float) ((king_bb>>8 & atk_bb) != 0);
}

inline float board_heuristic_attacker_on_edges(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return    (float) ((right_sideedge_bb & atk_bb) != 0)
            + (float) ((left_sideedge_bb & atk_bb) != 0)
            + (float) ((top_sideedge_bb & atk_bb) != 0)
            + (float) ((bottom_sideedge_bb & atk_bb) != 0);
}

inline float board_heuristic_attacker_on_corners(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return 0.1*__builtin_popcountll(diag2corner_bb & atk_bb);
}

inline float board_heuristic_atk_next_to_corners(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    // Standing next to the corners is often a bad idea, as you're easy to capture.
    return - 0.1 * (float) __builtin_popcountll(corner_neighbors_bb & atk_bb);
}
inline float board_heuristic_def_next_to_corners(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    // Standing next to the corners is often a bad idea, as you're easy to capture.
    return 0.1 * (float) __builtin_popcountll(corner_neighbors_bb & def_bb);
}



float combined_board_heuristics(Board &board, HeuristicsConfig *config){
    uint64_t atk_bb = board.atk_bb;
    uint64_t def_bb = board.def_bb;
    uint64_t king_bb = board.king_bb;
    return  1.0*config->atk_pieces_weight * __builtin_popcountll(atk_bb)
            - 0.3*config->def_pieces_weight * __builtin_popcountll(def_bb) // 2.0
            - 0.105*config->king_free_moves_weight * board_heuristic_king_free_moves(atk_bb, def_bb, king_bb)  // 0.05
            + 0.2*config->king_neighboring_enemies_weight * board_heuristic_king_neighboring_enemies(atk_bb, def_bb, king_bb)
            + 0.08*config->king_neighboring_allies_weight * board_heuristic_king_neighboring_allies(atk_bb, def_bb, king_bb)
            + 0.06*config->atk_pieces_on_edges_weight * board_heuristic_attacker_on_edges(atk_bb, def_bb, king_bb)
            + 0.1*config->atk_pieces_diag_to_corners_weight * __builtin_popcountll(diag2corner_bb & atk_bb)
            - 0.15*config->atk_pieces_next_to_corners_weight *  __builtin_popcountll(corner_neighbors_bb & atk_bb)
            + 0.15*config->def_pieces_next_to_corners_weight *  __builtin_popcountll(corner_neighbors_bb & def_bb);
}
