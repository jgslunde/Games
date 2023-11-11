#pragma once

#include <cstdint>
#include <vector>

using namespace std;

struct Move{
    uint64_t move_from;
    uint64_t move_to;
    float eval;
};


class Board{
    public:
        uint64_t atk_bb, def_bb, king_bb;
        unsigned short turn;
        int current_player;
        void reset_board();
        vector<uint64_t> get_all_legal_moves();
        void make_move(uint64_t &move_from, uint64_t &move_to);
        uint64_t perform_captures(uint64_t moved_piece_bb, uint64_t allied_pieces_bb, uint64_t enemy_pices_bb);
        int get_current_player();
        float get_board_wins();
};