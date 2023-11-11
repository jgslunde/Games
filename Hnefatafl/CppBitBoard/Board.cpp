#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <random>
#include <thread>
#include <map>
#include <random>
#include "Tools.h"
#include "Board.h"
#include "Heuristics.h"
#include "AI.h"


using namespace std;
using cuchar = const unsigned char;
using uchar = unsigned char;
using ushort = unsigned short;



void Board::reset_board(){
    atk_bb = 0x8080063000808;
    def_bb = 0x814080000;
    king_bb = 0x8000000;
    turn = 1;
    current_player = 1;  // Attackers turn.
}


vector<uint64_t> Board::get_all_legal_moves(){
    vector<uint64_t> legal_moves;
    legal_moves.reserve(100);
    uint64_t piece_bb;
    if(current_player == PLAYER_ATK){
        // Attacker moves.
        for(int imove=0; imove<64; imove++){
            piece_bb = 1ULL << imove;
            if(atk_bb & piece_bb){
                vector<uint64_t> legal_moves_piece = get_legal_moves_as_vector(piece_bb, atk_bb | def_bb | king_bb | edge_bb | corner_bb);
                for(int i=0; i<legal_moves_piece.size(); i++){
                    legal_moves.push_back(piece_bb);
                    legal_moves.push_back(legal_moves_piece[i]);
                }
            }
        }
    }
    else{
        for(int imove=0; imove<64; imove++){
            // Defender moves.
            piece_bb = 1ULL << imove;
            if(def_bb & piece_bb){
                vector<uint64_t> legal_moves_piece = get_legal_moves_as_vector(piece_bb, atk_bb | def_bb | king_bb | edge_bb | corner_bb);
                for(int i=0; i<legal_moves_piece.size(); i++){
                    legal_moves.push_back(piece_bb);
                    legal_moves.push_back(legal_moves_piece[i]);
                }
            }
        }
        for(int imove=0; imove<64; imove++){
            // King (defender) moves.
            piece_bb = 1ULL << imove;
            if(king_bb & piece_bb){
                vector<uint64_t> legal_moves_piece = get_legal_moves_as_vector(piece_bb, atk_bb | def_bb | edge_bb);
                for(int i=0; i<legal_moves_piece.size(); i++){
                    legal_moves.push_back(piece_bb);
                    legal_moves.push_back(legal_moves_piece[i]);
                }
            }
        }
    }
    return legal_moves;
}


void Board::make_move(uint64_t &move_from, uint64_t &move_to){
    // 95 instructions, including calls to "perform_captures".
    uint64_t captures;
    if(atk_bb & move_from){
        atk_bb ^= (move_from | move_to);
        captures = perform_captures(move_to, atk_bb | corner_bb, def_bb | king_bb);
        def_bb &= ~captures;
        king_bb &= ~captures;
    }
    else if(def_bb & move_from){
        def_bb ^= (move_from | move_to);
        captures = perform_captures(move_to, def_bb | king_bb | corner_bb, atk_bb);
        atk_bb &= ~captures;
    }
    else if(king_bb & move_from){
        king_bb ^= (move_from | move_to);
        captures = perform_captures(move_to, def_bb | king_bb | corner_bb, atk_bb);
        atk_bb &= ~captures;
    }
    current_player *= -1;
}


uint64_t Board::perform_captures(uint64_t moved_piece_bb, uint64_t allied_pieces_bb, uint64_t enemy_pices_bb){
    // 34 instructions.
    uint64_t potential_capture, captured;
    captured = 0;

    // Right capture
    potential_capture = ((moved_piece_bb << 1) & enemy_pices_bb);
    captured |= potential_capture*(((potential_capture << 1) & allied_pieces_bb) != 0);

    // Left capture
    potential_capture = ((moved_piece_bb >> 1) & enemy_pices_bb);
    captured |= potential_capture*(((potential_capture >> 1) & allied_pieces_bb) != 0);

    // Down capture
    potential_capture = ((moved_piece_bb << 8) & enemy_pices_bb);
    captured |= potential_capture*(((potential_capture << 8) & allied_pieces_bb) != 0);

    // Up capture
    potential_capture = ((moved_piece_bb >> 8) & enemy_pices_bb);
    captured |= potential_capture*(((potential_capture >> 8) & allied_pieces_bb) != 0);

    return captured;
}


int Board::get_current_player(){
    return current_player;
}


float Board::get_board_wins(){
    uint64_t king_hostile_bb = atk_bb | corner_bb;
    float score = 
            - 1000.0*(__builtin_popcountll(atk_bb) == 0)   // Defender wins if attacker is out of pieces.
            - 1000.0*((king_bb & corner_bb) > 0)   // Defender wins if the king reaces a corner.
            + 1000.0*(!king_bb);   // Attacker wins if king is captured, and therefore not on the board.
    return score;
}

