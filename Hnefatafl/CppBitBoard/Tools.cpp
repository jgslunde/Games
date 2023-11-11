#include <iostream>
#include <vector>
#include <cstdlib>
#include <random>
#include <thread>
#include "Tools.h"


using namespace std;

thread_local mt19937 generator(hash<thread::id>{}(this_thread::get_id()));


int thread_safe_rand() {
    return generator();
}

uint64_t board2bits(vector<vector<uint64_t>> board){
    uint64_t bb = 0;
    for(uint64_t y=0; y<8; y++){
        for(uint64_t x=0; x<8; x++){
            if((board[y][x] != 0) && (board[y][x] != 1)) throw invalid_argument("bitboard converter recieved non-1 value.");
            uint64_t idx = 8*y + x;
            bb += board[y][x] << (idx);
        }
    }
    return bb;
}


vector<vector<uint64_t>>bits2board(uint64_t bb){
    vector<vector<uint64_t>> board(8, vector<uint64_t>(8));
    for(uint64_t y=0; y<8; y++){
        for(uint64_t x=0; x<8; x++){
            uint64_t idx = 8*y + x;
            board[y][x] = (bb >> idx) & 1;
        }
    }
    return board;
}


void print_board(vector<vector<uint64_t>> board){
    cout << endl;
    for(int y=0; y<8; y++){
        for (int x=0; x<8; x++){
            if(x<7 && y<7)
                cout << board[y][x] << " ";
            else
                cout << "\033[0;90m" << board[y][x] << "\033[0m ";
        }
        cout << endl;
    }
    cout << endl;
}


void print_board_move(vector<vector<uint64_t>> board1, vector<vector<uint64_t>> board2){
    cout << endl;
    for(int y=0; y<8; y++){
        for (int x=0; x<8; x++){
            if(board1[y][x]){
                cout << "\033[32m1\033[0m ";
            }
            else if(board2[y][x]){
                cout << "\033[31m1\033[0m ";
            }
            else{
                cout << "0 ";
            }
        }
        cout << endl;
    }
    cout << endl;
}


void print_bitboard(uint64_t bb){
    vector<vector<uint64_t>> board = bits2board(bb);
    print_board(board);
}


void print_bitboard_move(uint64_t bb1, uint64_t bb2){
    vector<vector<uint64_t>> board1 = bits2board(bb1);
    vector<vector<uint64_t>> board2 = bits2board(bb2);
    print_board_move(board1, board2);
}


void print_bitgame(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    vector<vector<uint64_t>> atk = bits2board(atk_bb);
    vector<vector<uint64_t>> def = bits2board(def_bb);
    vector<vector<uint64_t>> king = bits2board(king_bb);

    cout << endl;
    for(int y=0; y<8; y++){
        for (int x=0; x<8; x++){
            if(atk[y][x]){
                cout << "○ ";
            }
            else if(def[y][x]){
                cout << "● ";
            }
            else if(king[y][x]){
                cout << "♚ ";
            }
            else if((y==0 || y==6) && (x==0 || x==6)){
                cout << "· ";
            }
            else if(x==3 && y==3 && ~king[y][x]){
                cout << "· ";
            }
            else{
                cout << "  ";
            }
        }
        cout << endl;
    }
        cout << endl;
}


vector<uint64_t> get_legal_moves_as_vector(uint64_t piece_bb, uint64_t blocker_bb){
    vector<uint64_t> legal_moves;
    legal_moves.reserve(12);
    uint64_t proposed_move;

    // RIGHT
    for(uint64_t imove=1; imove<7; imove++){
        proposed_move = piece_bb << imove;
        if(proposed_move & blocker_bb){
            break;
        }
        if((proposed_move != 0) && proposed_move){
            legal_moves.push_back(proposed_move);
        }
    }
    // LEFT
    for(uint64_t imove=1; imove<7; imove++){
        proposed_move = piece_bb >> imove;
        if(proposed_move & blocker_bb){
            break;
        }
        if((proposed_move != 0) && proposed_move){
            legal_moves.push_back(proposed_move);
        }
    }
    // UP
        for(uint64_t imove=1; imove<7; imove++){
        proposed_move = piece_bb << 8*imove;
        if(proposed_move & blocker_bb){
            break;
        }
        if((proposed_move != 0) && proposed_move){
            legal_moves.push_back(proposed_move);
        }
    }
    // DOWN
    for(uint64_t imove=1; imove<7; imove++){
        proposed_move = piece_bb >> 8*imove;
        if(proposed_move & blocker_bb){
            break;
        }
        if((proposed_move != 0) && proposed_move){
            legal_moves.push_back(proposed_move);
        }
    }
    return legal_moves;
}