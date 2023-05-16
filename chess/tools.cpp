// g++ -shared -lm -fPIC -o tools.so.1 tools.cpp -O3 -std=c++11 -fopenmp -lpthread
// g++ -shared -lm -fPIC -o tools.so.1 tools.cpp -std=c++11
#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;

int knight_moves[8][2] = {
    {-2,-1},
    {-1,-2},
    { 1,-2},
    {-2, 1},
    { 2,-1},
    {-1, 2},
    { 2, 1},
    { 1, 2}
};

int rook_dirs[4][2] = {
    {-1, 0},
    { 1, 0},
    { 0,-1},
    { 0, 1}
};

string pos2str_array[8] = {"a", "b", "c", "d", "e", "f", "g", "h"};

inline int is_inside_board(int y, int x){
    return (y >= 0)&&(x >= 0)&&(x < 8)&&(y < 8);
}


bool eval_sufficient_material(short *board){
    int bishops_and_knights = 0;
    for(int i=0; i<64; i++){
        if(board[i] == +1)
            return true;
        else if(board[i] == -1)
            return true;
        else if(board[i] == +4)
            return true;
        else if(board[i] == -4)
            return true;
        else if(board[i] == +5)
            return true;
        else if(board[i] == -5)
            return true;
        else if(board[i] == +2)
            bishops_and_knights++;
        else if(board[i] == -2)
            bishops_and_knights++;
        else if(board[i] == -3)
            bishops_and_knights++;
        else if(board[i] == -3)
            bishops_and_knights++;
    }
    if(bishops_and_knights >= 2){
        return true;
    }
    return false;
}


string pos2str(short pos_y, short pos_x){
    return pos2str_array[pos_x] + to_string(8 - pos_y);
}


struct move_struct{
    float score;
    short piece;
    short from_y, from_x, to_y, to_x;
    move_struct *next_move;
};


extern "C"
int get_all_threat_moves(short *board, short *move_list, int player){
    int piece;
    int flat_idx;
    int new_x, new_y;
    int move_list_idx = 0;
    for(int board_y=0; board_y<8; board_y++){
        for(int board_x=0; board_x<8; board_x++){
            flat_idx = board_y*8 + board_x;
            piece = board[flat_idx];
            if(piece*player > 0){

                // Pawn
                if(piece == 1*player){

                    // # Diagonal capture
                    for(int dx=-1; dx<2; dx+=2){
                        new_y = board_y-1*player;
                        new_x = board_x+dx;
                        if(is_inside_board(new_y, new_x)){
                            move_list[move_list_idx++] = piece;
                            move_list[move_list_idx++] = 1;  // Capture
                            move_list[move_list_idx++] = board_y;
                            move_list[move_list_idx++] = board_x;
                            move_list[move_list_idx++] = new_y;
                            move_list[move_list_idx++] = new_x;
                        }
                    }
                }

                // Knight
                else if(piece == 2*player){
                    for(int i_move=0; i_move<8; i_move++){
                        new_y = board_y + knight_moves[i_move][0];
                        new_x = board_x + knight_moves[i_move][1];
                        if(is_inside_board(new_y, new_x)){
                            move_list[move_list_idx++] = piece;
                            move_list[move_list_idx++] = 1;  // Capture
                            move_list[move_list_idx++] = board_y;
                            move_list[move_list_idx++] = board_x;
                            move_list[move_list_idx++] = new_y;
                            move_list[move_list_idx++] = new_x;
                        }
                    }
                }

                // Bishop
                else if(piece == 3*player){
                    for(int dy=-1; dy<2; dy+=2){
                        for(int dx=-1; dx<2; dx+=2){
                            for(int length=1; length<8; length++){
                                new_y = board_y + dx*length;
                                new_x = board_x + dy*length;
                                if(is_inside_board(new_y, new_x)){
                                    move_list[move_list_idx++] = piece;
                                    move_list[move_list_idx++] = 1;  // Capture
                                    move_list[move_list_idx++] = board_y;
                                    move_list[move_list_idx++] = board_x;
                                    move_list[move_list_idx++] = new_y;
                                    move_list[move_list_idx++] = new_x;
                                }
                                else
                                    break;
                                if(board[new_y*8+new_x] != 0)
                                    break;
                            }
                        }
                    }
                }

                // Rook
                else if(piece == 4*player){
                    for(int rook_dir=0; rook_dir<4; rook_dir++){
                        int dx = rook_dirs[rook_dir][0];
                        int dy = rook_dirs[rook_dir][1];
                        for(int length=1; length<8; length++){
                            new_y = board_y + dy*length;
                            new_x = board_x + dx*length;
                            if(is_inside_board(new_y, new_x)){
                                move_list[move_list_idx++] = piece;
                                move_list[move_list_idx++] = 1;  // Capture
                                move_list[move_list_idx++] = board_y;
                                move_list[move_list_idx++] = board_x;
                                move_list[move_list_idx++] = new_y;
                                move_list[move_list_idx++] = new_x;
                                }
                                else
                                    break;
                                if(board[new_y*8+new_x] != 0){
                                    break;
                            }
                        }
                    }
                }

                // Queen
                else if(piece == 5*player){
                    for(int rook_dir=0; rook_dir<4; rook_dir++){
                        int dx = rook_dirs[rook_dir][0];
                        int dy = rook_dirs[rook_dir][1];
                        for(int length=1; length<8; length++){
                            new_y = board_y + dy*length;
                            new_x = board_x + dx*length;
                            if(is_inside_board(new_y, new_x)){
                                move_list[move_list_idx++] = piece;
                                move_list[move_list_idx++] = 1;  // Capture
                                move_list[move_list_idx++] = board_y;
                                move_list[move_list_idx++] = board_x;
                                move_list[move_list_idx++] = new_y;
                                move_list[move_list_idx++] = new_x;
                            }
                            else
                                break;
                            if(board[new_y*8+new_x] != 0)
                                break;
                        }
                    }
                    for(int dy=-1; dy<2; dy+=2){
                        for(int dx=-1; dx<2; dx+=2){
                            for(int length=1; length<8; length++){
                                new_y = board_y + dx*length;
                                new_x = board_x + dy*length;
                                if(is_inside_board(new_y, new_x)){
                                    move_list[move_list_idx++] = piece;
                                    move_list[move_list_idx++] = 1;  // Capture
                                    move_list[move_list_idx++] = board_y;
                                    move_list[move_list_idx++] = board_x;
                                    move_list[move_list_idx++] = new_y;
                                    move_list[move_list_idx++] = new_x;
                                }
                                else
                                    break;
                                if(board[new_y*8+new_x] != 0)
                                    break;
                            }
                        }
                    }
                }

                // King
                else if(piece == 6*player){
                    for(int dx=-1; dx<2; dx++){
                        for(int dy=-1; dy<2; dy++){
                            if(dx != 0 || dy != 0){
                                new_y = board_y + dy;
                                new_x = board_x + dx;
                                if(is_inside_board(new_y, new_x)){
                                    move_list[move_list_idx++] = piece;
                                    move_list[move_list_idx++] = 1;  // Capture
                                    move_list[move_list_idx++] = board_y;
                                    move_list[move_list_idx++] = board_x;
                                    move_list[move_list_idx++] = new_y;
                                    move_list[move_list_idx++] = new_x;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return move_list_idx/6;  // num moves tot.
}


extern "C"
void get_threat_board(bool threat_board[8][8], short *board, int player){
    short move_list[600];
    int num_moves = get_all_threat_moves(board, move_list, player);
    for(int imove=0; imove<num_moves; imove++){
        if(move_list[imove*6+1]){  // If is capture move.
            int new_y = move_list[imove*6+4];
            int new_x = move_list[imove*6+5];
            threat_board[new_y][new_x] = true;
        }
    }
}


extern "C"
int get_all_possible_moves(short *board, short *move_list, int player){
    bool threat_board[8][8] = {0};
    get_threat_board(threat_board, board, -player);

    int piece;
    int flat_idx;
    int new_x, new_y;
    int move_list_idx = 0;
    for(int board_y=0; board_y<8; board_y++){
        for(int board_x=0; board_x<8; board_x++){
            flat_idx = board_y*8 + board_x;
            piece = board[flat_idx];
            if(piece*player > 0){

                // Pawn
                if(piece == 1*player){
                    // 1 step forward
                    new_y = board_y-1*player;
                    new_x = board_x;                    
                    if(is_inside_board(new_y, new_x)){
                        if(board[new_y*8+new_x] == 0){
                            move_list[move_list_idx++] = piece;
                            move_list[move_list_idx++] = 0;  // Capture
                            move_list[move_list_idx++] = board_y;
                            move_list[move_list_idx++] = board_x;
                            move_list[move_list_idx++] = new_y;
                            move_list[move_list_idx++] = new_x;
                        }
                    }

                    // # Diagonal capture
                    for(int dx=-1; dx<2; dx+=2){
                        new_y = board_y-1*player;
                        new_x = board_x+dx;
                        if(is_inside_board(new_y, new_x)){
                            if(board[new_y*8+new_x]*player < 0){
                                move_list[move_list_idx++] = piece;
                                move_list[move_list_idx++] = 1;  // Capture
                                move_list[move_list_idx++] = board_y;
                                move_list[move_list_idx++] = board_x;
                                move_list[move_list_idx++] = new_y;
                                move_list[move_list_idx++] = new_x;
                            }
                        }
                    }

                    // 2 steps forward
                    if(player == 1 && board_y == 6){
                        new_y = board_y-1;
                        new_x = board_x;
                        if(board[new_y*8+new_x] == 0){
                            new_y = board_y-2;
                            new_x = board_x;
                            if(board[new_y*8+new_x] == 0){
                                if(is_inside_board(new_y, new_x)){
                                    move_list[move_list_idx++] = piece;
                                    move_list[move_list_idx++] = 0;  // Capture
                                    move_list[move_list_idx++] = board_y;
                                    move_list[move_list_idx++] = board_x;
                                    move_list[move_list_idx++] = new_y;
                                    move_list[move_list_idx++] = new_x;
                                }
                            }
                        }
                    }
                }

                // Knight
                else if(piece == 2*player){
                    for(int i_move=0; i_move<8; i_move++){
                        new_y = board_y + knight_moves[i_move][0];
                        new_x = board_x + knight_moves[i_move][1];
                        if(is_inside_board(new_y, new_x)){
                            if(board[new_y*8+new_x]*player <= 0){
                                move_list[move_list_idx++] = piece;
                                move_list[move_list_idx++] = 1;  // Capture
                                move_list[move_list_idx++] = board_y;
                                move_list[move_list_idx++] = board_x;
                                move_list[move_list_idx++] = new_y;
                                move_list[move_list_idx++] = new_x;
                            }
                        }
                    }
                }

                // Bishop
                else if(piece == 3*player){
                    for(int dy=-1; dy<2; dy+=2){
                        for(int dx=-1; dx<2; dx+=2){
                            for(int length=1; length<8; length++){
                                new_y = board_y + dx*length;
                                new_x = board_x + dy*length;
                                if(is_inside_board(new_y, new_x)){
                                    if(board[new_y*8+new_x]*player <= 0){
                                        move_list[move_list_idx++] = piece;
                                        move_list[move_list_idx++] = 1;  // Capture
                                        move_list[move_list_idx++] = board_y;
                                        move_list[move_list_idx++] = board_x;
                                        move_list[move_list_idx++] = new_y;
                                        move_list[move_list_idx++] = new_x;
                                    }
                                }
                                else
                                    break;
                                if(board[new_y*8+new_x] != 0)
                                    break;
                            }
                        }
                    }
                }

                // Rook
                else if(piece == 4*player){
                    for(int rook_dir=0; rook_dir<4; rook_dir++){
                        int dx = rook_dirs[rook_dir][0];
                        int dy = rook_dirs[rook_dir][1];
                        for(int length=1; length<8; length++){
                            new_y = board_y + dy*length;
                            new_x = board_x + dx*length;
                            if(is_inside_board(new_y, new_x)){
                                if(board[new_y*8+new_x]*player <= 0){
                                    move_list[move_list_idx++] = piece;
                                    move_list[move_list_idx++] = 1;  // Capture
                                    move_list[move_list_idx++] = board_y;
                                    move_list[move_list_idx++] = board_x;
                                    move_list[move_list_idx++] = new_y;
                                    move_list[move_list_idx++] = new_x;
                                }
                            }
                            else
                                break;
                            if(board[new_y*8+new_x] != 0){
                                break;
                            }
                        }
                    }
                }

                // Queen
                else if(piece == 5*player){
                    for(int rook_dir=0; rook_dir<4; rook_dir++){
                        int dx = rook_dirs[rook_dir][0];
                        int dy = rook_dirs[rook_dir][1];
                        for(int length=1; length<8; length++){
                            new_y = board_y + dy*length;
                            new_x = board_x + dx*length;
                            if(is_inside_board(new_y, new_x)){
                                if(board[new_y*8+new_x]*player <= 0){
                                    move_list[move_list_idx++] = piece;
                                    move_list[move_list_idx++] = 1;  // Capture
                                    move_list[move_list_idx++] = board_y;
                                    move_list[move_list_idx++] = board_x;
                                    move_list[move_list_idx++] = new_y;
                                    move_list[move_list_idx++] = new_x;
                                }
                            }
                            else
                                break;
                            if(board[new_y*8+new_x] != 0)
                                break;
                        }
                    }
                    for(int dy=-1; dy<2; dy+=2){
                        for(int dx=-1; dx<2; dx+=2){
                            for(int length=1; length<8; length++){
                                new_y = board_y + dx*length;
                                new_x = board_x + dy*length;
                                if(is_inside_board(new_y, new_x)){
                                    if(board[new_y*8+new_x]*player <= 0){
                                        move_list[move_list_idx++] = piece;
                                        move_list[move_list_idx++] = 1;  // Capture
                                        move_list[move_list_idx++] = board_y;
                                        move_list[move_list_idx++] = board_x;
                                        move_list[move_list_idx++] = new_y;
                                        move_list[move_list_idx++] = new_x;
                                    }
                                }
                                else
                                    break;
                                if(board[new_y*8+new_x] != 0)
                                    break;
                            }
                        }
                    }
                }

                // King
                else if(piece == 6*player){
                    for(int dx=-1; dx<2; dx++){
                        for(int dy=-1; dy<2; dy++){
                            if(dx != 0 || dy != 0){
                                new_y = board_y + dy;
                                new_x = board_x + dx;
                                if(is_inside_board(new_y, new_x)){
                                    if(board[new_y*8+new_x]*player <= 0){
                                        if(!threat_board[new_y][new_x]){
                                            move_list[move_list_idx++] = piece;
                                            move_list[move_list_idx++] = 1;  // Capture
                                            move_list[move_list_idx++] = board_y;
                                            move_list[move_list_idx++] = board_x;
                                            move_list[move_list_idx++] = new_y;
                                            move_list[move_list_idx++] = new_x;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return move_list_idx/6;
}

extern "C"
int get_all_legal_moves(short *board, short *legal_move_list, int player){
    short move_list[600] = {0};
    int num_moves = get_all_possible_moves(board, move_list, player);

    short tmp_board[64] = {0};
    bool tmp_threat_board[8][8] = {0};
    int legal_move_list_idx = 0;
    for(int imove=0; imove<num_moves; imove++){
        int piece = move_list[imove*6+0];
        int capture = move_list[imove*6+1];
        int board_y = move_list[imove*6+2];
        int board_x = move_list[imove*6+3];
        int new_y = move_list[imove*6+4];
        int new_x = move_list[imove*6+5];
        if((piece == 1*player)&&(board_x != new_x)){  // pawn diagonal capture
            if(board[new_y*8+new_x]*player >= 0){
                continue;  // If no enemy piece, diagonal move is illegal.
            }
        }

        if(board[new_y*8+new_x]*player <= 0){  // Moves which end up at enemy or empty spot.
            for(int i=0; i<64; i++){
                tmp_board[i] = board[i];
            }
            tmp_board[new_y*8+new_x] = board[board_y*8+board_x];
            tmp_board[board_y*8+board_x] = 0;

            for(int x=0; x<8; x++)
                for(int y=0; y<8; y++)
                    tmp_threat_board[y][x] = 0;
            get_threat_board(tmp_threat_board, tmp_board, -player);
            bool threat_on_king = false;
            for(int kx=0; kx<8; kx++){
                for(int ky=0; ky<8; ky++){
                    if((tmp_board[ky*8+kx]==6*player)&&(tmp_threat_board[ky][kx])){
                        threat_on_king = true;
                        // cout << kx << " " << ky << endl;
                    }                    
                }
            }

            if(!threat_on_king){
                legal_move_list[legal_move_list_idx++] = piece;
                legal_move_list[legal_move_list_idx++] = capture;
                legal_move_list[legal_move_list_idx++] = board_y;
                legal_move_list[legal_move_list_idx++] = board_x;
                legal_move_list[legal_move_list_idx++] = new_y;
                legal_move_list[legal_move_list_idx++] = new_x;
            }
        }
    }
    return legal_move_list_idx/6;  // num moves tot.
}


float get_board_score(short *board){
    float score = 0.0;
    int piece;
    for(int i=0; i<64; i++){
        piece = board[i];
        if((piece == +2) || (piece == -2)){
            score += 1.5*piece;
        }
        else if((piece == +4) || (piece == -4)){
            score += 1.25*piece;
        }
        else if((piece == +5) || (piece == -5)){
            score += 1.8*piece;
        }
        else if((piece == +6) || (piece == -6)){
            score += 100*piece;
        }
        else{
            score += piece;
        }
    }
    return score;
}


extern "C"
move_struct eval_move_score(short *board, int player, int depth, int max_depth){
    short legal_moves[600] = {0};
    short tmp_board[64] = {0};
    int num_legal_moves = get_all_legal_moves(board, legal_moves, player);
    move_struct move_scores[num_legal_moves];
    if(num_legal_moves == 0){
        move_struct current_move;
        current_move.score = -9999.0*player;
        return current_move;
    }
    // cout << "depth: " << depth << endl;
    for(int imove=0; imove<num_legal_moves; imove++){
        for(int i=0; i<64; i++){
            tmp_board[i] = board[i];
        }

        int board_y = legal_moves[imove*6+2];
        int board_x = legal_moves[imove*6+3];
        int new_y = legal_moves[imove*6+4];
        int new_x = legal_moves[imove*6+5];
        move_struct current_move;
        current_move.piece  = legal_moves[imove*6+0];
        current_move.from_y = legal_moves[imove*6+2];
        current_move.from_x = legal_moves[imove*6+3];
        current_move.to_y   = legal_moves[imove*6+4];
        current_move.to_x   = legal_moves[imove*6+5];
        // move_scores[imove] =

        tmp_board[new_y*8+new_x] = tmp_board[board_y*8+board_x];
        tmp_board[board_y*8+board_x] = 0;

        if(depth == max_depth){
            move_scores[imove].score = get_board_score(tmp_board);
            // for(int i=0; i<depth-1; i++) cout << "  ";
            // cout << "depth " << depth << ": player " << player << ": " << pos2str(board_y, board_x) << " " << pos2str(new_y, new_x) << " " << move_scores[imove] << endl;
        }
        else{
            // for(int i=0; i<depth-1; i++) cout << "  ";
            // cout << "depth " << depth << ": player " << player << ": " << pos2str(board_y, board_x) << " " << pos2str(new_y, new_x) << endl;
            if(eval_sufficient_material(tmp_board)){
                move_scores[imove] = eval_move_score(tmp_board, -player, depth+1, max_depth);
            }
            else{
                move_struct current_move;
                current_move.score = -9999.0*player;
                move_scores[imove] = current_move;
            }
        }
    }
    int best_move_idx = 0;
    if(player == 1){
        for(int imove=1; imove<num_legal_moves; imove++){
            // cout << depth << " " << move_scores[imove] << " " << best_score << endl;
            if(move_scores[imove].score > move_scores[best_move_idx].score){
                best_move_idx = imove;
            }
        }
    }
    else{
        for(int imove=1; imove<num_legal_moves; imove++){
            if(move_scores[imove].score < move_scores[best_move_idx].score){
                best_move_idx = imove;
            }
        }
    }
    float best_score = move_scores[best_move_idx].score;

    // best_moves[(depth-1)*6+0] = best_score;
    // best_moves[(depth-1)*6+1] = legal_moves[best_move_idx*6+0];
    // best_moves[(depth-1)*6+2] = legal_moves[best_move_idx*6+2];
    // best_moves[(depth-1)*6+3] = legal_moves[best_move_idx*6+3];
    // best_moves[(depth-1)*6+4] = legal_moves[best_move_idx*6+4];
    // best_moves[(depth-1)*6+5] = legal_moves[best_move_idx*6+5];
    // cout << player << " " << best_score << " " << legal_moves[best_move_idx*6+2] << " " << legal_moves[best_move_idx*6+3] << " " << legal_moves[best_move_idx*6+4] << " " << legal_moves[best_move_idx*6+5] << endl;
    return current_move;
}

extern "C"
void eval_move_score_wrapper(short *best_moves, short *board, int player, int depth, int max_depth){
    
}