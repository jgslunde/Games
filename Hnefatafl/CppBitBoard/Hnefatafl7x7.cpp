#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace std;


const int EMPTY = 0;
const int ATTACKER = 1;
const int DEFENDER = 2;
const int KING = 3;

const int PLAYER_ATK = 1;
const int PLAYER_DEF = -1;

const int UP = 0;
const int RIGHT = 1;
const int DOWN = 2;
const int LEFT = 3;

const uint64_t corner_bb = 18295873486192705ull;
const uint64_t throne_bb = 134217728ull;
const uint64_t edge_bb = 18410856566090662016ull;

vector<int> NUM_NODES(10);


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
            cout << board[y][x] << " ";
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


uint64_t get_legal_moves_as_bb(uint64_t piece_bb, uint64_t blocker_bb, vector<vector<uint64_t>> &move_lookup){
    // 60 instructions.
    uint64_t allowed_moves = 0;
    uint64_t possible_moves, masked_blocker_bb;
    int idx = __builtin_ctzll(piece_bb);

    possible_moves = move_lookup[idx][RIGHT];
    masked_blocker_bb = possible_moves & blocker_bb;
    allowed_moves |= possible_moves;
    if(masked_blocker_bb){
        int first_blocking_idx = __builtin_ctzll(masked_blocker_bb) - 1;
        allowed_moves &= ~(move_lookup[first_blocking_idx][RIGHT]);
    }

    possible_moves = move_lookup[idx][DOWN];
    masked_blocker_bb = possible_moves & blocker_bb;
    allowed_moves |= possible_moves;
    if(masked_blocker_bb){
        int first_blocking_idx = __builtin_ctzll(masked_blocker_bb) - 8;
        allowed_moves &= ~(move_lookup[first_blocking_idx][DOWN]);
    }

    possible_moves = move_lookup[idx][LEFT];
    masked_blocker_bb = possible_moves & blocker_bb;
    allowed_moves |= possible_moves;
    if(masked_blocker_bb){
        int first_blocking_idx = 63 - __builtin_clzll(masked_blocker_bb) + 1;
        uint64_t asdf = 1;
        asdf = asdf << first_blocking_idx - 1;
        allowed_moves &= ~(move_lookup[first_blocking_idx][LEFT]);
    }

    possible_moves = move_lookup[idx][UP];
    masked_blocker_bb = possible_moves & blocker_bb;
    allowed_moves |= possible_moves;
    if(masked_blocker_bb){
        int first_blocking_idx = 63 - __builtin_clzll(masked_blocker_bb) + 8;
        allowed_moves &= ~(move_lookup[first_blocking_idx][UP]);
    }

    return allowed_moves;
}


vector<uint64_t> get_legal_moves_as_vector(uint64_t piece_bb, uint64_t blocker_bb){
    // 165 instructions??
    vector<uint64_t> legal_moves;
    legal_moves.reserve(12);
    uint64_t proposed_move;

    // RIGHT
    for(uint64_t imove=1; imove<7; imove++){
        proposed_move = piece_bb << imove;
        if(proposed_move & blocker_bb){
            break;
        }
        if((proposed_move != 0) && proposed_move != throne_bb){
            // legal_moves.push_back(piece_bb);
            legal_moves.push_back(proposed_move);
        }
    }
    // LEFT
    for(uint64_t imove=1; imove<7; imove++){
        proposed_move = piece_bb >> imove;
        if(proposed_move & blocker_bb){
            break;
        }
        if((proposed_move != 0) && proposed_move != throne_bb){
            // legal_moves.push_back(piece_bb);
            legal_moves.push_back(proposed_move);
        }
    }
    // UP
        for(uint64_t imove=1; imove<7; imove++){
        proposed_move = piece_bb << 8*imove;
        if(proposed_move & blocker_bb){
            break;
        }
        if((proposed_move != 0) && proposed_move != throne_bb){
            // legal_moves.push_back(piece_bb);
            legal_moves.push_back(proposed_move);
        }
    }
    // DOWN
    for(uint64_t imove=1; imove<7; imove++){
        proposed_move = piece_bb >> 8*imove;
        if(proposed_move & blocker_bb){
            break;
        }
        if((proposed_move != 0) && proposed_move != throne_bb){
            // legal_moves.push_back(piece_bb);
            legal_moves.push_back(proposed_move);
        }
    }
    return legal_moves;
}


int get_board_score(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    // 106 instructions.
    uint64_t king_hostile_bb = atk_bb | corner_bb;
    int score = 
            __builtin_popcountll(atk_bb)
            - 2*__builtin_popcountll(def_bb)
            - 1000*(__builtin_popcountll(atk_bb) == 0)
            - 1000*((king_bb & corner_bb) > 0);
            if(king_bb & throne_bb){  // If king on throne, we need 4 capturing pieces.
                if((king_bb>>1 & atk_bb) && (king_bb<<1 & atk_bb) && (king_bb>>8 & atk_bb) && (king_bb<<8 & atk_bb))
                score += 1000;
            }
            else if((king_bb << 1 & throne_bb) || (king_bb >> 1 & throne_bb) || (king_bb << 8 & throne_bb) || (king_bb >> 8 & throne_bb)){
                // If king is next to throne, we need 3 capturing pieces.
                int enemy_right = ((king_bb>>1 & atk_bb) != 0);
                int enemy_left = ((king_bb<<1 & atk_bb) != 0);
                int enemy_down = ((king_bb>>8 & atk_bb) != 0);
                int enemy_up = ((king_bb<<8 & atk_bb) != 0);
                if(enemy_right+enemy_left+enemy_down+enemy_up >= 3){
                    score += 1000;
                }
            }
            else{
                if(((king_bb>>1 & king_hostile_bb) && (king_bb<<1 & king_hostile_bb)) || ((king_bb>>8 & king_hostile_bb) && (king_bb<<8 & king_hostile_bb))){
                    score += 1000;
                }
            }

    return score;
}


uint64_t perform_captures(uint64_t moved_piece_bb, uint64_t allied_pieces_bb, uint64_t enemy_pices_bb){
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


void make_move_on_board(uint64_t &atk_bb, uint64_t &def_bb, uint64_t &king_bb, uint64_t move_from, uint64_t move_to){
    // 95 instructions, including calls to "perform_captures".
    if(atk_bb & move_from){
        atk_bb ^= (move_from | move_to);
        uint64_t captures = perform_captures(move_to, atk_bb | corner_bb, def_bb);
        def_bb &= ~captures;
    }
    else if(def_bb & move_from){
        def_bb ^= (move_from | move_to);
        uint64_t captures = perform_captures(move_to, def_bb | king_bb | corner_bb, atk_bb);
        atk_bb &= ~captures;
    }
    else if(king_bb & move_from){
        king_bb ^= (move_from | move_to);
        uint64_t captures = perform_captures(move_to, def_bb | king_bb | corner_bb, atk_bb);
        atk_bb &= ~captures;
    }
}


vector<uint64_t> get_all_legal_moves_as_vector(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player){
    // 284 instructions, including calls to "get_legal_moves_as_vector".
    vector<uint64_t> legal_moves;
    legal_moves.reserve(100);
    uint64_t piece_bb;
    if(player == PLAYER_ATK){
        // Attacker moves.
        // int x = 0;
        for(int imove=0; imove<64; imove++){
            piece_bb = (uint64_t) 1 << imove;
            if(atk_bb & piece_bb){
                // cout << imove << endl;
                // print_bitboard(piece_bb);
                vector<uint64_t> legal_moves_piece = get_legal_moves_as_vector(piece_bb, atk_bb | def_bb | king_bb | edge_bb | corner_bb);
                for(int i=0; i<legal_moves_piece.size(); i++){
                    // cout << x << endl;
                    // x ++;
                    // print_bitboard_move(piece_bb, legal_moves_piece[i]);
                    // cout << "size before: " << legal_moves.size();
                    legal_moves.push_back(piece_bb);
                    // cout << "  size after1: " << legal_moves.size();
                    legal_moves.push_back(legal_moves_piece[i]);
                    // cout << "  size after2: " << legal_moves.size() << endl;;
                }
            }
        }
        // cout << legal_moves.size() << endl;
    }
    else{
        for(int imove=0; imove<64; imove++){
            // Defender moves.
            piece_bb = (uint64_t) 1 << imove;
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
            piece_bb = (uint64_t) 1 << imove;
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


int get_board_score_by_width_search(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, int depth, int max_depth){
    NUM_NODES[depth] += 1;
    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size()/2;
    vector<int> move_scores(num_legal_moves);
    for(int imove=0; imove<num_legal_moves; imove++){
        uint64_t atk_bb_new = atk_bb;
        uint64_t def_bb_new = def_bb;
        uint64_t king_bb_new = king_bb;
        make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*imove], legal_moves[2*imove+1]);
        int move_score = get_board_score(atk_bb_new, def_bb_new, king_bb_new);
        if((depth == max_depth) || abs(move_score) > 900){
            move_scores[imove] = move_score;
        }
        else{
            move_scores[imove] = get_board_score_by_width_search(atk_bb_new, def_bb_new, king_bb_new, -player, depth+1, max_depth);
        }
    }
    // int best_board_score;
    if(player == PLAYER_ATK){
        int best_board_score = -999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] > best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
    else{
        int best_board_score = 999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] < best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
}

// vector<int> get_move_scores(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, int max_depth){
// }




void compare_bbs_to_expected(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, uint64_t atk_bb_exp, uint64_t def_bb_exp, uint64_t king_bb_exp){
    if(atk_bb ^ atk_bb_exp){
        cout << "Expected atk:" << endl;
        print_bitboard(atk_bb_exp);
        cout << "Got atk:" << endl;
        print_bitboard(atk_bb);
        throw invalid_argument("ATK BB WRONG!");
    }
    if(def_bb ^ def_bb_exp){
        cout << "Expected atk:" << endl;
        print_bitboard(def_bb_exp);
        cout << "Got def:" << endl;
        print_bitboard(def_bb);
        throw invalid_argument("def BB WRONG!");
    }
    if(king_bb ^ king_bb_exp){
        cout << "Expected king:" << endl;
        print_bitboard(king_bb_exp);
        cout << "Got king:" << endl;
        print_bitboard(king_bb);
        throw invalid_argument("king BB WRONG!");
    }
}





int main(){
    vector<vector<uint64_t>> move_lookup(64, vector<uint64_t>(4));

    vector<vector<uint64_t>> board(8, vector<uint64_t>(8));
    for(int y=0; y<7; y++){
        for(int x=0; x<7; x++){
            // UP
            for(int y_loc=0; y_loc<8; y_loc++){
                for(int x_loc=0; x_loc<8; x_loc++){
                    board[y_loc][x_loc] = 0;
                }
            }
            for(int y_loc=y-1; y_loc>=0; y_loc--){
                board[y_loc][x] = 1;                
            }
            move_lookup[y*8 + x][UP] = board2bits(board);

            // RIGHT
            for(int y_loc=0; y_loc<8; y_loc++){
                for(int x_loc=0; x_loc<8; x_loc++){
                    board[y_loc][x_loc] = 0;
                }
            }
            for(int x_loc=x+1; x_loc<=6; x_loc++){
                board[y][x_loc] = 1;                
            }
            move_lookup[y*8 + x][RIGHT] = board2bits(board);

            // DOWN
            for(int y_loc=0; y_loc<8; y_loc++){
                for(int x_loc=0; x_loc<8; x_loc++){
                    board[y_loc][x_loc] = 0;
                }
            }
            for(int y_loc=y+1; y_loc<=6; y_loc++){
                board[y_loc][x] = 1;                
            }
            move_lookup[y*8 + x][DOWN] = board2bits(board);

            // LEFT
            for(int y_loc=0; y_loc<8; y_loc++){
                for(int x_loc=0; x_loc<8; x_loc++){
                    board[y_loc][x_loc] = 0;
                }
            }
            for(int x_loc=x-1; x_loc>=0; x_loc--){
                board[y][x_loc] = 1;
            }
            move_lookup[y*8 + x][LEFT] = board2bits(board);

        }
    }

    vector<vector<uint64_t>> initial_atk_board = {
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 1, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    vector<vector<uint64_t>> initial_def_board = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 1, 0, 1, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    vector<vector<uint64_t>> initial_king_board = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };


    // vector<vector<uint64_t>> initial_board = {
    //     {0, 0, 0, 1, 0, 0, 0, 0},
    //     {0, 0, 0, 1, 0, 0, 0, 0},
    //     {0, 0, 0, 2, 0, 0, 0, 0},
    //     {1, 1, 2, 3, 2, 1, 1, 0},
    //     {0, 0, 0, 2, 0, 0, 0, 0},
    //     {0, 0, 0, 1, 0, 0, 0, 0},
    //     {0, 0, 0, 1, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0}
    // };


    // vector<vector<uint64_t>> test_piece_board = {
    //     {1, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0}
    // };

    // vector<vector<uint64_t>> test_blocker_board = {
    //     {0, 0, 0, 1, 0, 0, 1, 0},
    //     {0, 0, 0, 0, 0, 1, 0, 0},
    //     {0, 0, 1, 1, 0, 0, 1, 0},
    //     {0, 1, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 1, 0, 1, 0, 0, 0},
    //     {1, 0, 0, 1, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0}
    // };

    vector<vector<uint64_t>> test_atk_board = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    vector<vector<uint64_t>> test_def_board = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    vector<vector<uint64_t>> test_king_board = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };


    // uint64_t test_blocker_bb = board2bits(test_blocker_board);
    // uint64_t test_piece_bb = board2bits(test_piece_board);

    uint64_t test_atk_bb = board2bits(test_atk_board);
    uint64_t test_def_bb = board2bits(test_def_board);
    uint64_t test_king_bb = board2bits(test_king_board);

    uint64_t initial_atk_bb = board2bits(initial_atk_board);
    uint64_t initial_def_bb = board2bits(initial_def_board);
    uint64_t initial_king_bb = board2bits(initial_king_board);


    // cout << get_board_score(test_atk_bb, test_def_bb, test_king_bb) << endl;

    cout << get_board_score_by_width_search(initial_atk_bb, initial_def_bb, initial_king_bb, 1, 1, 7) << endl;
    for(int i=0; i<8; i++){
        cout << NUM_NODES[i] << endl;
    }
    // print_bitboard(edge_bb);

    // cout << "BEFORE" << endl;
    // print_bitboard(test_atk_bb);
    // print_bitboard(test_def_bb);
    // print_bitboard(test_king_bb);
    // vector<uint64_t> moves = get_all_legal_moves_as_vector(test_atk_bb, test_def_bb, test_king_bb, 1);
    // for(int i=0; i<moves.size()/2; i++){
    //     print_bitboard_move(moves[2*i], moves[2*i+1]);
    // }

    // for(int j=0; j<10000000; j++){
    //     uint64_t atk_bb, def_bb, king_bb;
    //     atk_bb = test_atk_bb;
    //     def_bb = test_def_bb;
    //     king_bb = test_king_bb;
    //     vector<uint64_t> moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, 1);
    //     for(int i=0; i<moves.size()/2; i++){
    //         make_move_on_board(atk_bb, def_bb, king_bb, moves[2*i], moves[2*i+1]);
    //     }
    // }

    // make_move_on_board(test_atk_bb, test_def_bb, test_king_bb, 4294967296, 34359738368);
    // cout << "AFTER" << endl;
    // print_bitboard(test_atk_bb);
    // print_bitboard(test_def_bb);
    // print_bitboard(test_king_bb);

    // for(int i=0; i<10000000; i++){
    //     get_legal_moves(test_piece_bb, test_blocker_bb, move_lookup);
    // }

    // uint64_t asdf = 1;
    // cout << __builtin_clzll(asdf) << endl;
    // cout << __builtin_ctzll(asdf) << endl;
    // asdf = asdf << 63;
    // cout << __builtin_clz(asdf) << endl;
    // cout << __builtin_ctz(asdf) << endl;
    // cout << __builtin_ffs(asdf) << endl;

    // uint64_t test_bb = board2bits(test_board);

    // cout << test_bb << endl;

    // vector<vector<uint64_t>> test_board_out = bits2board(test_bb + 1);

    // print_board(test_board_out);

    // print_bitboard(test_bb + 1);

}