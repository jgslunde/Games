#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <random>
#include <thread>
#include <map>
#include <fstream>

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

// const uint64_t throne_bb = 0x8000000;
const uint64_t corner_bb = 0x41000000000041;
const uint64_t edge_bb = 0xff80808080808080;
const uint64_t diag2corner_bb = 0x220000002200;
const uint64_t fouredgesides_wrong_bb = 0x80022000800;
const uint64_t right_sideedge_bb = 0x4040400000;
const uint64_t bottom_sideedge_bb = 0x1c00;
const uint64_t left_sideedge_bb = 0x10101000000;
const uint64_t top_sideedge_bb = 0x1c00000000000000;
const uint64_t corner_neighbors_bb = 0x22410000004122;

vector<uint64_t> NUM_NODES(12);
map<uint64_t, int> board_counts;

uint64_t ZOBRIST_NUM_TIMES_READ = 0;
uint64_t ZOBRIST_NUM_TIMES_UPDATED = 0;
uint64_t ZOBRIST_NUM_TIMES_WRITTEN = 0;
uint64_t ZOBRIST_NUM_TIMES_MISSED = 0;
const int BOARD_SIZE = 7;
const int PIECE_TYPES = 3; // Attacker, Defender, King

// Initialize the Zobrist table
uint64_t zobristTable[BOARD_SIZE][BOARD_SIZE][PIECE_TYPES];
uint64_t zobristTurn;

void initializeZobristTable() {
    std::random_device rd;
    std::mt19937_64 engine(rd());
    engine.seed(42);
    std::uniform_int_distribution<uint64_t> dist;

    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            for (int k = 0; k < PIECE_TYPES; ++k) {
                zobristTable[i][j][k] = dist(engine);
            }
        }
    }

    zobristTurn = dist(engine);
}

uint64_t computeHash(const uint64_t atk_bb, const uint64_t def_bb, const uint64_t king_bb, bool isAttackersTurn) {
    uint64_t hash = 0;

    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            int pos = i * 8 + j;
            if (atk_bb & (1ULL << pos)) {
                hash ^= zobristTable[i][j][0];
            } else if (def_bb & (1ULL << pos)) {
                hash ^= zobristTable[i][j][1];
            } else if (king_bb & (1ULL << pos)) {
                hash ^= zobristTable[i][j][2];
            }
        }
    }

    if (isAttackersTurn) {
        hash ^= zobristTurn;
    }

    return hash;
}


uint64_t updateHash(uint64_t old_hash, const uint64_t atk_bb, const uint64_t def_bb, const uint64_t king_bb, uint64_t move_from, uint64_t move_to){
    ZOBRIST_NUM_TIMES_UPDATED++;
    uint64_t new_hash = old_hash;
    int move_from_bit_loc = __builtin_ctzll(move_from);
    int move_to_bit_loc = __builtin_ctzll(move_to);
    int move_from_x = move_from_bit_loc%8;
    int move_from_y = move_from_bit_loc/8;
    int move_to_x = move_to_bit_loc%8;
    int move_to_y = move_to_bit_loc/8;
    if(move_to & atk_bb){
        new_hash ^= zobristTable[move_from_y][move_from_x][0];
        new_hash ^= zobristTable[move_to_y][move_to_x][0];
    }
    else if(move_to & def_bb){
        new_hash ^= zobristTable[move_from_y][move_from_x][1];
        new_hash ^= zobristTable[move_to_y][move_to_x][1];
    }
    else if(move_to & king_bb){
        new_hash ^= zobristTable[move_from_y][move_from_x][2];
        new_hash ^= zobristTable[move_to_y][move_to_x][2];
    }
    new_hash ^= zobristTurn;

    return new_hash;
}


const size_t TABLE_SIZE = 1ULL << 24;
struct TTEntry {
    uint64_t hash;
    float eval;  // example data
    unsigned short int depth;
    // Add other game-related data as needed.
};
TTEntry transpositionTable[TABLE_SIZE];
// TTEntry *transpositionTable = new TTEntry[TABLE_SIZE];


void storeInTable(uint64_t hash, float eval, unsigned short int depth){
    ZOBRIST_NUM_TIMES_WRITTEN++;
    size_t index = hash & (TABLE_SIZE - 1);
    transpositionTable[index].hash = hash;
    transpositionTable[index].eval = eval;
    transpositionTable[index].depth = depth;
}

float* retrieveFromTable(uint64_t hash, unsigned short int search_depth_left) {
    size_t index = hash & (TABLE_SIZE - 1);
    // cout << hash << " " << transpositionTable[index].hash << endl;
    if ((transpositionTable[index].hash == hash) && (search_depth_left <= transpositionTable[index].depth)) {  // Depth of our stored eval >= depth left of current search.
        ZOBRIST_NUM_TIMES_READ++;
        return &transpositionTable[index].eval;
    } else {
        ZOBRIST_NUM_TIMES_MISSED++;
        return nullptr;  // Entry not found (collision or never stored)
    }
}

thread_local std::mt19937 generator(std::hash<std::thread::id>{}(std::this_thread::get_id()));

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


uint64_t flip_bb_vertically(uint64_t bb) {
    uint64_t r1 = (bb & 0x00000000000000FFULL) << 48;
    uint64_t r2 = (bb & 0x000000000000FF00ULL) << 32;
    uint64_t r3 = (bb & 0x0000000000FF0000ULL) << 16;
    uint64_t r4 = (bb & 0x00000000FF000000ULL); // No change in position
    uint64_t r5 = (bb & 0x000000FF00000000ULL) >> 16;
    uint64_t r6 = (bb & 0x0000FF0000000000ULL) >> 32;
    uint64_t r7 = (bb & 0x00FF000000000000ULL) >> 48;

    return r1 | r2 | r3 | r4 | r5 | r6 | r7;
}


uint64_t flip_bb_horizontally(uint64_t bb) {
    // Swap the 1st and 7th columns
    uint64_t swap1 = ((bb & 0x0040404040404040ULL) >> 6) | 
                     ((bb & 0x0001010101010101ULL) << 6);

    // Swap the 2nd and 6th columns
    uint64_t swap2 = ((bb & 0x0020202020202020ULL) >> 4) | 
                     ((bb & 0x0002020202020202ULL) << 4);

    // Swap the 3rd and 5th columns
    uint64_t swap3 = ((bb & 0x0010101010101010ULL) >> 2) | 
                     ((bb & 0x0004040404040404ULL) << 2);

    // Preserve the 4th column as is
    uint64_t middle = bb & 0x0008080808080808ULL;

    return swap1 | swap2 | swap3 | middle;
}


uint64_t transpose_bb(uint64_t board) {
    // Slow, rewrite later.
    uint64_t transposed = 0;

    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            // Calculate the bit position in the original bitboard
            int originalPos = i * 8 + j;

            // Calculate the bit position in the transposed bitboard
            int transposedPos = j * 8 + i;

            // Check if the bit at the original position is set
            if (board & (1ULL << originalPos)) {
                // Set the bit at the transposed position
                transposed |= (1ULL << transposedPos);
            }
        }
    }

    return transposed;
}


uint64_t anti_transpose_bb(uint64_t board){
    uint64_t reflected = 0;
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
            // Calculate the bit position in the original bitboard
            int originalPos = i * 8 + j;
            // Calculate the bit position in the reflected bitboard
            int reflectedPos = (6 - j) * 8 + (6 - i);

            // Check if the bit at the original position is set
            if (board & (1ULL << originalPos)) {
                // Set the bit at the reflected position
                reflected |= (1ULL << reflectedPos);
            }
        }
    }

    return reflected;
}


vector<uint64_t> get_all_board_symetries(uint64_t board){
    vector<uint64_t> all_boards(8);
    all_boards[0] = board;
    all_boards[1] = flip_bb_horizontally(transpose_bb(board));  // 90 degree rotation.
    all_boards[2] = flip_bb_vertically(flip_bb_horizontally(board));  // 180 degree rotation.
    all_boards[3] = flip_bb_vertically(transpose_bb(board));  // 270 degree rotation.
    all_boards[4] = flip_bb_horizontally(board);  // Horizontal flip.
    all_boards[5] = flip_bb_vertically(board);  // Vertical flip.
    all_boards[6] = transpose_bb(board);  // Diagonal reflection.
    all_boards[7] = anti_transpose_bb(board);  // Anti-diagonal reflection.
    return all_boards;
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
        asdf = asdf << (first_blocking_idx - 1);
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
        if((proposed_move != 0) && proposed_move){
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
        if((proposed_move != 0) && proposed_move){
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
        if((proposed_move != 0) && proposed_move){
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
        if((proposed_move != 0) && proposed_move){
            // legal_moves.push_back(piece_bb);
            legal_moves.push_back(proposed_move);
        }
    }
    return legal_moves;
}


float board_heuristic_pieces_only(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  __builtin_popcountll(atk_bb)   // Num of attacking pieces.
            - 2.0*__builtin_popcountll(def_bb);   // Number of defending pieces times two (they are half as many).
}


float board_heuristic_king_free_moves(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
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


// float board_heuristic_king_blocking_pieces_to_corners(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
//     unsigned short int king_loc = __builtin_ctzll(king_bb);
//     float score = 0.0;
//     score += __builtin_popcountll(generate_king_paths_to_corners(king_bb) & atk_bb);
// }


float board_heuristic_king_free_moves_wrong(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    uint64_t blocker_bb = atk_bb | def_bb | edge_bb;
    float score = 0;
    for(int i=0; i<6; i++){
        score -= (float) 0.01*(((king_bb<<i) & (~blocker_bb)) != 0)
                        + (((king_bb>>i) & (~blocker_bb))  != 0)
                        + (((king_bb<<i*8) & (~blocker_bb)) != 0)
                        + (((king_bb>>i*8) & (~blocker_bb)) != 0);
    }
    return score;
}

float board_heuristic_king_neighboring_enemies(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  (float) ((king_bb<<1 & atk_bb) != 0)
          + (float) ((king_bb>>1 & atk_bb) != 0)
          + (float) ((king_bb<<8 & atk_bb) != 0)
          + (float) ((king_bb>>8 & atk_bb) != 0);
}

float board_heuristic_king_neighboring_allies(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  (float) ((king_bb<<1 & atk_bb) != 0)
          + (float) ((king_bb>>1 & atk_bb) != 0)
          + (float) ((king_bb<<8 & atk_bb) != 0)
          + (float) ((king_bb>>8 & atk_bb) != 0);
}

float board_heuristic_attacker_on_edges(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return    (float) ((right_sideedge_bb & atk_bb) != 0)
            + (float) ((left_sideedge_bb & atk_bb) != 0)
            + (float) ((top_sideedge_bb & atk_bb) != 0)
            + (float) ((bottom_sideedge_bb & atk_bb) != 0);
}

float board_heuristic_attacker_on_corners(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return 0.1*__builtin_popcountll(diag2corner_bb & atk_bb);
}

float board_heuristic_atk_next_to_corners(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    // Standing next to the corners is often a bad idea, as you're easy to capture.
    return - 0.1 * (float) __builtin_popcountll(corner_neighbors_bb & atk_bb);
}
float board_heuristic_def_next_to_corners(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    // Standing next to the corners is often a bad idea, as you're easy to capture.
    return 0.1 * (float) __builtin_popcountll(corner_neighbors_bb & def_bb);
}


float board_heuristic_v1(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  __builtin_popcountll(atk_bb)   // Num of attacking pieces.
            - 2.0*__builtin_popcountll(def_bb)   // Number of defending pieces times two (they are half as many).
            + 0.1*__builtin_popcountll(diag2corner_bb & atk_bb);
}


float board_heuristic_v2(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  board_heuristic_v1(atk_bb, def_bb, king_bb);
            + 0.05*__builtin_popcountll(fouredgesides_wrong_bb & atk_bb);
}

float board_heuristic_v3(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return board_heuristic_v2(atk_bb, def_bb, king_bb) + board_heuristic_king_free_moves_wrong(atk_bb, def_bb, king_bb);
}


float board_heuristic_v4(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return board_heuristic_v3(atk_bb, def_bb, king_bb) + board_heuristic_king_neighboring_enemies(atk_bb, def_bb, king_bb);
}


float board_heuristic_v5(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  __builtin_popcountll(atk_bb)   // Num of attacking pieces.
            - 1.5*__builtin_popcountll(def_bb)   // Number of defending pieces times two (they are half as many).
            + 0.1*__builtin_popcountll(diag2corner_bb & atk_bb)
            + 0.05*__builtin_popcountll(fouredgesides_wrong_bb & atk_bb)
            + board_heuristic_king_free_moves_wrong(atk_bb, def_bb, king_bb)
            + board_heuristic_king_neighboring_enemies(atk_bb, def_bb, king_bb);
}

float board_heuristic_v6(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  __builtin_popcountll(atk_bb)   // Num of attacking pieces.
            - 1.5*__builtin_popcountll(def_bb)   // Number of defending pieces times two (they are half as many).
            + 0.1*__builtin_popcountll(diag2corner_bb & atk_bb)
            + 0.05*__builtin_popcountll(fouredgesides_wrong_bb & atk_bb)
            + board_heuristic_king_free_moves(atk_bb, def_bb, king_bb)
            + board_heuristic_king_neighboring_enemies(atk_bb, def_bb, king_bb);
}

float board_heuristic_v7(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  __builtin_popcountll(atk_bb)   // Num of attacking pieces.
            - 1.5*__builtin_popcountll(def_bb)   // Number of defending pieces times two (they are half as many).
            + board_heuristic_attacker_on_corners(atk_bb, def_bb, king_bb)
            + board_heuristic_king_free_moves(atk_bb, def_bb, king_bb)
            + board_heuristic_king_neighboring_enemies(atk_bb, def_bb, king_bb)
            + board_heuristic_attacker_on_edges(atk_bb, def_bb, king_bb)
            - 2.0;
}

float board_heuristic_v8(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return board_heuristic_v7(atk_bb, def_bb, king_bb)
    + board_heuristic_atk_next_to_corners(atk_bb, def_bb, king_bb)
    + board_heuristic_def_next_to_corners(atk_bb, def_bb, king_bb);
}


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


float combined_board_heuristics(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, HeuristicsConfig *config){
    return  1.0*config->atk_pieces_weight * __builtin_popcountll(atk_bb)
            - 2.0*config->def_pieces_weight * __builtin_popcountll(def_bb)
            - 0.05*config->king_free_moves_weight * board_heuristic_king_free_moves(atk_bb, def_bb, king_bb)
            + 0.2*config->king_neighboring_enemies_weight * board_heuristic_king_neighboring_enemies(atk_bb, def_bb, king_bb)
            + 0.08*config->king_neighboring_allies_weight * board_heuristic_king_neighboring_allies(atk_bb, def_bb, king_bb)
            + 0.06*config->atk_pieces_on_edges_weight * board_heuristic_attacker_on_edges(atk_bb, def_bb, king_bb)
            + 0.1*config->atk_pieces_diag_to_corners_weight * __builtin_popcountll(diag2corner_bb & atk_bb)
            - 0.15*config->atk_pieces_next_to_corners_weight *  __builtin_popcountll(corner_neighbors_bb & atk_bb)
            + 0.15*config->def_pieces_next_to_corners_weight *  __builtin_popcountll(corner_neighbors_bb & def_bb);
}




float get_board_wins(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    // 106 instructions.
    uint64_t king_hostile_bb = atk_bb | corner_bb;
    float score = 
            - 1000.0*(__builtin_popcountll(atk_bb) == 0)   // Defender wins if attacker is out of pieces.
            - 1000.0*((king_bb & corner_bb) > 0)   // Defender wins if the king reaces a corner.
            + 1000.0*(!king_bb);   // Attacker wins if king is captured, and therefore not on the board.

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


uint64_t make_move_on_board(uint64_t &atk_bb, uint64_t &def_bb, uint64_t &king_bb, uint64_t move_from, uint64_t move_to){
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
    return captures;
}


vector<uint64_t> get_all_legal_moves_as_vector(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player){
    // 284 instructions, including calls to "get_legal_moves_as_vector".
    vector<uint64_t> legal_moves;
    legal_moves.reserve(100);
    uint64_t piece_bb;
    if(player == PLAYER_ATK){
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


float get_board_score_by_width_search_zobrist(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, unsigned short int depth, unsigned short int max_depth, uint64_t hash, float (*heuristic_function)(uint64_t, uint64_t, uint64_t)){
    // Width-search using zobrist hashing, without symmetries, meaning that only exact board configurations will be stored, but reduces number of hashes.
    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size()/2;
    vector<float> move_scores(num_legal_moves);
    NUM_NODES[depth] += num_legal_moves;
    for(int imove=0; imove<num_legal_moves; imove++){
        uint64_t atk_bb_new = atk_bb;
        uint64_t def_bb_new = def_bb;
        uint64_t king_bb_new = king_bb;
        uint64_t captures = make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*imove], legal_moves[2*imove+1]);

        // uint64_t new_hash = computeHash(atk_bb_new, def_bb_new, king_bb_new, (-player)==1);
        uint64_t new_hash;
        if(captures){
            new_hash = computeHash(atk_bb_new, def_bb_new, king_bb_new, (-player)==1);
        }else{
            new_hash = updateHash(hash, atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*imove], legal_moves[2*imove+1]);
        }
        // if(new_hash != new_hash_2){
        //     cout << depth << "  " << new_hash << " " << new_hash_2 << "  -  " << (new_hash_2==new_hash) << endl;
        //     print_bitgame(atk_bb_new, def_bb_new, king_bb_new);
        //     print_bitboard_move(legal_moves[2*imove], legal_moves[2*imove+1]);
        // }
        float* zobrist_score_ptr = retrieveFromTable(new_hash, max_depth-depth);
        if(zobrist_score_ptr){  // If we didn't get a nullptr in return, the board position already exists in our table.
            // We then therminate the entire rest of the search down this branch, and set the score:
            // cout << "hello" << endl;
            move_scores[imove] = *zobrist_score_ptr;
        }
        else{
            // If not, we will have to calculate the board score:
            float move_score = get_board_wins(atk_bb_new, def_bb_new, king_bb_new) + heuristic_function(atk_bb_new, def_bb_new, king_bb_new);
            if((depth >= max_depth) || abs(move_score) > 900){  // If we're at max depth, or someone has won, we terminate.
                move_scores[imove] = (1 - 0.01*depth)*move_score;  // Encourage winning moves earlier rather than later.
            }
            else{  // If nobody has won and we're not at max depth, we continue down the tree:
                move_scores[imove] = get_board_score_by_width_search_zobrist(atk_bb_new, def_bb_new, king_bb_new, -player, depth+1, max_depth, new_hash, heuristic_function);
            }
        storeInTable(new_hash, move_scores[imove], max_depth-depth);
        }
    }

    // int best_board_score;
    if(player == PLAYER_ATK){
        float best_board_score = -999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] > best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
    else{
        float best_board_score = 999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] < best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
}


float get_board_score_by_width_search_sym_zobrist(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, unsigned short int depth, unsigned short int max_depth, float (*heuristic_function)(uint64_t, uint64_t, uint64_t)){
    // Width-search using zobrist hashing with symmetries, meaning that all 8 board symmetries must be tracked, but allows for more hash table hits.
    NUM_NODES[depth] += 1;
    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size()/2;
    vector<float> move_scores(num_legal_moves);
    for(int imove=0; imove<num_legal_moves; imove++){
        uint64_t atk_bb_new = atk_bb;
        uint64_t def_bb_new = def_bb;
        uint64_t king_bb_new = king_bb;
        make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*imove], legal_moves[2*imove+1]);

        vector<uint64_t> all_sym_atk_bb = get_all_board_symetries(atk_bb_new);
        vector<uint64_t> all_sym_def_bb = get_all_board_symetries(def_bb_new);
        vector<uint64_t> all_sym_king_bb = get_all_board_symetries(king_bb_new);
        vector<uint64_t> sym_hashes(8);
        for(int i=0; i<8; i++)
            sym_hashes[i] = computeHash(all_sym_atk_bb[i], all_sym_def_bb[i], all_sym_king_bb[i], player==1);
        uint64_t hash = min(min(min(sym_hashes[0], sym_hashes[1]), min(sym_hashes[2], sym_hashes[3])), min(min(sym_hashes[4], sym_hashes[5]), min(sym_hashes[6], sym_hashes[7])));
        // uint64_t hash = computeHash(atk_bb_new, def_bb_new, king_bb_new, player==1);
        float* zobrist_score_ptr = retrieveFromTable(hash, max_depth-depth);
        if(zobrist_score_ptr){  // If we didn't get a nullptr in return, the board position already exists in our table.
            // We then therminate the entire rest of the search down this branch, and set the score:
            // cout << "hello" << endl;
            move_scores[imove] = *zobrist_score_ptr;
        }
        else{
            // If not, we will have to calculate the board score:
            float move_score = get_board_wins(atk_bb_new, def_bb_new, king_bb_new) + heuristic_function(atk_bb_new, def_bb_new, king_bb_new);
            if((depth >= max_depth) || abs(move_score) > 900){  // If we're at max depth, or someone has won, we terminate.
                move_scores[imove] = (1 - 0.01*depth)*move_score;
            }
            else{  // If nobody has won and we're not at max depth, we continue down the tree:
                move_scores[imove] = get_board_score_by_width_search_sym_zobrist(atk_bb_new, def_bb_new, king_bb_new, -player, depth+1, max_depth, heuristic_function);
            }
        storeInTable(hash, move_scores[imove], max_depth-depth);
        }
    }

    // int best_board_score;
    if(player == PLAYER_ATK){
        float best_board_score = -999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] > best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
    else{
        float best_board_score = 999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] < best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
}


float get_board_score_by_width_search(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, unsigned short int depth, unsigned short int max_depth, float (*heuristic_function)(uint64_t, uint64_t, uint64_t)){
    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size()/2;
    vector<float> move_scores(num_legal_moves);
    NUM_NODES[depth] += num_legal_moves;
    for(int imove=0; imove<num_legal_moves; imove++){
        uint64_t atk_bb_new = atk_bb;
        uint64_t def_bb_new = def_bb;
        uint64_t king_bb_new = king_bb;
        make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*imove], legal_moves[2*imove+1]);
        // If not, we will have to calculate the board score:
        float move_score = get_board_wins(atk_bb_new, def_bb_new, king_bb_new) + heuristic_function(atk_bb_new, def_bb_new, king_bb_new);
        if((depth >= max_depth) || abs(move_score) > 900){  // If we're at max depth, or someone has won, we terminate.
            move_scores[imove] = (1 - 0.01*depth)*move_score;
        }
        else{  // If nobody has won and we're not at max depth, we continue down the tree:
            move_scores[imove] = get_board_score_by_width_search(atk_bb_new, def_bb_new, king_bb_new, -player, depth+1, max_depth, heuristic_function);
        }
    }

    if(player == PLAYER_ATK){
        float best_board_score = -999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] > best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
    else{
        float best_board_score = 999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] < best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
}

float get_board_score_by_alpha_beta_search(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, unsigned short int depth, unsigned short int max_depth, float alpha, float beta, HeuristicsConfig *config){
    float board_wins = get_board_wins(atk_bb, def_bb, king_bb);
    if((depth >= max_depth) || (abs(board_wins) > 100)){ // Base case: terminal depth or leaf node
        return (combined_board_heuristics(atk_bb, def_bb, king_bb, config) + board_wins)*(1 - 0.01*(depth-1));
    }

    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size() / 2;
    if(num_legal_moves == 0){
        return (float) -1000*player;
    }
    NUM_NODES[depth] += num_legal_moves;

    if(player == PLAYER_ATK){
        float best_score = -999999;
        for(int imove = 0; imove < num_legal_moves; imove++){
            uint64_t atk_bb_new = atk_bb;
            uint64_t def_bb_new = def_bb;
            uint64_t king_bb_new = king_bb;
            make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*imove], legal_moves[2*imove + 1]);
            float score = get_board_score_by_alpha_beta_search(atk_bb_new, def_bb_new, king_bb_new, -player, depth + 1, max_depth, alpha, beta, config);
            best_score = max(best_score, score);
            alpha = max(alpha, best_score);
            if(beta <= alpha){ // Beta cut-off
                break;
            }
        }
        return best_score;
    }
    else{
        float best_score = 999999;
        for(int imove = 0; imove < num_legal_moves; imove++){
            uint64_t atk_bb_new = atk_bb;
            uint64_t def_bb_new = def_bb;
            uint64_t king_bb_new = king_bb;
            make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*imove], legal_moves[2*imove + 1]);
            float score = get_board_score_by_alpha_beta_search(atk_bb_new, def_bb_new, king_bb_new, -player, depth + 1, max_depth, alpha, beta, config);
            best_score = min(best_score, score);
            beta = min(beta, best_score);
            if(beta <= alpha){ // Alpha cut-off
                break;
            }
        }
        return best_score;
    }
}


vector<uint64_t> AI_alpha_beta_get_move(float *eval, uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, unsigned short int max_depth, bool verbose, HeuristicsConfig *config){
    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size()/2;
    if(num_legal_moves == 0){
        return vector<uint64_t>{0, 0};
    }
    int preffered_move_idx = 0;
    vector<float> move_scores(num_legal_moves);

    NUM_NODES[0] = num_legal_moves;

    #pragma omp parallel for
    for(int i=0; i<num_legal_moves; i++){
        uint64_t atk_bb_new, def_bb_new, king_bb_new;
        atk_bb_new = atk_bb;
        def_bb_new = def_bb;
        king_bb_new = king_bb;
        make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*i], legal_moves[2*i+1]);
        float move_score;
        move_score = get_board_score_by_alpha_beta_search(atk_bb_new, def_bb_new, king_bb_new, -player, 1, max_depth, -INFINITY, INFINITY, config);
        move_scores[i] = move_score;
    }

    // Finding the preffered (highest or lowest, depending on player) score among the options.
    float best_move_score = -9999999.0*player;
    for(int i=0; i<num_legal_moves; i++){
        if(move_scores[i]*player > best_move_score*player){
            best_move_score = move_scores[i];
        }
    }
    // Finding all the moves with the preffered score.
    vector<int> preffered_move_indices;
    for(int i=0; i<num_legal_moves; i++){
        if(move_scores[i] == best_move_score){
            preffered_move_indices.push_back(i);
        }
    }
    // Chosing a random among those.
    uint64_t chosen_move_index = preffered_move_indices[thread_safe_rand()%preffered_move_indices.size()];

    if(verbose){
        cout << "Player: " << player << ". Board eval: " << best_move_score << endl;
    }
    *eval = best_move_score;
    return {legal_moves[2*chosen_move_index], legal_moves[2*chosen_move_index+1]};
}


vector<uint64_t> AI_zobrist_get_move(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, unsigned short int max_depth, bool verbose, float (*heuristic_function)(uint64_t, uint64_t, uint64_t)){
    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size()/2;
    int preffered_move_idx = 0;
    vector<float> move_scores(num_legal_moves);

    #pragma omp parallel for
    for(int imove=0; imove<num_legal_moves; imove++){
        uint64_t atk_bb_new, def_bb_new, king_bb_new;
        atk_bb_new = atk_bb;
        def_bb_new = def_bb;
        king_bb_new = king_bb;
        make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*imove], legal_moves[2*imove+1]);
        float move_score;
        uint64_t hash = computeHash(atk_bb, def_bb, king_bb, player==1);
        float* zobrist_score_ptr = retrieveFromTable(hash, max_depth-1);
        if(zobrist_score_ptr){  // If we didn't get a nullptr in return, the board position already exists in our table.
            // We then therminate the entire rest of the search down this branch, and set the score:
            move_scores[imove] = *zobrist_score_ptr;
        }
        else{
            move_score = get_board_score_by_width_search_zobrist(atk_bb_new, def_bb_new, king_bb_new, -player, 2, max_depth, hash, heuristic_function);
            move_scores[imove] = move_score;
            storeInTable(hash, move_scores[imove], max_depth-1);
        }
    }

    // Finding the preffered (highest or lowest, depending on player) score among the options.
    float best_move_score = -9999999.0*player;
    for(int i=0; i<num_legal_moves; i++){
        if(move_scores[i]*player > best_move_score*player){
            best_move_score = move_scores[i];
        }
    }
    // Finding all the moves with the preffered score.
    vector<int> preffered_move_indices;
    for(int i=0; i<num_legal_moves; i++){
        if(move_scores[i] == best_move_score){
            preffered_move_indices.push_back(i);
        }
    }
    // Chosing a random among those.
    uint64_t chosen_move_index = preffered_move_indices[thread_safe_rand()%preffered_move_indices.size()];

    if(verbose){
        cout << "Player: " << player << ". Board eval: " << best_move_score << endl;
    }
    return {legal_moves[2*chosen_move_index], legal_moves[2*chosen_move_index+1]};
}


vector<uint64_t> AI_sym_zobrist_get_move(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, unsigned short int max_depth, bool verbose, float (*heuristic_function)(uint64_t, uint64_t, uint64_t)){
    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size()/2;
    int preffered_move_idx = 0;
    vector<float> move_scores(num_legal_moves);

    #pragma omp parallel for
    for(int imove=0; imove<num_legal_moves; imove++){
        uint64_t atk_bb_new, def_bb_new, king_bb_new;
        atk_bb_new = atk_bb;
        def_bb_new = def_bb;
        king_bb_new = king_bb;
        make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*imove], legal_moves[2*imove+1]);
        float move_score;
        vector<uint64_t> all_sym_atk_bb = get_all_board_symetries(atk_bb_new);
        vector<uint64_t> all_sym_def_bb = get_all_board_symetries(def_bb_new);
        vector<uint64_t> all_sym_king_bb = get_all_board_symetries(king_bb_new);
        vector<uint64_t> sym_hashes(8);
        for(int i=0; i<8; i++)
            sym_hashes[i] = computeHash(all_sym_atk_bb[i], all_sym_def_bb[i], all_sym_king_bb[i], player==1);
        uint64_t hash = min(min(min(sym_hashes[0], sym_hashes[1]), min(sym_hashes[2], sym_hashes[3])), min(min(sym_hashes[4], sym_hashes[5]), min(sym_hashes[6], sym_hashes[7])));
        float* zobrist_score_ptr = retrieveFromTable(hash, max_depth-1);
        if(zobrist_score_ptr){  // If we didn't get a nullptr in return, the board position already exists in our table.
            // We then therminate the entire rest of the search down this branch, and set the score:
            move_scores[imove] = *zobrist_score_ptr;
        }
        else{
            move_score = get_board_score_by_width_search_sym_zobrist(atk_bb_new, def_bb_new, king_bb_new, -player, 2, max_depth, heuristic_function);
            move_scores[imove] = move_score;
            storeInTable(hash, move_scores[imove], max_depth-1);
        }
    }

    // Finding the preffered (highest or lowest, depending on player) score among the options.
    float best_move_score = -9999999.0*player;
    for(int i=0; i<num_legal_moves; i++){
        if(move_scores[i]*player > best_move_score*player){
            best_move_score = move_scores[i];
        }
    }
    // Finding all the moves with the preffered score.
    vector<int> preffered_move_indices;
    for(int i=0; i<num_legal_moves; i++){
        if(move_scores[i] == best_move_score){
            preffered_move_indices.push_back(i);
        }
    }
    // Chosing a random among those.
    uint64_t chosen_move_index = preffered_move_indices[thread_safe_rand()%preffered_move_indices.size()];

    if(verbose){
        cout << "Player: " << player << ". Board eval: " << best_move_score << endl;
    }
    return {legal_moves[2*chosen_move_index], legal_moves[2*chosen_move_index+1]};
}


vector<uint64_t> AI_1_get_move(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, unsigned short int max_depth, bool verbose, float (*heuristic_function)(uint64_t, uint64_t, uint64_t)){
    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size()/2;
    int preffered_move_idx = 0;
    vector<float> move_scores(num_legal_moves);

    #pragma omp parallel for
    for(int i=0; i<num_legal_moves; i++){
        uint64_t atk_bb_new, def_bb_new, king_bb_new;
        atk_bb_new = atk_bb;
        def_bb_new = def_bb;
        king_bb_new = king_bb;
        make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*i], legal_moves[2*i+1]);
        float move_score;
        move_score = get_board_score_by_width_search(atk_bb_new, def_bb_new, king_bb_new, -player, 2, max_depth, heuristic_function);
        move_scores[i] = move_score;
    }

    // Finding the preffered (highest or lowest, depending on player) score among the options.
    float best_move_score = -9999999.0*player;
    for(int i=0; i<num_legal_moves; i++){
        if(move_scores[i]*player > best_move_score*player){
            best_move_score = move_scores[i];
        }
    }
    // Finding all the moves with the preffered score.
    vector<int> preffered_move_indices;
    for(int i=0; i<num_legal_moves; i++){
        if(move_scores[i] == best_move_score){
            preffered_move_indices.push_back(i);
        }
    }
    // Chosing a random among those.
    uint64_t chosen_move_index = preffered_move_indices[thread_safe_rand()%preffered_move_indices.size()];

    if(verbose){
        cout << "Player: " << player << ". Board eval: " << best_move_score << endl;
    }
    return {legal_moves[2*chosen_move_index], legal_moves[2*chosen_move_index+1]};
}


extern "C" {
    void AI_web_get_move(float *eval, int *move_from_x, int *move_from_y, int *move_to_x, int *move_to_y, int *JS_board, int player, int max_depth){
        vector<vector<uint64_t>> board_atk(8, vector<uint64_t>(8));
        vector<vector<uint64_t>> board_def(8, vector<uint64_t>(8));
        vector<vector<uint64_t>> board_king(8, vector<uint64_t>(8));
        for(int y=0; y<7; y++){
            for(int x=0; x<7; x++){
                int idx = y*7 + x;  // The JS boards are only 7x7.
                if(JS_board[idx] == 1)
                    board_atk[y][x] = 1;
                if(JS_board[idx] == 2)
                    board_def[y][x] = 1;
                if(JS_board[idx] == 3)
                    board_king[y][x] = 1;
            }
        }
        uint64_t atk_bb = board2bits(board_atk);
        uint64_t def_bb = board2bits(board_def);
        uint64_t king_bb = board2bits(board_king);

        HeuristicsConfig config;

        vector<uint64_t> chosen_move = AI_alpha_beta_get_move(eval, atk_bb, def_bb, king_bb, player, max_depth, false, &config);
        uint64_t move_from = chosen_move[0];
        uint64_t move_to = chosen_move[1];
        int move_from_bit_loc = __builtin_ctzll(move_from);
        int move_to_bit_loc = __builtin_ctzll(move_to);
        *move_from_x = move_from_bit_loc%8;
        *move_from_y = move_from_bit_loc/8;
        *move_to_x = move_to_bit_loc%8;
        *move_to_y = move_to_bit_loc/8;
    }
}


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


float AI_vs_AI_tournament(int num_games, int depth1, int depth2, HeuristicsConfig *config1, HeuristicsConfig *config2, bool verbose, bool vverbose){
    uint64_t initial_atk_bb = 0x8080063000808;
    uint64_t initial_def_bb = 0x814080000;
    uint64_t initial_king_bb = 0x8000000;

    int num_AI_1_wins_atk = 0;
    int num_AI_1_wins_def = 0;
    int num_AI_1_ties_atk = 0;
    int num_AI_1_ties_def = 0;
    int num_AI_2_wins_atk = 0;
    int num_AI_2_wins_def = 0;
    int num_ties = 0;
    int AI_1_playing_as = 1;
    for(int game=0; game<num_games; game++){
        if(vverbose)
            cout << "Game: " << game << endl;
        uint64_t atk_bb = initial_atk_bb;
        uint64_t def_bb = initial_def_bb;
        uint64_t king_bb = initial_king_bb;
        int current_player = 1;
        int score = 0;
        unsigned short int depth = 4;
        int iturn = 0;
        while (true){
            if(current_player == 1){
                depth=depth1;
            }else{
                depth=depth2;
            }
            vector<uint64_t> preffered_move;
            float eval;
            if(current_player*AI_1_playing_as == 1)
                preffered_move = AI_alpha_beta_get_move(&eval, atk_bb, def_bb, king_bb, current_player, depth, false, config1);
            else
                preffered_move = AI_alpha_beta_get_move(&eval, atk_bb, def_bb, king_bb, current_player, depth, false, config2);
            if((preffered_move[0] != 0) && (preffered_move[1] != 0)){
                make_move_on_board(atk_bb, def_bb, king_bb, preffered_move[0], preffered_move[1]);
                score = get_board_wins(atk_bb, def_bb, king_bb) + board_heuristic_pieces_only(atk_bb, def_bb, king_bb);
            }
            else{ // No legal moves.
                score = -1000*current_player;
            }

            if(abs(score) > 800){
                if(score < 0){
                    if(AI_1_playing_as == -1)
                        num_AI_1_wins_def++;
                    else
                        num_AI_2_wins_def++;
                    if(vverbose){
                        cout << "DEFENDER WINS" << endl;
                        print_bitgame(atk_bb, def_bb, king_bb);
                    }
                }else{
                    if(AI_1_playing_as == 1)
                        num_AI_1_wins_atk++;
                    else
                        num_AI_2_wins_atk++;
                    if(vverbose){
                        cout << "ATTACKER WINS" << endl;
                        print_bitgame(atk_bb, def_bb, king_bb);
                    }
                }
                break;
            }
            current_player *= -1;
            iturn++;
            if(iturn >= 100){
                if(AI_1_playing_as == 1)
                    num_AI_1_ties_atk++;
                else
                    num_AI_1_ties_def++;
                if(vverbose){
                    cout << "100 TURNS REACHED" << endl;
                    print_bitgame(atk_bb, def_bb, king_bb);
                }
                break;
            }
        }
        AI_1_playing_as *= -1;
    }
    float AI_1_score = (float) ((num_AI_1_wins_atk + num_AI_1_wins_def) - (num_AI_2_wins_atk + num_AI_2_wins_def))/(float) num_games;
    if(verbose){
        cout << "Total number of atk wins for AI_1: " << num_AI_1_wins_atk << endl;
        cout << "Total number of atk wins for AI_2: " << num_AI_2_wins_atk << endl;
        cout << "Total number of def wins for AI_1: " << num_AI_1_wins_def << endl;
        cout << "Total number of def wins for AI_2: " << num_AI_2_wins_def << endl;
        cout << "Total number of draws with AI_1 as atk: " << num_AI_1_ties_atk << endl;
        cout << "Total number of draws with AI_1 as def: " << num_AI_1_ties_def << endl;
        cout << "AI_1/AI_2 win ratio as atk: " << (float) num_AI_1_wins_atk / (float)num_AI_2_wins_atk << endl;
        cout << "AI_1/AI_2 win ratio as def: " << (float) num_AI_1_wins_def / (float)num_AI_2_wins_def << endl;
        cout << "AI_1_score: " << AI_1_score << endl;
    }


    return AI_1_score;
}


float modified_AI_vs_AI_tournament(int num_games, int depth1, int depth2, HeuristicsConfig *config1, HeuristicsConfig *config2, bool verbose, bool vverbose){
    uint64_t initial_atk_bb = 0x8080063000808;
    uint64_t initial_def_bb = 0x814080000;
    uint64_t initial_king_bb = 0x8000000;

    int num_AI_1_wins_atk = 0;
    int num_AI_1_wins_def = 0;
    int num_AI_1_ties_atk = 0;
    int num_AI_1_ties_def = 0;
    int num_AI_2_wins_atk = 0;
    int num_AI_2_wins_def = 0;
    int num_ties = 0;
    int AI_1_playing_as = 1;
    uint64_t starting_atk_bb = initial_atk_bb;
    uint64_t starting_def_bb = initial_def_bb;
    uint64_t starting_king_bb = initial_king_bb;
    for(int game=0; game<num_games; game++){
        if(AI_1_playing_as == 1){  // Only create new board after both players had a chance to play both sides.
            int num_premoves = (int) floor(((float) 3*game)/num_games);
            if(vverbose){
                cout << "Num premoves: " << num_premoves << endl;
            }
            starting_atk_bb = initial_atk_bb;
            starting_def_bb = initial_def_bb;
            starting_king_bb = initial_king_bb;
            uint64_t move_from, move_to, move_idx;
            vector<uint64_t> legal_moves;
            for(int ipremove=0; ipremove<num_premoves; ipremove++){
                // Attacker move.
                legal_moves = get_all_legal_moves_as_vector(starting_atk_bb, starting_def_bb, starting_king_bb, 1);
                move_idx = thread_safe_rand()%(legal_moves.size()/2);
                move_from = legal_moves[2*move_idx];
                move_to = legal_moves[2*move_idx+1];
                make_move_on_board(starting_atk_bb, starting_def_bb, starting_king_bb, move_from, move_to);
                // Defender move.
                legal_moves = get_all_legal_moves_as_vector(starting_atk_bb, starting_def_bb, starting_king_bb, -1);
                move_idx = thread_safe_rand()%(legal_moves.size()/2);
                move_from = legal_moves[2*move_idx];
                move_to = legal_moves[2*move_idx+1];
                make_move_on_board(starting_atk_bb, starting_def_bb, starting_king_bb, move_from, move_to);
            }
        }
        uint64_t atk_bb = starting_atk_bb;
        uint64_t def_bb = starting_def_bb;
        uint64_t king_bb = starting_king_bb;

        if(vverbose)
            cout << "Game: " << game << endl;
        int current_player = 1;
        int score = 0;
        unsigned short int depth = 4;
        int iturn = 0;
        while (true){
            if(current_player == 1){
                depth=depth1;
            }else{
                depth=depth2;
            }
            vector<uint64_t> preffered_move;
            float eval;
            if(current_player*AI_1_playing_as == 1)
                preffered_move = AI_alpha_beta_get_move(&eval, atk_bb, def_bb, king_bb, current_player, depth, false, config1);
            else
                preffered_move = AI_alpha_beta_get_move(&eval, atk_bb, def_bb, king_bb, current_player, depth, false, config2);
            if((preffered_move[0] != 0) && (preffered_move[1] != 0)){
                make_move_on_board(atk_bb, def_bb, king_bb, preffered_move[0], preffered_move[1]);
                score = get_board_wins(atk_bb, def_bb, king_bb) + board_heuristic_pieces_only(atk_bb, def_bb, king_bb);
            }
            else{ // No legal moves.
                score = -1000*current_player;
            }

            if(abs(score) > 800){
                if(score < 0){
                    if(AI_1_playing_as == -1)
                        num_AI_1_wins_def++;
                    else
                        num_AI_2_wins_def++;
                    if(vverbose){
                        cout << "DEFENDER WINS" << endl;
                        print_bitgame(atk_bb, def_bb, king_bb);
                    }
                }else{
                    if(AI_1_playing_as == 1)
                        num_AI_1_wins_atk++;
                    else
                        num_AI_2_wins_atk++;
                    if(vverbose){
                        cout << "ATTACKER WINS" << endl;
                        print_bitgame(atk_bb, def_bb, king_bb);
                    }
                }
                break;
            }
            current_player *= -1;
            iturn++;
            if(iturn >= 100){
                if(AI_1_playing_as == 1)
                    num_AI_1_ties_atk++;
                else
                    num_AI_1_ties_def++;
                if(vverbose){
                    cout << "100 TURNS REACHED" << endl;
                    print_bitgame(atk_bb, def_bb, king_bb);
                }
                break;
            }
        }
        AI_1_playing_as *= -1;
    }
    float AI_1_score = (float) ((num_AI_1_wins_atk + num_AI_1_wins_def) - (num_AI_2_wins_atk + num_AI_2_wins_def))/(float) num_games;
    if(verbose){
        cout << "Total number of atk wins for AI_1: " << num_AI_1_wins_atk << endl;
        cout << "Total number of atk wins for AI_2: " << num_AI_2_wins_atk << endl;
        cout << "Total number of def wins for AI_1: " << num_AI_1_wins_def << endl;
        cout << "Total number of def wins for AI_2: " << num_AI_2_wins_def << endl;
        cout << "Total number of draws with AI_1 as atk: " << num_AI_1_ties_atk << endl;
        cout << "Total number of draws with AI_1 as def: " << num_AI_1_ties_def << endl;
        cout << "AI_1/AI_2 win ratio as atk: " << (float) num_AI_1_wins_atk / (float)num_AI_2_wins_atk << endl;
        cout << "AI_1/AI_2 win ratio as def: " << (float) num_AI_1_wins_def / (float)num_AI_2_wins_def << endl;
        cout << "AI_1_score: " << AI_1_score << endl;
    }

    return AI_1_score;
}


void SPSA_update_parameters(HeuristicsConfig &current_config, vector<HeuristicsConfig>all_configs, double alpha, double sigma){
    cout << "alpha = " << alpha << endl;
    cout << "sigma = " << sigma << endl;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0, sigma);
    HeuristicsConfig delta_weights = {
        0,
        distribution(generator),
        distribution(generator),
        distribution(generator),
        distribution(generator),
        distribution(generator),
        distribution(generator),
        distribution(generator),
        distribution(generator),
    };


    HeuristicsConfig config_plus = current_config;
    config_plus.atk_pieces_weight;
    config_plus.def_pieces_weight += delta_weights.def_pieces_weight;
    config_plus.king_free_moves_weight += delta_weights.king_free_moves_weight;
    config_plus.king_neighboring_enemies_weight; // += delta_weights.king_neighboring_enemies_weight;
    config_plus.king_neighboring_allies_weight; // += delta_weights.king_neighboring_allies_weight;
    config_plus.atk_pieces_on_edges_weight; // += delta_weights.atk_pieces_on_edges_weight;
    config_plus.atk_pieces_diag_to_corners_weight; // += delta_weights.atk_pieces_diag_to_corners_weight;
    config_plus.atk_pieces_next_to_corners_weight; // += delta_weights.atk_pieces_next_to_corners_weight;
    config_plus.def_pieces_next_to_corners_weight; // += delta_weights.def_pieces_next_to_corners_weight;

    HeuristicsConfig config_minus = current_config;
    config_minus.atk_pieces_weight;
    config_minus.def_pieces_weight - delta_weights.def_pieces_weight;
    config_minus.king_free_moves_weight - delta_weights.king_free_moves_weight;
    config_minus.king_neighboring_enemies_weight; // - delta_weights.king_neighboring_enemies_weight;
    config_minus.king_neighboring_allies_weight; // - delta_weights.king_neighboring_allies_weight;
    config_minus.atk_pieces_on_edges_weight; // - delta_weights.atk_pieces_on_edges_weight;
    config_minus.atk_pieces_diag_to_corners_weight; // - delta_weights.atk_pieces_diag_to_corners_weight;
    config_minus.atk_pieces_next_to_corners_weight; // - delta_weights.atk_pieces_next_to_corners_weight;
    config_minus.def_pieces_next_to_corners_weight; // - delta_weights.def_pieces_next_to_corners_weight;

    float AI_plus_score = 0;  // The win performance score of the "plus delta" AI. Number between -1 and 1.
    // float AI_minus_score = 0;  // The win performance score of the "plus delta" AI. Number between -1 and 1.
    // First, play once the "plus AI" vs the "minus AI".
    // AI_plus_score += modified_AI_vs_AI_tournament(200, 2, 2, &config_plus, &config_minus, false, false);
    // AI_plus_score += modified_AI_vs_AI_tournament(200, 2, 2, &config_plus, &config_minus, false, false);
    
    // // Then they both play against the initial AI. The plus AI lose points if the minus AI wins against the initial AI.
    AI_plus_score += modified_AI_vs_AI_tournament(1000, 2, 2, &config_plus, &all_configs[0], false, false);
    AI_plus_score -= modified_AI_vs_AI_tournament(1000, 2, 2, &config_minus, &all_configs[0], false, false);

    // for(int i=0; i<4; i++){  // 4 random previous configs to play against.
    //     unsigned int rnd_config_idx = thread_safe_rand()%all_configs.size();
    //     HeuristicsConfig rnd_config = all_configs[rnd_config_idx];
    //     // Both AIs play against the previous AI.
    //     AI_plus_score += modified_AI_vs_AI_tournament(200, 2, 2, &config_plus, &rnd_config, false, false);
    //     AI_plus_score -= modified_AI_vs_AI_tournament(200, 2, 2, &config_minus, &rnd_config, false, false);
    // }
    // AI_plus_score /= 12;  // We've played a total of 12 tournaments, each with a score from -1 to 1, so we shrink the range back to -1 to 1.

    cout << "AI plus win rate score = " << AI_plus_score << endl;

    // current_config.atk_pieces_weight += alpha*AI_plus_score*delta_weights.atk_pieces_weight;
    current_config.def_pieces_weight += alpha*AI_plus_score*delta_weights.def_pieces_weight;
    current_config.king_free_moves_weight += alpha*AI_plus_score*delta_weights.king_free_moves_weight;
    //current_config.king_neighboring_enemies_weight += alpha*AI_plus_score*delta_weights.king_neighboring_enemies_weight;
    //current_config.king_neighboring_allies_weight += alpha*AI_plus_score*delta_weights.king_neighboring_allies_weight;
    //current_config.atk_pieces_on_edges_weight += alpha*AI_plus_score*delta_weights.atk_pieces_on_edges_weight;
    //current_config.atk_pieces_diag_to_corners_weight += alpha*AI_plus_score*delta_weights.atk_pieces_diag_to_corners_weight;
    //current_config.atk_pieces_next_to_corners_weight += alpha*AI_plus_score*delta_weights.atk_pieces_next_to_corners_weight;
    //current_config.def_pieces_next_to_corners_weight += alpha*AI_plus_score*delta_weights.def_pieces_next_to_corners_weight;
}


void SPSA_optimization(){
    int Niter = 10000;
    HeuristicsConfig current_config;
    HeuristicsConfig initial_config;
    vector<HeuristicsConfig> all_configs;
    all_configs.reserve(Niter+1);
    all_configs.push_back(current_config);
    ofstream myfile;
    myfile.open("data/SPSA_results_i1000_d2_fixed_2params.txt");
    double alpha, sigma;
    for(int i=0; i<Niter; i++){
        cout << i << " / " << "1000" << endl;
        if(i < 900){
            alpha = 10.0 - i/125.0;
            sigma = 0.1 - i/11250.0;
        }
        else{
            alpha = 2.0;
            sigma = 0.02;
        }
        SPSA_update_parameters(current_config, all_configs, alpha, sigma);
        all_configs.push_back(current_config);
        // cout << "def_pieces_weight:                 " << current_config.def_pieces_weight << endl;
        // cout << "atk_pieces_weight:                 " << current_config.atk_pieces_weight << endl;
        // cout << "king_free_moves_weight:            " << current_config.king_free_moves_weight << endl;
        // cout << "king_neighboring_enemies_weight:   " << current_config.king_neighboring_enemies_weight << endl;
        // cout << "atk_pieces_on_edges_weight:        " << current_config.atk_pieces_on_edges_weight << endl;
        // cout << "atk_pieces_diag_to_corners_weight: " << current_config.atk_pieces_diag_to_corners_weight << endl;
        // cout << "atk_pieces_next_to_corners_weight: " << current_config.atk_pieces_next_to_corners_weight << endl;
        // cout << "def_pieces_next_to_corners_weight: " << current_config.def_pieces_next_to_corners_weight << endl;
        myfile << current_config.atk_pieces_weight << " ";
        myfile << current_config.def_pieces_weight << " ";
        myfile << current_config.king_free_moves_weight << " ";
        myfile << current_config.king_neighboring_enemies_weight << " ";
        myfile << current_config.king_neighboring_allies_weight << " ";
        myfile << current_config.atk_pieces_on_edges_weight << " ";
        myfile << current_config.atk_pieces_diag_to_corners_weight << " ";
        myfile << current_config.atk_pieces_next_to_corners_weight << " ";
        myfile << current_config.def_pieces_next_to_corners_weight << " ";
        myfile << endl;

        // modified_AI_vs_AI_tournament(1000, 2, 2, &current_config, &initial_config, true, false);
    }
    myfile.close();



    HeuristicsConfig pieces_only_config;
    pieces_only_config.atk_pieces_weight = 1.0;
    pieces_only_config.def_pieces_weight = 1.0;
    pieces_only_config.king_free_moves_weight = 0.0;
    pieces_only_config.king_neighboring_enemies_weight = 0.0;
    pieces_only_config.king_neighboring_allies_weight = 0.0;
    pieces_only_config.atk_pieces_on_edges_weight = 0.0;
    pieces_only_config.atk_pieces_diag_to_corners_weight = 0.0;
    pieces_only_config.atk_pieces_next_to_corners_weight = 0.0;
    pieces_only_config.def_pieces_next_to_corners_weight = 0.0;
    
    cout << "Depth 2 vs initial:" << endl;
    AI_vs_AI_tournament(1000, 2, 2, &current_config, &initial_config, true, false);
    cout << "Depth 2 vs naive:" << endl;
    AI_vs_AI_tournament(1000, 2, 2, &current_config, &pieces_only_config, true, false);
    cout << "Depth 2 init vs naive:" << endl;
    AI_vs_AI_tournament(1000, 2, 2, &initial_config, &pieces_only_config, true, false);

    cout << "(mod) Depth 2 vs initial:" << endl;
    modified_AI_vs_AI_tournament(1000, 2, 2, &current_config, &initial_config, true, false);
    cout << "(mod) Depth 2 vs naive:" << endl;
    modified_AI_vs_AI_tournament(1000, 2, 2, &current_config, &pieces_only_config, true, false);
    cout << "(mod) Depth 2 init vs naive:" << endl;
    modified_AI_vs_AI_tournament(1000, 2, 2, &initial_config, &pieces_only_config, true, false);

    cout << "Depth 3 vs initial:" << endl;
    AI_vs_AI_tournament(1000, 3, 3, &current_config, &initial_config, true, false);
    cout << "Depth 3 vs naive:" << endl;
    AI_vs_AI_tournament(1000, 3, 3, &current_config, &pieces_only_config, true, false);
    cout << "Depth 3 init vs naive:" << endl;
    AI_vs_AI_tournament(1000, 3, 3, &initial_config, &pieces_only_config, true, false);

    cout << "(mod) Depth 3 vs initial:" << endl;
    modified_AI_vs_AI_tournament(1000, 3, 3, &current_config, &initial_config, true, false);
    cout << "(mod) Depth 3 vs naive:" << endl;
    modified_AI_vs_AI_tournament(1000, 3, 3, &current_config, &pieces_only_config, true, false);
    cout << "(mod) Depth 3 init vs naive:" << endl;
    modified_AI_vs_AI_tournament(1000, 3, 3, &initial_config, &pieces_only_config, true, false);

    // cout << "Depth 4 vs initial:" << endl;
    // AI_vs_AI_tournament(1000, 4, 4, &current_config, &initial_config, true, false);
    // cout << "Depth 4 vs naive:" << endl;
    // AI_vs_AI_tournament(1000, 4, 4, &current_config, &pieces_only_config, true, false);
}


void grid_search_optimization(){
    HeuristicsConfig initial_config;
    vector<HeuristicsConfig> all_configs;
    vector<float> AI_scores;

    ofstream myfile;
    myfile.open("data/grid_results_20x20.txt");
    for(int i=0; i<20; i++){
        for(int j=0; j<20; j++){
            HeuristicsConfig current_config;
            current_config.def_pieces_weight = 0.0 + i*0.1;
            current_config.king_free_moves_weight = 0.0 + i*0.1;
            all_configs.push_back(current_config);
    
            float AI_score = modified_AI_vs_AI_tournament(4000, 2, 2, &current_config, &initial_config, false, false);
            AI_scores.push_back(AI_score);

            myfile << AI_score << " ";
            myfile << current_config.atk_pieces_weight << " ";
            myfile << current_config.def_pieces_weight << " ";
            myfile << current_config.king_free_moves_weight << " ";
            myfile << current_config.king_neighboring_enemies_weight << " ";
            myfile << current_config.king_neighboring_allies_weight << " ";
            myfile << current_config.atk_pieces_on_edges_weight << " ";
            myfile << current_config.atk_pieces_diag_to_corners_weight << " ";
            myfile << current_config.atk_pieces_next_to_corners_weight << " ";
            myfile << current_config.def_pieces_next_to_corners_weight << " ";
            myfile << endl;
        }
    }
}



int main(){
    uint64_t initial_atk_bb = 0x8080063000808;
    uint64_t initial_def_bb = 0x814080000;
    uint64_t initial_king_bb = 0x8000000;

    vector<vector<uint64_t>> test_atk_board = {
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 1, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0},
        {1, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    vector<vector<uint64_t>> test_def_board = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    vector<vector<uint64_t>> test_king_board = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    uint64_t test_atk_bb = board2bits(test_atk_board);
    uint64_t test_def_bb = board2bits(test_def_board);
    uint64_t test_king_bb = board2bits(test_king_board);

    // SPSA_optimization();
    grid_search_optimization();

    // print_bitgame(test_atk_bb, test_def_bb, test_king_bb);
    // cout << "Pieces:         " << board_heuristic_pieces_only(test_atk_bb, test_def_bb, test_king_bb) << endl;
    // cout << "King freedom:   " << board_heuristic_king_free_moves(test_atk_bb, test_def_bb, test_king_bb) << endl;
    // cout << "King neighbors: " << board_heuristic_king_neighboring_enemies(test_atk_bb, test_def_bb, test_king_bb) << endl;
    // cout << "Atk on corners: " << board_heuristic_attacker_on_corners(test_atk_bb, test_def_bb, test_king_bb) << endl;
    // cout << "Atk on edges:   " << board_heuristic_attacker_on_edges(test_atk_bb, test_def_bb, test_king_bb) << endl;

    // print_bitgame(initial_atk_bb, initial_def_bb, initial_king_bb);
    // cout << "Pieces:         " <<  board_heuristic_pieces_only(initial_atk_bb, initial_def_bb, initial_king_bb) << endl;
    // cout << "King freedom:   " <<  board_heuristic_king_free_moves(initial_atk_bb, initial_def_bb, initial_king_bb) << endl;
    // cout << "King neighbors: " <<  board_heuristic_king_neighboring_enemies(initial_atk_bb, initial_def_bb, initial_king_bb) << endl;
    // cout << "Atk on corners: " <<  board_heuristic_attacker_on_corners(initial_atk_bb, initial_def_bb, initial_king_bb) << endl;
    // cout << "Atk on edges:   " <<  board_heuristic_attacker_on_edges(initial_atk_bb, initial_def_bb, initial_king_bb) << endl;

    // initializeZobristTable();
    // float score = 0;
    // AI_alpha_beta_get_move(&score, test_atk_bb, test_def_bb, test_king_bb, 1, 5, true, board_heuristic_v7);

    // uint64_t hash = computeHash(initial_atk_bb, initial_def_bb, initial_king_bb, true);
    // uint64_t new_atk_bb, new_def_bb, new_king_bb;
    // new_atk_bb = (initial_atk_bb ^ 1ULL << 3);
    // new_atk_bb = (new_atk_bb ^ 1ULL << 4);
    // new_def_bb = initial_def_bb;
    // new_king_bb = initial_king_bb;
    // uint64_t new_hash_1 = updateHash(hash, new_atk_bb, new_def_bb, new_king_bb, (1ULL << 3), (1ULL << 4));
    // uint64_t new_hash_2 = computeHash(new_atk_bb, new_def_bb, new_king_bb, false);
    // cout << hash << " " << new_hash_1 << " " << new_hash_2 << endl;

    // uint64_t hash = computeHash(test_atk_bb, test_def_bb, test_king_bb, true);
    // float score = get_board_score_by_width_search_zobrist(test_atk_bb, test_def_bb, test_king_bb, 1, 1, 6, hash, board_heuristic_pieces_only);
    // float score = get_board_score_by_width_search(test_atk_bb, test_def_bb, test_king_bb, 1, 1, 6, board_heuristic_pieces_only);
    // float score = get_board_score_by_alpha_beta_search(test_atk_bb, test_def_bb, test_king_bb, 1, 1, 6+1, -INFINITY, INFINITY, board_heuristic_v5);
    // cout << score << endl;

    // cout << "Zobrist times read:    " << ZOBRIST_NUM_TIMES_READ << endl;
    // cout << "Zobrist times updated: " << ZOBRIST_NUM_TIMES_UPDATED << endl;
    // cout << "Zobrist times written: " << ZOBRIST_NUM_TIMES_WRITTEN << endl;
    // cout << "Zobrist times missed:  " << ZOBRIST_NUM_TIMES_MISSED << endl;

    // uint64_t flipped_bb;
    // print_bitboard(test_atk_bb);
    // flipped_bb = anti_transpose_bb(test_atk_bb);
    // print_bitboard(flipped_bb);

    // vector<uint64_t> all_boards = get_all_board_symetries(test_atk_bb);
    // for(int i=0; i<8; i++){
    //     print_bitboard(all_boards[i]);
    // }



    // cout << "same heuristics" << endl;
    // AI_vs_AI_tournament(100, board_heuristic_pieces_only, board_heuristic_pieces_only, true);

    // cout << "v1" << endl;
    // AI_vs_AI_tournament(1000, board_heuristic_v1, board_heuristic_pieces_only, true);

    // cout << "v2" << endl;
    // AI_vs_AI_tournament(1000, board_heuristic_v2, board_heuristic_pieces_only, false);

    // cout << "v3" << endl;
    // AI_vs_AI_tournament(1000, board_heuristic_v3, board_heuristic_pieces_only, false);

    // cout << "v4" << endl;
    // AI_vs_AI_tournament(1000, board_heuristic_v4, board_heuristic_pieces_only, false);

    // cout << "v5" << endl;
    // AI_vs_AI_tournament(1000, board_heuristic_v5, board_heuristic_pieces_only, true);

    // cout << "v6 vs v5" << endl;

    // HeuristicsConfig config;
    // modified_AI_vs_AI_tournament(3, 2, 2, &config, &config, false, false);


    // float score = get_board_score_by_alpha_beta_search(test_atk_bb, test_def_bb, test_king_bb, 1, 0, 6, -INFINITY, INFINITY, &config);
    // cout << score << endl;

    // float eval;
    // vector<uint64_t> move = AI_alpha_beta_get_move(&eval, test_atk_bb, test_def_bb, test_king_bb, 1, 6, true, combined_board_heuristics, &config);
    // cout << eval << endl;

    // for(int i=0; i<12; i++){
    //     cout << i << " " << NUM_NODES[i] << endl;
    // }
}