#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <random>
#include <thread>
#include <map>
#include <random>

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
const uint64_t fouredgesides_bb = 0x80022000800;

vector<int> NUM_NODES(12);
map<uint64_t, int> board_counts;

uint64_t ZOBRIST_NUM_TIMES_READ = 0;
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

uint64_t computeHash(const uint64_t attackers, const uint64_t defenders, const uint64_t king, bool isAttackersTurn) {
    uint64_t hash = 0;

    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            int pos = i * 8 + j;
            if (attackers & (1ULL << pos)) {
                hash ^= zobristTable[i][j][0];
            } else if (defenders & (1ULL << pos)) {
                hash ^= zobristTable[i][j][1];
            } else if (king & (1ULL << pos)) {
                hash ^= zobristTable[i][j][2];
            }
        }
    }

    if (isAttackersTurn) {
        hash ^= zobristTurn;
    }

    return hash;
}


const size_t TABLE_SIZE = 1 << 26;
struct TTEntry {
    uint64_t hash;
    double eval;  // example data
    // Add other game-related data as needed.
};
TTEntry transpositionTable[TABLE_SIZE];

void storeInTable(uint64_t hash, double eval) {
    ZOBRIST_NUM_TIMES_WRITTEN++;
    size_t index = hash & (TABLE_SIZE - 1);
    transpositionTable[index].hash = hash;
    transpositionTable[index].eval = eval;
}

double* retrieveFromTable(uint64_t hash) {
    size_t index = hash & (TABLE_SIZE - 1);
    // cout << hash << " " << transpositionTable[index].hash << endl;
    if (transpositionTable[index].hash == hash) {
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


inline vector<uint64_t> get_legal_moves_as_vector(uint64_t piece_bb, uint64_t blocker_bb){
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


double board_heuristic_pieces_only(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  __builtin_popcountll(atk_bb)   // Num of attacking pieces.
            - 2.0*__builtin_popcountll(def_bb);   // Number of defending pieces times two (they are half as many).
}


double board_heuristic_king_free_moves(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    uint64_t blocker_bb = atk_bb | def_bb | edge_bb;
    double score = 0;
    for(int i=0; i<6; i++){
        score -= (double) 0.01*(((king_bb<<i) & (~blocker_bb)) != 0)
                        + (((king_bb>>i) & (~blocker_bb))  != 0)
                        + (((king_bb<<i*8) & (~blocker_bb)) != 0)
                        + (((king_bb>>i*8) & (~blocker_bb)) != 0);
    }
    return score;
}

inline double board_heuristic_king_neighboring_enemies(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return 0.3*(((king_bb<<1 & atk_bb) != 0)
                + ((king_bb>>1 & atk_bb) != 0)
                + ((king_bb<<8 & atk_bb) != 0)
                + ((king_bb>>8 & atk_bb) != 0));
}


inline double board_heuristic_v1(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  __builtin_popcountll(atk_bb)   // Num of attacking pieces.
            - 2.0*__builtin_popcountll(def_bb)   // Number of defending pieces times two (they are half as many).
            + 0.1*__builtin_popcountll(diag2corner_bb & atk_bb);
}


inline double board_heuristic_v2(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  board_heuristic_v1(atk_bb, def_bb, king_bb);
            + 0.05*__builtin_popcountll(fouredgesides_bb & atk_bb);
}

inline double board_heuristic_v3(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return board_heuristic_v2(atk_bb, def_bb, king_bb) + board_heuristic_king_free_moves(atk_bb, def_bb, king_bb);
}


inline double board_heuristic_v4(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return board_heuristic_v3(atk_bb, def_bb, king_bb) + board_heuristic_king_neighboring_enemies(atk_bb, def_bb, king_bb);
}


inline double board_heuristic_v5(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    return  __builtin_popcountll(atk_bb)   // Num of attacking pieces.
            - 1.5*__builtin_popcountll(def_bb)   // Number of defending pieces times two (they are half as many).
            + 0.1*__builtin_popcountll(diag2corner_bb & atk_bb)
            + 0.05*__builtin_popcountll(fouredgesides_bb & atk_bb)
            + board_heuristic_king_free_moves(atk_bb, def_bb, king_bb)
            + board_heuristic_king_neighboring_enemies(atk_bb, def_bb, king_bb);
}


inline double get_board_score(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb){
    // 106 instructions.
    uint64_t king_hostile_bb = atk_bb | corner_bb;
    double score = 
            - 1000.0*(__builtin_popcountll(atk_bb) == 0)   // Defender wins if attacker is out of pieces.
            - 1000.0*((king_bb & corner_bb) > 0)   // Defender wins if the king reaces a corner.
            + 1000.0*(!king_bb);   // Attacker wins if king is captured, and therefore not on the board.

    return score;
}


inline uint64_t perform_captures(uint64_t moved_piece_bb, uint64_t allied_pieces_bb, uint64_t enemy_pices_bb){
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


inline void make_move_on_board(uint64_t &atk_bb, uint64_t &def_bb, uint64_t &king_bb, uint64_t move_from, uint64_t move_to){
    // 95 instructions, including calls to "perform_captures".
    if(atk_bb & move_from){
        atk_bb ^= (move_from | move_to);
        uint64_t captures = perform_captures(move_to, atk_bb | corner_bb, def_bb | king_bb);
        def_bb &= ~captures;
        king_bb &= ~captures;
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


inline vector<uint64_t> get_all_legal_moves_as_vector(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player){
    // 284 instructions, including calls to "get_legal_moves_as_vector".
    vector<uint64_t> legal_moves;
    legal_moves.reserve(100);
    uint64_t piece_bb;
    if(player == PLAYER_ATK){
        // Attacker moves.
        for(int imove=0; imove<64; imove++){
            piece_bb = (uint64_t) 1 << imove;
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



inline double get_board_score_by_width_search_zobrist(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, int depth, int max_depth, double (*heuristic_function)(uint64_t, uint64_t, uint64_t)){
    NUM_NODES[depth] += 1;
    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size()/2;
    vector<double> move_scores(num_legal_moves);
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
        double* zobrist_score_ptr = retrieveFromTable(hash);
        if(zobrist_score_ptr){  // If we didn't get a nullptr in return, the board position already exists in our table.
            // We then therminate the entire rest of the search down this branch, and set the score:
            // cout << "hello" << endl;
            move_scores[imove] = *zobrist_score_ptr;
        }
        else{
            // If not, we will have to calculate the board score:
            double move_score = get_board_score(atk_bb_new, def_bb_new, king_bb_new) + heuristic_function(atk_bb_new, def_bb_new, king_bb_new);
            if((depth >= max_depth) || abs(move_score) > 900){  // If we're at max depth, or someone has won, we terminate.
                move_scores[imove] = (1 - 0.01*depth)*move_score;
            }
            else{  // If nobody has won and we're not at max depth, we continue down the tree:
                move_scores[imove] = get_board_score_by_width_search_zobrist(atk_bb_new, def_bb_new, king_bb_new, -player, depth+1, max_depth, heuristic_function);
            }
        storeInTable(hash, move_scores[imove]);
        }
    }

    // int best_board_score;
    if(player == PLAYER_ATK){
        double best_board_score = -999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] > best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
    else{
        double best_board_score = 999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] < best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
}


inline double get_board_score_by_width_search(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, int depth, int max_depth, double (*heuristic_function)(uint64_t, uint64_t, uint64_t)){
    NUM_NODES[depth] += 1;
    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size()/2;
    vector<double> move_scores(num_legal_moves);
    for(int imove=0; imove<num_legal_moves; imove++){
        uint64_t atk_bb_new = atk_bb;
        uint64_t def_bb_new = def_bb;
        uint64_t king_bb_new = king_bb;
        make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*imove], legal_moves[2*imove+1]);
        // If not, we will have to calculate the board score:
        double move_score = get_board_score(atk_bb_new, def_bb_new, king_bb_new) + heuristic_function(atk_bb_new, def_bb_new, king_bb_new);
        if((depth >= max_depth) || abs(move_score) > 900){  // If we're at max depth, or someone has won, we terminate.
            move_scores[imove] = (1 - 0.01*depth)*move_score;
        }
        else{  // If nobody has won and we're not at max depth, we continue down the tree:
            move_scores[imove] = get_board_score_by_width_search(atk_bb_new, def_bb_new, king_bb_new, -player, depth+1, max_depth, heuristic_function);
        }
    }

    // int best_board_score;
    if(player == PLAYER_ATK){
        double best_board_score = -999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] > best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
    else{
        double best_board_score = 999999;
        for(int imove=0; imove<num_legal_moves; imove++){
            if(move_scores[imove] < best_board_score){
                best_board_score = move_scores[imove];
            }
        }
    return best_board_score;
    }
}



inline vector<uint64_t> AI_1_get_move(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb, int player, int max_depth, bool verbose, double (*heuristic_function)(uint64_t, uint64_t, uint64_t)){
    vector<uint64_t> legal_moves = get_all_legal_moves_as_vector(atk_bb, def_bb, king_bb, player);
    int num_legal_moves = legal_moves.size()/2;
    int preffered_move_idx = 0;
    vector<double> move_scores(num_legal_moves);

    #pragma omp parallel for
    for(int i=0; i<num_legal_moves; i++){
        uint64_t atk_bb_new, def_bb_new, king_bb_new;
        atk_bb_new = atk_bb;
        def_bb_new = def_bb;
        king_bb_new = king_bb;
        make_move_on_board(atk_bb_new, def_bb_new, king_bb_new, legal_moves[2*i], legal_moves[2*i+1]);
        double move_score;
        if(player==1)
            move_score = get_board_score_by_width_search(atk_bb_new, def_bb_new, king_bb_new, -player, 2, max_depth, heuristic_function);
        else
            move_score = get_board_score_by_width_search_zobrist(atk_bb_new, def_bb_new, king_bb_new, -player, 2, max_depth, heuristic_function);
        move_scores[i] = move_score;
    }

    // Finding the preffered (highest or lowest, depending on player) score among the options.
    double best_move_score = -9999999.0*player;
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
    uint chosen_move_index = preffered_move_indices[thread_safe_rand()%preffered_move_indices.size()];

    if(verbose){
        cout << "Player: " << player << ". Board eval: " << best_move_score << endl;
    }
    return {legal_moves[2*chosen_move_index], legal_moves[2*chosen_move_index+1]};
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


void AI_vs_AI_tournament(int num_games, double (*AI_1_heuristic_function)(uint64_t, uint64_t, uint64_t), double (*AI_2_heuristic_function)(uint64_t, uint64_t, uint64_t), bool verbose){
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
        if(verbose)
            cout << "Game: " << game << endl;
        uint64_t atk_bb = initial_atk_bb;
        uint64_t def_bb = initial_def_bb;
        uint64_t king_bb = initial_king_bb;
        int current_player = 1;
        int score = 0;
        int depth = 4;
        int iturn = 0;
        while (true){
            if(current_player == 1){
                depth=4;
            }else{
                depth=4;
            }
            vector<uint64_t> preffered_move;
            if(current_player*AI_1_playing_as == 1)
                preffered_move = AI_1_get_move(atk_bb, def_bb, king_bb, current_player, depth, false, AI_1_heuristic_function);
            else
                preffered_move = AI_1_get_move(atk_bb, def_bb, king_bb, current_player, depth, false, AI_2_heuristic_function);
            make_move_on_board(atk_bb, def_bb, king_bb, preffered_move[0], preffered_move[1]);
            score = get_board_score(atk_bb, def_bb, king_bb) + board_heuristic_pieces_only(atk_bb, def_bb, king_bb);
            if(abs(score) > 800){
                if(score < 0){
                    if(AI_1_playing_as == -1)
                        num_AI_1_wins_def++;
                    else
                        num_AI_2_wins_def++;
                    if(verbose){
                        cout << "DEFENDER WINS" << endl;
                        print_bitgame(atk_bb, def_bb, king_bb);
                    }
                }else{
                    if(AI_1_playing_as == 1)
                        num_AI_1_wins_atk++;
                    else
                        num_AI_2_wins_atk++;
                    if(verbose){
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
                if(verbose){
                    cout << "100 TURNS REACHED" << endl;
                    print_bitgame(atk_bb, def_bb, king_bb);
                }
                break;
            }
        }
        AI_1_playing_as *= -1;
    }
    cout << "Total number of atk wins for AI_1: " << num_AI_1_wins_atk << endl;
    cout << "Total number of atk wins for AI_2: " << num_AI_2_wins_atk << endl;
    cout << "Total number of def wins for AI_1: " << num_AI_1_wins_def << endl;
    cout << "Total number of def wins for AI_2: " << num_AI_2_wins_def << endl;
    cout << "Total number of draws with AI_1 as atk: " << num_AI_1_ties_atk << endl;
    cout << "Total number of draws with AI_1 as def: " << num_AI_1_ties_def << endl;
    cout << "AI_1/AI_2 win ratio as atk: " << (double) num_AI_1_wins_atk / (double)num_AI_2_wins_atk << endl;
    cout << "AI_1/AI_2 win ratio as def: " << (double) num_AI_1_wins_def / (double)num_AI_2_wins_def << endl;
}



int main(){
    uint64_t initial_atk_bb = 0x8080063000808;
    uint64_t initial_def_bb = 0x814080000;
    uint64_t initial_king_bb = 0x8000000;

    vector<vector<uint64_t>> test_atk_board = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 1, 0},
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

    uint64_t test_atk_bb = board2bits(test_atk_board);
    uint64_t test_def_bb = board2bits(test_def_board);
    uint64_t test_king_bb = board2bits(test_king_board);

    initializeZobristTable();

    double score = get_board_score_by_width_search(initial_atk_bb, initial_def_bb, initial_king_bb, 1, 1, 10, board_heuristic_pieces_only);
    cout << score << endl;

    for(int i=0; i<12; i++){
        cout << i << " " << NUM_NODES[i] << endl;
    }
    cout << "Zobrist times read: " << ZOBRIST_NUM_TIMES_READ << endl;
    cout << "Zobrist times written: " << ZOBRIST_NUM_TIMES_WRITTEN << endl;
    cout << "Zobrist times missed: " << ZOBRIST_NUM_TIMES_MISSED << endl;

    // uint64_t flipped_bb;
    // print_bitboard(test_atk_bb);
    // flipped_bb = anti_transpose_bb(test_atk_bb);
    // print_bitboard(flipped_bb);

    // vector<uint64_t> all_boards = get_all_board_symetries(test_atk_bb);
    // for(int i=0; i<8; i++){
    //     print_bitboard(all_boards[i]);
    // }

    cout << "same heuristics" << endl;
    AI_vs_AI_tournament(1, board_heuristic_pieces_only, board_heuristic_pieces_only, true);

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

}