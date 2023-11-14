#pragma once

#include <cstdint>
#include <vector>
#include <random>
#include <thread>

using namespace std;

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
const char PLAYER_ATK = 1;
const char PLAYER_DEF = -1;


int thread_safe_rand();
float thread_safe_rand_float(float min, float max);
void print_bitgame(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb);
vector<uint64_t> get_legal_moves_as_vector(uint64_t piece_bb, uint64_t blocker_bb);