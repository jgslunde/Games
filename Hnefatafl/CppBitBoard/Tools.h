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
const uint64_t right_eedge_bb = 0x404040404000;
const uint64_t bottom_edge_bb = 0x3e;
const uint64_t left_edge_bb = 0x10101010100;
const uint64_t top_edge_bb = 0x3e000000000000;
const uint64_t corner_neighbors_bb = 0x22410000004122;
const uint64_t right_smalledge_bb = 0x4040400000;
const uint64_t top_smalledge_bb = 0x1c000000000000;
const uint64_t left_smalledge_bb = 0x101010000;
const uint64_t bottom_smalledge_bb = 0x1c;

const char PLAYER_ATK = 1;
const char PLAYER_DEF = -1;


int thread_safe_rand();
float thread_safe_rand_float(float min, float max);
void print_bitgame(uint64_t atk_bb, uint64_t def_bb, uint64_t king_bb);
vector<uint64_t> get_legal_moves_as_vector(uint64_t piece_bb, uint64_t blocker_bb);