#pragma once

#include <vector>
#include <string>
#include "Heuristics.h"

using namespace std;

void grid_search(vector<HeuristicsConfig> search_config_arr, vector<HeuristicsConfig> opponent_config_arr, int num_battles, string outfile);
void SPSA_optimization(HeuristicsConfig initial_config, vector<HeuristicsConfig> opponent_config_arr, double alpha, double sigma, int num_iter, int num_battles, string outfile);