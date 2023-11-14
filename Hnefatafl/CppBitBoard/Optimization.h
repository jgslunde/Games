#pragma once

#include <vector>
#include <string>
#include "Heuristics.h"

using namespace std;

void grid_search(vector<HeuristicsConfig> search_config_arr, vector<HeuristicsConfig> opponent_config_arr, int num_battles, string outfile);