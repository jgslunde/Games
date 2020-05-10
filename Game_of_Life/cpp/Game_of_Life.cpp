#include <armadillo>
#include <stdlib.h>
#include <chrono>

arma::Mat<int> make_random_grid(int x_max, int y_max, int seed, bool zero_padding=true){
    srand(seed);
    if(zero_padding){
        arma::Mat<int> grid(x_max+2, y_max+2, arma::fill::zeros);
        for(int x=1; x<x_max+1; x++){
            for(int y=1; y<y_max+1; y++){
                grid(x,y) = rand()%2;
            }
        }
    return grid;
    }
    else{
        arma::Mat<int> grid(x_max, y_max, arma::fill::zeros);
        for(int x=0; x<x_max; x++){
            for(int y=0; y<y_max; y++){
                grid(x,y) = rand()%2;
            }
        }
    return grid;
    }
}


int main(int argc, char **argv){
    auto start = std::chrono::high_resolution_clock::now();

    int x_max = 10;
    int y_max = 10;

    int max_iterations = 1000;
    int nr_games = 100000000;

    // arma::Mat<int> active_cells(nr_games, max_iterations, arma::fill::zeros);
    // arma::Mat<int> alive_cells(nr_games, max_iterations, arma::fill::zeros);
    arma::Col<int> run_lengths(nr_games, arma::fill::zeros);
    arma::Col<int> equality_distances(nr_games, arma::fill::zeros);

    int longest_run = 0;
    int best_game = -1;

    for(int game_nr=0; game_nr<nr_games; game_nr++){
        std::vector<arma::Mat<int>> all_grids;

        arma::Mat<int> start_grid(x_max+2, y_max+2, arma::fill::zeros);
        arma::Mat<int> grid(x_max+2, y_max+2, arma::fill::zeros);
        arma::Mat<int> new_grid(x_max+2, y_max+2, arma::fill::zeros);

        start_grid = make_random_grid(x_max, y_max, game_nr, true);
        grid = start_grid;
        all_grids.push_back(start_grid);

        bool running = true;
        int i = 0;
        while(running){
            for(int j=0; j<i; j++){
                arma::Mat<int> old_grid = all_grids[j];
                if(arma::accu(arma::abs(old_grid - grid)) == 0){
                    // printf("Finished game %d. Old grid %d is equal to new grid %d\n", game_nr, j, i);
                    running = false;
                    run_lengths[game_nr] = i;
                    equality_distances[game_nr] = i - j;
                    if(i > longest_run){
                        longest_run = i;
                        best_game = game_nr;
                        printf("New longest run %d by game %d\n", i, game_nr);
                    }
                }
            }

            for(int x=1; x<x_max+1; x++){
                for(int y=1; y<y_max+1; y++){
                    int neighbors = grid(x-1, y-1) + grid(x, y-1) + grid(x+1, y-1)\
                                + grid(x-1, y)                  + grid(x+1, y)\
                                + grid(x-1, y+1) + grid(x, y+1) + grid(x+1, y+1);
                    if((neighbors == 3) || ((grid(x,y) == 1) && (neighbors == 2))){
                        new_grid(x,y) = 1;
                    }
                    else{
                        new_grid(x,y) = 0;
                    }
                }
            }
            // alive_cells(game_nr, i) = arma::accu(arma::abs(new_grid));
            // active_cells(game_nr, i) = arma::accu(arma::abs(new_grid - grid));
            grid = new_grid;
            all_grids.push_back(new_grid);
            i++;
        }

    }
    printf("Wiring best game %d of length %d to file\n", best_game, longest_run);
    arma::Mat<int> best_grid = make_random_grid(x_max, y_max, best_game, false);
    best_grid.save("../data/best_grid.dat", arma::raw_ascii);
    // active_cells.save("../data/active_cells.dat", arma::raw_ascii);
    // alive_cells.save("../data/alive_cells.dat", arma::raw_ascii);
    run_lengths.save("../data/run_lengths.dat", arma::raw_ascii);
    equality_distances.save("../data/equality_distances.dat", arma::raw_ascii);

    auto stop = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time usage = " << dur.count()/1000.0 << " s." << std::endl;
    return 0;
}