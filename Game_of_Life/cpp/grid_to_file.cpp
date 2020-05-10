#include <armadillo>
#include <stdlib.h>
#include <string.h>

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
    int x_max = atoi(argv[1]);
    int y_max = atoi(argv[2]);

    int seed = atoi(argv[3]);

    arma::Mat<int> grid = make_random_grid(x_max, y_max, seed, false);

    std::string filename = "../data/grid_" + std::to_string(seed) + ".dat";
    grid.save(filename, arma::raw_ascii);

    return 0;
}