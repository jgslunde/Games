#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <random>
#include <thread>
#include <map>
#include <random>
#include <fstream>
#include "Tools.h"
#include "Board.h"
#include "Heuristics.h"
#include "AI.h"
#include "Optimization.h"

using namespace std;

int main(){
    HeuristicsConfig config{1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    AI ai(config);
    cout << config.atk_pieces_on_edges_weight << endl;

    Board board;

    // DEPTH 1
    vector<uint64_t>legal_moves1 = board.get_all_legal_moves();
    int num_legal_moves1 = legal_moves1.size()/2;
    cout << num_legal_moves1 << endl;
    ofstream myfile1, myfile2, myfile3, myfile4, myfile5;
    myfile1.open("openings/openings_even_1.txt");
    myfile2.open("openings/openings_even_2.txt");
    myfile3.open("openings/openings_even_3.txt");
    myfile4.open("openings/openings_even_4.txt");
    for(int imove1=0; imove1<num_legal_moves1; imove1++){
        cout << imove1 << "/" << num_legal_moves1 << endl;
        Board board1 = board;
        board1.make_move(legal_moves1[2*imove1], legal_moves1[2*imove1+1]);
        Move move = ai.get_preffered_move(board1, 4);
        if(move.eval == 0){

            myfile1 << board1.atk_bb << " " << board1.def_bb << " " << board1.king_bb << "\n";
            // DEPTH 2
            vector<uint64_t>legal_moves2 = board1.get_all_legal_moves();
            int num_legal_moves2 = legal_moves2.size()/2;
            for(int imove2=0; imove2<num_legal_moves2; imove2++){
                Board board2 = board1;
                board2.make_move(legal_moves2[2*imove2], legal_moves2[2*imove2+1]);
                Move move = ai.get_preffered_move(board2, 4);
                if(move.eval == 0){

                    myfile2 << board2.atk_bb << " " << board2.def_bb << " " << board2.king_bb << "\n";

                    // DEPTH 3
                    vector<uint64_t>legal_moves3 = board1.get_all_legal_moves();
                    int num_legal_moves3 = legal_moves3.size()/2;
                    for(int imove3=0; imove3<num_legal_moves3; imove3++){
                        Board board3 = board2;
                        board3.make_move(legal_moves3[2*imove3], legal_moves3[2*imove3+1]);
                        Move move = ai.get_preffered_move(board3, 4);
                        if(move.eval == 0){

                            myfile3 << board3.atk_bb << " " << board3.def_bb << " " << board3.king_bb << "\n";

                            // DEPTH 4
                            vector<uint64_t>legal_moves4 = board1.get_all_legal_moves();
                            int num_legal_moves4 = legal_moves4.size()/2;
                            for(int imove4=0; imove4<num_legal_moves4; imove4++){
                                Board board4 = board3;
                                board4.make_move(legal_moves4[2*imove4], legal_moves4[2*imove4+1]);
                                Move move = ai.get_preffered_move(board4, 4);
                                if(move.eval == 0){
                                    myfile4 << board4.atk_bb << " " << board4.def_bb << " " << board4.king_bb << "\n";
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    myfile1.close();
    myfile2.close();
    myfile3.close();
}