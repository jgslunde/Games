#include "HnefataflGame.h"
#include <iostream>

HnefataflGame::HnefataflGame() : board(std::make_unique<HnefataflBoard>()) {}

void HnefataflGame::start() {
    chooseMode();

    bool gameRunning = true;
    HnefataflBoard::Player currentPlayer = HnefataflBoard::PLAYER_ATTACKER;

    while (gameRunning) {
        board->printBoard();

        if ((gameMode == Mode::PVP) || 
            (gameMode == Mode::PVA && currentPlayer == HnefataflBoard::PLAYER_ATTACKER) || 
            (gameMode == Mode::AVP && currentPlayer == HnefataflBoard::PLAYER_DEFENDER)) {
            gameRunning = promptForMove(currentPlayer);
        } else {
            gameRunning = board->performRandomMove(currentPlayer);
        }

        if (board->checkForWin()) {
            board->printBoard();
            std::cout << (currentPlayer == HnefataflBoard::PLAYER_ATTACKER ? "Attackers" : "Defenders") << " win!\n";
            gameRunning = false;
        }

        currentPlayer = (currentPlayer == HnefataflBoard::PLAYER_ATTACKER) ? HnefataflBoard::PLAYER_DEFENDER : HnefataflBoard::PLAYER_ATTACKER;
    }

    // Add logic to ask if they want to restart or quit.
}

void HnefataflGame::chooseMode() {
    int choice = 0;
    std::cout << "Choose game mode:\n";
    std::cout << "1. Player vs Player\n";
    std::cout << "2. Player vs AI\n";
    std::cout << "3. AI vs Player\n";
    std::cout << "4. AI vs AI\n";
    std::cin >> choice;

    switch (choice) {
    case 1: gameMode = Mode::PVP; break;
    case 2: gameMode = Mode::PVA; break;
    case 3: gameMode = Mode::AVP; break;
    case 4: gameMode = Mode::AVA; break;
    default: 
        std::cout << "Invalid choice, defaulting to Player vs Player.\n";
        gameMode = Mode::PVP;
        break;
    }
}

bool HnefataflGame::promptForMove(HnefataflBoard::Player player) {
    std::string startSquare, endSquare;
    bool valid_move = false;
    while(! valid_move){
        std::cout << (player == HnefataflBoard::PLAYER_ATTACKER ? "Attackers" : "Defenders") << ", enter your move (e.g., 'A1 A2'): ";
        std::cin >> startSquare >> endSquare;

        int startX, startY, endX, endY;
        if (!board->parseSquare(startSquare, startX, startY) || !board->parseSquare(endSquare, endX, endY)) {
            std::cout << "Invalid move. Please try again.\n";
            // return true;  // continue game
        }
        else{
            if (!board->makeMove(player, startX, startY, endX, endY)) {
                std::cout << "Invalid move. Please try again.\n";
                // return true;  // continue game
            }
            else{
                valid_move = true;
            }
        }
    }

    return true;  // continue game
}

int main() {
    HnefataflGame game;
    game.start();
    return 0;
}
