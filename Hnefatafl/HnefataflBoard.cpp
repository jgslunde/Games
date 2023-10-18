#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include "HnefataflBoard.h"

HnefataflBoard::HnefataflBoard() {
    board.resize(11, std::vector<Piece>(11, EMPTY));
    resetBoard();
}

void HnefataflBoard::resetBoard() {
    std::vector<std::vector<Piece>> initialPosition = {
        {EMPTY,    EMPTY,    EMPTY,     ATTACKER,  ATTACKER,  ATTACKER,  ATTACKER,  ATTACKER,  EMPTY,   EMPTY,     EMPTY},
        {EMPTY,    EMPTY,    EMPTY,     EMPTY,     EMPTY,     ATTACKER,  EMPTY,     EMPTY,     EMPTY,   EMPTY,     EMPTY},
        {EMPTY,    EMPTY,    EMPTY,     EMPTY,     EMPTY,     EMPTY,     EMPTY,     EMPTY,     EMPTY,   EMPTY,     EMPTY},
        {ATTACKER, EMPTY,    EMPTY,     EMPTY,     EMPTY,     DEFENDER,  EMPTY,     EMPTY,     EMPTY,   EMPTY,     ATTACKER},
        {ATTACKER, EMPTY,    EMPTY,     EMPTY,     DEFENDER,  DEFENDER,  DEFENDER,  EMPTY,     EMPTY,   EMPTY,     ATTACKER},
        {ATTACKER, ATTACKER, EMPTY,     DEFENDER,  DEFENDER,  KING,      DEFENDER,  DEFENDER,  EMPTY,   ATTACKER,  ATTACKER},
        {ATTACKER, EMPTY,    EMPTY,     EMPTY,     DEFENDER,  DEFENDER,  DEFENDER,  EMPTY,     EMPTY,   EMPTY,     ATTACKER},
        {ATTACKER, EMPTY,    EMPTY,     EMPTY,     EMPTY,     DEFENDER,  EMPTY,     EMPTY,     EMPTY,   EMPTY,     ATTACKER},
        {EMPTY,    EMPTY,    EMPTY,     EMPTY,     EMPTY,     EMPTY,     EMPTY,     EMPTY,     EMPTY,   EMPTY,     EMPTY},
        {EMPTY,    EMPTY,    EMPTY,     EMPTY,     EMPTY,     ATTACKER,  EMPTY,     EMPTY,     EMPTY,   EMPTY,     EMPTY},
        {EMPTY,    EMPTY,    EMPTY,     ATTACKER,  ATTACKER,  ATTACKER,  ATTACKER,  ATTACKER,  EMPTY,   EMPTY,     EMPTY}
    };

    board = initialPosition;
    currentPlayer = PLAYER_ATTACKER;
    turnNum = 1;
    srand(time(0));
}



void HnefataflBoard::printBoard() const {
    int BOARD_SIZE = 11;
    const std::string RESET = "\033[0m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string BOLD_BLUE = "\033[1;34m";

    std::cout << "Turn number: " << HnefataflBoard::turnNum << std::endl;
    // Print the column labels (A-K) with extra spacing
    std::cout << "   ";
    for (char colLabel = 'A'; colLabel <= 'K'; ++colLabel) {
        std::cout << " " << colLabel << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    for (int i = BOARD_SIZE - 1; i >= 0; i--) { 
        // Print the row labels (1-11) with extra spacing
        if (i < 9) { // Single-digit row numbers
            std::cout << (i + 1) << "  ";
        } else {    // Double-digit row numbers
            std::cout << (i + 1) << " ";
        }

        for (int j = 0; j < BOARD_SIZE; j++) {
            switch (board[i][j]) {
                case EMPTY: std::cout << " . "; break;
                case ATTACKER: std::cout << RED << " A " << RESET; break;
                case DEFENDER: std::cout << GREEN << " D " << RESET; break;
                case KING: std::cout << BOLD_BLUE << " K " << RESET; break;
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "   ";
    for (char colLabel = 'A'; colLabel <= 'K'; ++colLabel) {
        std::cout << " " << colLabel << " ";
    }
    std::cout << std::endl;
}

HnefataflBoard::Player HnefataflBoard::getCurrentPlayer() const {
    return currentPlayer;
}


HnefataflBoard::Move::Move(int sX, int sY, int eX, int eY) 
    : startX(sX), startY(sY), endX(eX), endY(eY) { }

std::vector<HnefataflBoard::Move> HnefataflBoard::generateLegalMoves(HnefataflBoard::Player player) const {
    std::vector<HnefataflBoard::Move> legalMoves;

    for (int i = 0; i < 11; i++) {
        for (int j = 0; j < 11; j++) {
            if ((board[i][j] == ATTACKER && player == PLAYER_ATTACKER) ||
                (board[i][j] == DEFENDER && player == PLAYER_DEFENDER) ||
                (board[i][j] == KING && player == PLAYER_DEFENDER)) {

                // Check in each direction: up, down, left, right
                int dx[] = {0, 0, -1, 1};
                int dy[] = {-1, 1, 0, 0};
                for (int dir = 0; dir < 4; dir++) {
                    int x = i + dx[dir], y = j + dy[dir];
                    while (x >= 0 && x < 11 && y >= 0 && y < 11 && board[x][y] == EMPTY) {
                        legalMoves.emplace_back(i, j, x, y);
                        x += dx[dir];
                        y += dy[dir];
                    }
                }
            }
        }
    }

    return legalMoves;
}

bool HnefataflBoard::makeMove(HnefataflBoard::Player player, int startX, int startY, int endX, int endY) {
    // Check if it's the correct player's turn
    if (player != currentPlayer) {
        std::cout << "It's not this player's turn." << std::endl;
        return false;
    }

    // Check if the starting square contains the player's piece
    if ((player == HnefataflBoard::PLAYER_ATTACKER && board[startX][startY] != HnefataflBoard::ATTACKER) ||
        (player == HnefataflBoard::PLAYER_DEFENDER && (board[startX][startY] != HnefataflBoard::DEFENDER && board[startX][startY] != HnefataflBoard::KING))) {
        std::cout << "The starting square does not contain the player's piece." << std::endl;
        return false;
    }

    if (!isMoveLegal(startX, startY, endX, endY)) {
        std::cout << "Move is diagonal, or passes through a piece. Try again." << std::endl;
        return false;
    }

    // If all checks pass, check if the destination is a corner square and if the moving piece isn't the king
    if ((endX == 0 || endX == 10) && (endY == 0 || endY == 10) && board[startX][startY] != KING) {
        std::cout << "Only the king can move to the corner squares." << std::endl;
        return false;
    }
    // Check if the ending square is empty
    if (board[endX][endY] != HnefataflBoard::EMPTY) {
        std::cout << "The destination square is not empty." << std::endl;
        return false;
    }

    // If all checks pass, move the piece and switch the current player
    board[endX][endY] = board[startX][startY];
    board[startX][startY] = HnefataflBoard::EMPTY;

    std::cout << "Piece moved from " << startX << ", " << startY << " to " << endX << ", " << endY << std::endl;

    evaluateCaptures(endX, endY);

    currentPlayer = (currentPlayer == HnefataflBoard::PLAYER_ATTACKER) ? HnefataflBoard::PLAYER_DEFENDER : HnefataflBoard::PLAYER_ATTACKER;

    HnefataflBoard::turnNum += 1;
    return true;
}


HnefataflBoard::Move HnefataflBoard::getRandomMove(HnefataflBoard::Player player) const {
    std::vector<Move> legalMoves = generateLegalMoves(player);

    if (legalMoves.empty()) {
        throw std::runtime_error("No legal moves available for the player!");
    }

    // Pick and return a random move
    int randomIndex = std::rand() % legalMoves.size();
    return legalMoves[randomIndex];
}

bool HnefataflBoard::performRandomMove(HnefataflBoard::Player player) {
    HnefataflBoard::Move randomMove = getRandomMove(player);
    return makeMove(player, randomMove.startX, randomMove.startY, randomMove.endX, randomMove.endY);
}

bool HnefataflBoard::checkForWin() {
    // Check for the king's victory
    for (int i = 0; i < 11; i++) {
        for (int j = 0; j < 11; j++) {
            if (board[i][j] == HnefataflBoard::KING) {
                if (isKingAtCorner(i, j)) {
                    std::cout << "Defenders win! The king has escaped!" << std::endl;
                    return true;
                }
            }
        }
    }

    // Check for the attackers' victory
    if (isKingCaptured()) {
        std::cout << "Attackers win! The king has been captured!" << std::endl;
        return true;
    }

    return false; // No one has won yet
}


bool HnefataflBoard::parseSquare(const std::string& squareName, int& x, int& y) {
    if (squareName.size() < 2 || squareName.size() > 3) return false;

    char colLetter = std::toupper(squareName[0]);
    if (colLetter < 'A' || colLetter > 'K') return false;

    int rowNum;
    try {
        rowNum = std::stoi(squareName.substr(1));
    } catch (std::exception&) {
        return false;
    }

    if (rowNum < 1 || rowNum > 11) return false;

    y = colLetter - 'A';                // Convert column letter to index
    x = rowNum - 1;              // Convert row number to index

    return true;
}


bool HnefataflBoard::isKingAtCorner(int x, int y) const {
    return (x == 0 || x == 10) && (y == 0 || y == 10);
}

bool HnefataflBoard::isKingCaptured() const {
    for (int i = 0; i < 11; i++) {
        for (int j = 0; j < 11; j++) {
            if (board[i][j] == KING) {
                // Check for top edge or ATTACKER piece above
                bool topHostile = (i == 0) || (board[i - 1][j] == ATTACKER);

                // Check for bottom edge or ATTACKER piece below
                bool bottomHostile = (i == 10) || (board[i + 1][j] == ATTACKER);

                // Check for left edge or ATTACKER piece to the left
                bool leftHostile = (j == 0) || (board[i][j - 1] == ATTACKER);

                // Check for right edge or ATTACKER piece to the right
                bool rightHostile = (j == 10) || (board[i][j + 1] == ATTACKER);

                std::cout << topHostile << " " << bottomHostile << " " << leftHostile << " " << rightHostile << std::endl;
                // If all sides around the king are hostile, then the king is captured
                return topHostile && bottomHostile && leftHostile && rightHostile;
            }
        }
    }
    return false; // This should never be reached, but it's here just in case.
}

HnefataflBoard::Piece HnefataflBoard::getPieceAt(int x, int y) const {
    // You might want to add boundary checks to ensure x and y are valid
    if(x < 0 || x >= board.size() || y < 0 || y >= board[0].size()) {
        // Return a default value or throw an exception
        return EMPTY; // or throw std::out_of_range("Invalid board coordinates");
    }
    return board[x][y];
}


HnefataflBoard::Player currentPlayer = HnefataflBoard::PLAYER_ATTACKER;  // We'll assume the attacker goes first.


bool HnefataflBoard::isMoveLegal(int startX, int startY, int endX, int endY) const {
    if (startX != endX && startY != endY) {
        return false; // Diagonal move is illegal
    }
    
    // Horizontal move
    if (startX == endX) {
        int step = (startY < endY) ? 1 : -1;
        for (int y = startY + step; y != endY; y += step) {
            if (board[startX][y] != HnefataflBoard::EMPTY) {
                return false; // Cannot move past another piece
            }
        }
    }

    // Vertical move
    if (startY == endY) {
        int step = (startX < endX) ? 1 : -1;
        for (int x = startX + step; x != endX; x += step) {
            if (board[x][startY] != HnefataflBoard::EMPTY) {
                return false; // Cannot move past another piece
            }
        }
    }

    return true;
}

void HnefataflBoard::evaluateCaptures(int x, int y) {
    Piece movedPiece = board[x][y];
    Piece opponentPiece = (movedPiece == ATTACKER) ? DEFENDER : ATTACKER;

    // Function to check if the position is hostile
    auto isHostile = [this, movedPiece](int x, int y) {
        return ((x == 0 && y == 0) || (x == 10 && y == 0) || (x == 0 && y == 10) || (x == 10 && y == 10)) || board[x][y] == movedPiece;
    };

    // Check horizontal capture
    if (x > 1 && board[x - 1][y] == opponentPiece && isHostile(x - 2, y)) {
        board[x - 1][y] = EMPTY;
        std::cout << "### Piece captured on " << x - 1 << ", " << y << std::endl;
    }
    if (x < 9 && board[x + 1][y] == opponentPiece && isHostile(x + 2, y)) {
        board[x + 1][y] = EMPTY;
        std::cout << "### Piece captured on " << x + 1 << ", " << y << std::endl;
    }

    // Check vertical capture
    if (y > 1 && board[x][y - 1] == opponentPiece && isHostile(x, y - 2)) {
        board[x][y - 1] = EMPTY;
        std::cout << "### Piece captured on " << x << ", " << y - 1 << std::endl;
    }
    if (y < 9 && board[x][y + 1] == opponentPiece && isHostile(x, y + 2)) {
        board[x][y + 1] = EMPTY;
        std::cout << "### Piece captured on " << x << ", " << y + 1 << std::endl;
    }
}