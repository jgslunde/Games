#pragma once

#include <iostream>
#include <vector>

class HnefataflBoard {
public:
    enum Piece {
        EMPTY = 0,
        ATTACKER = 1,
        DEFENDER = 2,
        KING = 3
    };

    enum Player {
        PLAYER_ATTACKER = ATTACKER,
        PLAYER_DEFENDER = DEFENDER
    };

    struct Move {
        int startX, startY, endX, endY;
        Move(int sX, int sY, int eX, int eY);
    };

    HnefataflBoard();
    void resetBoard();
    void printBoard() const;
    Player getCurrentPlayer() const;
    std::vector<Move> generateLegalMoves(Player player) const;
    bool parseSquare(const std::string& squareName, int& x, int& y);
    bool makeMove(Player player, int startX, int startY, int endX, int endY);
    Move getRandomMove(Player player) const;
    bool performRandomMove(Player player);
    bool checkForWin();
    Piece getPieceAt(int x, int y) const;

private:
    std::vector<std::vector<Piece>> board;
    Player currentPlayer;
    int turnNum;

    bool isKingAtCorner(int x, int y) const;
    bool isKingCaptured() const;
    bool isMoveLegal(int startX, int startY, int endX, int endY) const;
    void evaluateCaptures(int x, int y);
};