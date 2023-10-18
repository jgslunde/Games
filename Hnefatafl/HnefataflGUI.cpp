#include "HnefataflGUI.h"
#include <QGridLayout>

HnefataflGui::HnefataflGui(QWidget *parent) : QWidget(parent), selectedSquare(-1, -1) {
    createBoardButtons();
    updateBoardDisplay();
}

HnefataflGui::~HnefataflGui() {
    // Qt will take care of deleting the buttons
}

void HnefataflGui::createBoardButtons() {
    QGridLayout* layout = new QGridLayout(this);
    for(int i = 0; i < 11; i++) {
        for(int j = 0; j < 11; j++) {
            squares[i][j] = new QPushButton(this);
            squares[i][j]->setFixedSize(50, 50);
            connect(squares[i][j], &QPushButton::clicked, [=] { handleSquareClick(i, j); });
            layout->addWidget(squares[i][j], i, j);
        }
    }
    setLayout(layout);
}

void HnefataflGui::handleSquareClick(int x, int y) {
    if(selectedSquare.first == -1) {
        selectedSquare = {x, y};
        squares[x][y]->setStyleSheet("background-color: yellow; font-size: 48px;");
    } else {
        // Move piece and update board display
        board.makeMove(board.getCurrentPlayer(), selectedSquare.first, selectedSquare.second, x, y);
        selectedSquare = {-1, -1};
        updateBoardDisplay();
    }
}

void HnefataflGui::updateBoardDisplay() {
    squares[5][5]->setText("○");
    for(int i = 0; i < 11; i++) {
        for(int j = 0; j < 11; j++) {
            switch(board.getPieceAt(i, j)) {
                case HnefataflBoard::EMPTY:
                    if ((i == 5 && j == 5)){  // Center square
                        squares[i][j]->setText("○"); // Thin circle for throne
                    }else if(
                        (i == 0 && j == 0) ||  // Top-left corner
                        (i == 0 && j == 10) || // Top-right corner
                        (i == 10 && j == 0) || // Bottom-left corner
                        (i == 10 && j == 10))  // Bottom-right corner
                    {
                        squares[i][j]->setText("·");
                    } else {
                        squares[i][j]->setText("");
                    }
                    break;
                case HnefataflBoard::ATTACKER: squares[i][j]->setText("♖"); break; // ⚫
                case HnefataflBoard::DEFENDER: squares[i][j]->setText("♜"); break; // ⚪
                case HnefataflBoard::KING: squares[i][j]->setText("♚"); break;
            }
            squares[i][j]->setStyleSheet(""); // Reset the background color
            squares[i][j]->setStyleSheet("font-size: 48px;");
        }
    }
}
