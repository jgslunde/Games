#ifndef HNEFATAFLGUI_H
#define HNEFATAFLGUI_H

#include "HnefataflBoard.h"
#include <QWidget>
#include <QPushButton>

class HnefataflGui : public QWidget {
    Q_OBJECT

public:
    explicit HnefataflGui(QWidget *parent = nullptr);
    ~HnefataflGui();

private slots:
    void handleSquareClick(int x, int y);

private:
    void createBoardButtons();
    void updateBoardDisplay();

    HnefataflBoard board;
    QPushButton* squares[11][11];
    std::pair<int, int> selectedSquare; // (-1, -1) if no square is selected
};

#endif // HNEFATAFLGUI_H
