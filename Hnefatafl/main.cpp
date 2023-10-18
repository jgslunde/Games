// #include "HnefataflGame.h"
#include "HnefataflGUI.h"
#include <QApplication>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    HnefataflGui gui;
    gui.show();

    return app.exec();

    // HnefataflGame game;
    // game.start();

}