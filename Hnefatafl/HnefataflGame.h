#include "HnefataflBoard.h"
#include <memory>

class HnefataflGame {
public:
    enum class Mode {
        PVP,  // Player vs Player
        PVA,  // Player vs AI
        AVP,  // AI vs Player
        AVA   // AI vs AI
    };

    HnefataflGame();
    void start();
    void chooseMode();
    bool promptForMove(HnefataflBoard::Player player);

private:
    std::unique_ptr<HnefataflBoard> board;
    Mode gameMode;
};