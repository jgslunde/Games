# Tablut Implementation

## Overview
Tablut is a 9x9 variant of Tafl games, now fully implemented alongside the existing Brandubh (7x7) game.

## Files Created/Modified

### New Files
1. **tablut.py** - Complete Tablut game implementation
   - 9x9 board
   - Standard Tablut piece layout: 16 attackers, 8 defenders, 1 king
   - Mirrors brandubh.py structure exactly
   - Same tunable rules:
     - `king_capture_pieces`: How many pieces needed to capture king (2, 3, or 4)
     - `king_can_capture`: Whether king participates in captures (bool)
     - `throne_is_hostile`: Whether throne acts as hostile square (bool)
     - `throne_enabled`: Whether non-king pieces can occupy throne (bool)

2. **test_tablut.py** - Comprehensive test suite
   - 20 tests covering all game mechanics
   - Tests initialization, movement, captures, rule variations, victory conditions
   - All tests passing ✓

### Modified Files
1. **gui_play.py** - Updated to support both games
   - Renamed `BrandubhGUI` to `TaflGUI`
   - Added `--game` command-line argument to choose between brandubh and tablut
   - Dynamic board sizing based on game type
   - Updated window title to reflect game variant

## Usage

### Playing with AI Evaluation
```bash
# Brandubh with AI
python gui_play.py --checkpoint checkpoints/best_model.pth

# Tablut with AI
python gui_play.py --checkpoint checkpoints/best_model.pth --game tablut
```

### Playing Without AI (Human vs Human)
```bash
# Brandubh without AI
python gui_play.py

# Tablut without AI
python gui_play.py --game tablut
```

### Additional Options
```bash
python gui_play.py --checkpoint checkpoints/best_model.pth --game tablut --simulations 200 --c-puct 1.5
```

## Testing
Run the Tablut test suite:
```bash
python test_tablut.py
```

Expected output:
```
Running Tablut tests...
✓ Board initialization test passed
✓ Basic movement test passed
✓ Rook movement test passed
[... 17 more tests ...]
==================================================
All Tablut tests passed! ✓
==================================================
```

## Game Rules
Tablut follows standard Tafl rules on a 9x9 board:

### Starting Position
- **Attackers (16)**: Placed on edges of the board
- **Defenders (8)**: Surround the king in a cross pattern
- **King (1)**: Starts on the throne at center (4,4)

### Special Squares
- **Throne**: Center square (4,4)
- **Corners**: (0,0), (0,8), (8,0), (8,8)

### Victory Conditions
- **Defenders win**: King reaches any corner
- **Attackers win**: King is captured according to capture rules
- **Draw**: Threefold repetition or no legal moves (stalemate)

### Movement
- All pieces move like rooks (orthogonal, any distance)
- Pieces cannot jump over other pieces
- Only king can occupy throne (unless throne_enabled=False)
- Corners cannot be occupied (but can be moved through)

### Captures
- Standard custodian capture: sandwich enemy between two friendly pieces
- King capture rules depend on `king_capture_pieces` setting:
  - 2: Standard sandwich (like regular pieces)
  - 3: Three sides must be occupied by attackers
  - 4: All four sides must be occupied by attackers
- Hostile squares (throne/corners) can substitute for friendly pieces in captures

## Training
To train a model specifically for Tablut, you would need to:
1. Update `network.py` to handle 9x9 board (or make it dynamic)
2. Create a training script using `tablut.py` instead of `brandubh.py`
3. Follow the same self-play + MCTS approach as with Brandubh

## Implementation Notes
- The `tablut.py` implementation exactly mirrors `brandubh.py` structure
- All methods adapted for 9x9 board (changed `range(7)` to `range(9)`)
- Fixed a bug in `_is_friendly()` method for `king_can_capture=False` rule
- GUI now dynamically calculates square sizes and offsets based on board size
- Both games share the same piece constants (EMPTY, ATTACKER, DEFENDER, KING)
