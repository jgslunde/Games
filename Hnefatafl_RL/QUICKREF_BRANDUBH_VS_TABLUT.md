# Quick Reference: Brandubh vs Tablut

## File Mapping

| Purpose | Brandubh (7x7) | Tablut (9x9) | Shared/Generic |
|---------|----------------|--------------|----------------|
| Game Logic | `brandubh.py` | `tablut.py` | `games.py` (registry of all variants) |
| Neural Network | `network.py` (legacy BrandubhNet, MoveEncoder) | `network_tablut.py` (legacy TablutNet, TablutMoveEncoder) | **`tafl_network.py`** (configurable network for all boards, see `network_config.py`, `features.py`, NETWORK_ARCHITECTURES.md) |
| Agent | - | - | **`agent.py` (Agent, RandomAgent, MCTSAgent)** |
| Training config | `configs/brandubh/*.json` | `configs/tablut/*.json` | `run_training.py` + `train.py` (generic pipeline) |
| Tests | `test_brandubh.py` | `test_tablut.py` | - |

**Note:** The `agent.py` module is now completely generic and works with both games!

## Shared/Generic Files

These work with both games:
- **`agent.py`** - Generic agent classes (Agent, RandomAgent, MCTSAgent)
- `train.py` - Generic training pipeline
- `mcts.py` - Monte Carlo Tree Search (game-agnostic)
- `gui_play.py` - GUI for both games (use `--game brandubh` or `--game tablut`)
- `play.py` - Command-line play (if updated for Tablut)

### Using the Generic Agent

The `Agent` class can be used with any game and network:

```python
from agent import Agent, RandomAgent, MCTSAgent

# For Brandubh
from network import BrandubhNet
brandubh_agent = Agent(network_class=BrandubhNet, num_simulations=100)

# For Tablut
from network_tablut import TablutNet
tablut_agent = Agent(network_class=TablutNet, num_simulations=100)

# Backward compatibility (aliases still work)
from agent import BrandubhAgent  # Same as Agent
from agent import TablutAgent    # Same as Agent
```

## Playing Each Game

### Brandubh (7x7)
```bash
# GUI play
python gui_play.py --game brandubh --checkpoint checkpoints/best_model.pth

# Training
python run_training.py configs/brandubh/baseline.json --checkpoint-dir checkpoints
```

### Tablut (9x9)
```bash
# GUI play
python gui_play.py --game tablut --checkpoint checkpoints_tablut/best_model.pth

# Training
python run_training.py configs/tablut/baseline.json --checkpoint-dir checkpoints_tablut
```

## Key Constants

### Brandubh
```python
# From brandubh.py
BOARD_SIZE = 7
ATTACKER = 1    # Used for pieces and player identification
DEFENDER = 2
KING = 3
```

### Tablut
```python
# From tablut.py
BOARD_SIZE = 9
ATTACKER = 1    # Used for pieces
DEFENDER = 2
KING = 3
ATTACKER_PLAYER = 0  # Used for player identification
DEFENDER_PLAYER = 1
```

## Network Architecture

### Input (Both)
- 4 channels: [attackers, defenders, king, current_player]
- Shape: `(4, board_size, board_size)`

### Output (Both)
- Policy: Probability distribution over all possible moves
- Value: Win probability [-1, 1]

### Dimensions

| Aspect | Brandubh | Tablut |
|--------|----------|--------|
| Input Shape | (4, 7, 7) | (4, 9, 9) |
| Board Squares | 49 | 81 |
| Max Move Distance | 6 | 8 |
| Policy Size | 1176 | 2592 |
| Policy Formula | `49 * 4 * 6` | `81 * 4 * 8` |

## Move Encoding

Both use: `policy_index = from_square * (4 * max_distance) + direction * max_distance + (distance - 1)`

### Brandubh
- from_square: `row * 7 + col` (0-48)
- directions: 4 (up, down, left, right)
- distance: 1-6
- Total: 49 × 4 × 6 = **1176 moves**

### Tablut
- from_square: `row * 9 + col` (0-80)
- directions: 4 (up, down, left, right)
- distance: 1-8
- Total: 81 × 4 × 8 = **2592 moves**

## Default Checkpoints

- Brandubh: `checkpoints/`
- Tablut: `checkpoints_tablut/`

Each contains:
- `best_model.pth` - Best network weights
- `checkpoint_iter_N.pth` - Periodic checkpoints
- `training_history.json` - Training metrics

## Testing

```bash
# Test Brandubh network
python network.py

# Test Tablut network
python network_tablut.py

# Test generic agent (works with both games)
python agent.py

# Run unit tests
python -m pytest test_brandubh.py
python -m pytest test_tablut.py
```

## Game Rules (Tunable)

Both games support the same rule variations:

- `king_capture_pieces`: 2, 3, or 4 pieces needed to capture king
- `king_can_capture`: Whether king can participate in captures
- `throne_is_hostile`: Whether throne acts as enemy piece for captures
- `throne_enabled`: Whether throne exists and blocks non-king movement

Set these in:
- Game constructor: `Brandubh(king_capture_pieces=3)`
- Training script: `--king-capture-pieces 3`
- GUI: Not yet configurable via command line (uses game defaults)

## Performance Considerations

Tablut is computationally heavier than Brandubh:
- ~65% more squares (81 vs 49)
- ~92% more pieces (25 vs 13)
- ~120% larger policy space (2592 vs 1176)
- Longer average game length

**Recommendation**: Start Tablut training with reduced MCTS simulations or games per iteration for faster iteration during development.
