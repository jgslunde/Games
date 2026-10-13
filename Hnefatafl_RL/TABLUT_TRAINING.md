# Tablut Training Setup

This document describes the new Tablut (9x9) training infrastructure.

## Overview

The Tablut training setup mirrors the Brandubh (7x7) training setup, but adapted for the larger 9x9 board. All Tablut-specific components have been created to match the existing architecture.

## New Files Created

### 1. `network_tablut.py`
- **TablutNet**: Neural network for 9x9 boards
  - Input: 4 planes of 9x9 (attackers, defenders, king, current player)
  - Policy output: 2592 possible moves (81 squares × 4 directions × 8 distances)
  - Value output: Win probability estimate [-1, 1]
  - Architecture: Same as BrandubhNet but scaled for 9x9

- **TablutMoveEncoder**: Move encoding/decoding for 9x9 board
  - Encoding formula: `policy_index = from_square * 32 + direction * 8 + (distance - 1)`
  - from_square: 0-80 (9×9 board positions)
  - direction: 0-3 (up, down, left, right)
  - distance: 1-8 (number of squares)

### 2. `train_tablut.py`
- Complete training script for Tablut
- Mirrors `train_brandubh.py` exactly
- Key difference: `config.board_size = 9` instead of 7
- Default checkpoint directory: `checkpoints_tablut/`
- Uses generic `Agent` class from `agent.py`

## Updated Files

### `agent.py` - Now Generic!
The agent module has been refactored to be completely generic and work with any Tafl game:

- **Agent**: Generic MCTS agent with neural network (replaces BrandubhAgent/TablutAgent)
  - Accepts `network_class` parameter to specify which network to use
  - Works with Brandubh, Tablut, or any other Tafl variant
  
- **RandomAgent**: Random player (already was game-agnostic)
- **MCTSAgent**: MCTS with random rollouts (already was game-agnostic)

**Backward compatibility maintained:**
```python
# Old way (still works):
from agent import BrandubhAgent  # Alias to Agent
from agent import TablutAgent    # Alias to Agent

# New way (recommended):
from agent import Agent
agent = Agent(network_class=TablutNet, num_simulations=100)
```

## Usage

### Quick Start - Training

```bash
# Basic training with defaults
python train_tablut.py

# Custom configuration
python train_tablut.py \
    --iterations 100 \
    --games 256 \
    --sims-attacker-selfplay 200 \
    --sims-defender-selfplay 200 \
    --checkpoint-dir my_tablut_checkpoints
```

### Playing Against the AI

Once you have a trained checkpoint:

```bash
python gui_play.py --game tablut --checkpoint checkpoints_tablut/best_model.pth
```

### Testing Components

Test the network and move encoder:
```bash
python network_tablut.py
```

Test the generic agent with Tablut:
```bash
python -c "
from agent import Agent, RandomAgent, MCTSAgent
from tablut import Tablut
from network_tablut import TablutNet

# Create agents
random_agent = RandomAgent()
mcts_agent = MCTSAgent(num_simulations=10)
nn_agent = Agent(network_class=TablutNet, num_simulations=10)

# Test them
game = Tablut()
print(random_agent.select_move(game))
print(mcts_agent.select_move(game))
"
```

## Training Parameters

All the same parameters from `train_brandubh.py` are available:

- **Core Training**: `--iterations`, `--games`, `--sims-attacker-selfplay`, `--sims-defender-selfplay`
- **Network**: `--res-blocks`, `--channels`
- **Learning**: `--lr`, `--lr-decay`, `--weight-decay`, `--batch-size`
- **MCTS**: `--c-puct`, `--temperature`, `--temperature-threshold`
- **Game Rules**: `--king-capture-pieces`, `--king-can-capture`, `--throne-is-hostile`, `--throne-enabled`
- **Boosting**: `--use-dynamic-boosting`, `--dynamic-boost-alpha`, etc.
- **Evaluation**: `--eval-games`, `--eval-win-rate`, `--eval-frequency`
- **System**: `--num-workers`, `--device`
- **Checkpointing**: `--save-frequency`, `--checkpoint-dir`, `--resume`

## Architecture Comparison

| Component | Brandubh (7x7) | Tablut (9x9) |
|-----------|---------------|--------------|
| Board Size | 7×7 = 49 squares | 9×9 = 81 squares |
| Input Shape | (4, 7, 7) | (4, 9, 9) |
| Policy Size | 1176 (49×4×6) | 2592 (81×4×8) |
| Max Distance | 6 squares | 8 squares |
| Network File | `network.py` | `network_tablut.py` |
| Agent File | `agent.py` | `agent_tablut.py` |
| Training Script | `train_brandubh.py` | `train_tablut.py` |
| Game File | `brandubh.py` | `tablut.py` |

## Move Encoding Details

### Brandubh (7x7)
- Total moves: 49 squares × 4 directions × 6 distances = 1176
- Encoding: `index = from_square * 24 + direction * 6 + (distance - 1)`

### Tablut (9x9)
- Total moves: 81 squares × 4 directions × 8 distances = 2592
- Encoding: `index = from_square * 32 + direction * 8 + (distance - 1)`

## Integration with Existing Code

The generic training module (`train.py`) works seamlessly with both Brandubh and Tablut because:

1. Both game classes implement the same interface:
   - `get_legal_moves()`
   - `make_move(move)`
   - `clone()`
   - `get_state()`
   - `game_over` property
   - `winner` attribute

2. The MCTS code (`mcts.py`) is game-agnostic (uses only the common interface)

3. The training configuration explicitly specifies which classes to use:
   ```python
   config.game_class = Tablut
   config.network_class = TablutNet
   config.agent_class = TablutAgent
   config.move_encoder_class = TablutMoveEncoder
   config.board_size = 9
   ```

## Bug Fix

While implementing Tablut training, a bug was discovered and fixed in `mcts.py`:

**Issue**: In `RandomRolloutMCTS._rollout()`, when selecting a random child for rollout, the child's game state wasn't being initialized (lazy initialization issue).

**Fix**: Added explicit game state creation before calling `_rollout()`:
```python
# Lazy initialization: create game state if not already done
if node.game is None:
    parent = search_path[-2]
    node.game = parent.game.clone()
    node.game.make_move(action)
```

This fix applies to both Brandubh and Tablut.

## Next Steps

1. Start training: `python train_tablut.py`
2. Monitor progress: Check `checkpoints_tablut/training_history.json`
3. Test the agent: Use `gui_play.py --game tablut --checkpoint checkpoints_tablut/best_model.pth`
4. Adjust hyperparameters based on training results
5. Compare Tablut learning to Brandubh learning (larger board may require more iterations)

## Expected Training Time

Tablut training will likely take longer than Brandubh due to:
- Larger board (81 vs 49 squares)
- More pieces (25 vs 13)
- Larger policy space (2592 vs 1176 moves)
- Longer games on average

Consider starting with fewer MCTS simulations or fewer games per iteration to get faster feedback during initial testing.
