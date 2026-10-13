"""
Summary of Tunable Game Rules Implementation
=============================================

## Overview
Modified the Brandubh game to support tunable rules that can be configured at instantiation.
All rules are integrated into the training pipeline and displayed at training start.

## Tunable Rules

### 1. king_capture_pieces (default: 2)
Controls how many pieces are required to capture the king:
- 2: Standard custodian capture (king must be sandwiched between two attackers on opposite sides)
- 3: King must be surrounded on 3 out of 4 sides
- 4: King must be surrounded on all 4 sides

### 2. king_can_capture (default: True)
Controls whether the king can participate in capturing enemy pieces:
- True: King can help defenders capture attackers (king + defender sandwich)
- False: King cannot participate in captures; only defender + defender works

### 3. throne_is_hostile (default: False)
Controls whether the throne (center square) acts as a hostile square for captures:
- True: Throne can be used as one side of a custodian capture (like corners)
- False: Throne is neutral; only actual pieces can capture

## Files Modified

### brandubh.py
- Updated `__init__` to accept rule parameters
- Modified `_check_capture_in_direction` to handle different king capture modes
- Added `_is_king_captured` method with logic for 2/3/4-piece captures
- Added `_is_hostile_square` method to handle throne/corner hostility
- Updated `_is_friendly` to respect king_can_capture rule
- Modified `clone` to preserve rule parameters

### train.py
- Added rule parameters to `TrainingConfig` class (top of file)
- Added "Game Rules" section to training configuration printout
- Updated `_play_self_play_game_worker` function signature to accept rule parameters
- Modified worker function call to pass rule parameters
- Updated `play_self_play_game` to create games with configured rules

### test_tunable_rules.py (new file)
Comprehensive test suite covering:
- All three king_capture_pieces modes (2, 3, 4)
- Both king_can_capture settings (True, False)
- Both throne_is_hostile settings (True, False)
- Combined rule scenarios
- Clone preservation of rules
- Edge cases (king at edges, corners, etc.)

All 21 tests pass successfully.

## Usage Examples

### Option 1: Command-Line Arguments (Easiest)
```bash
# Standard rules (default)
python train.py

# King requires 3 pieces to capture
python train.py --king-capture-pieces 3

# King requires 4 pieces to capture (all sides)
python train.py --king-capture-pieces 4

# King cannot participate in captures
python train.py --king-cannot-capture

# Throne acts as hostile square
python train.py --throne-is-hostile

# Combine multiple rules
python train.py \
    --king-capture-pieces 3 \
    --king-cannot-capture \
    --throne-is-hostile \
    --checkpoint-dir checkpoints_custom_rules

# See all available options
python train.py --help
```

### Option 2: Edit Default Values
Edit defaults in `train.py` (around line 1920):
```python
# Game rules
DEFAULT_KING_CAPTURE_PIECES = 2  # Change to 2, 3, or 4
DEFAULT_KING_CAN_CAPTURE = True  # Change to False
DEFAULT_THRONE_IS_HOSTILE = False  # Change to True
```

### Option 3: Direct Game Instantiation (Python code)
```python
from brandubh import Brandubh

# Standard rules (default)
game = Brandubh()

# Require 4 pieces to capture king
game = Brandubh(king_capture_pieces=4)

# King cannot help with captures
game = Brandubh(king_can_capture=False)

# Throne acts as hostile square
game = Brandubh(throne_is_hostile=True)

# Combined custom rules
game = Brandubh(
    king_capture_pieces=3,
    king_can_capture=False,
    throne_is_hostile=True
)
```

### Option 4: Programmatic Training Config
```python
from train import TrainingConfig, train

config = TrainingConfig()
config.king_capture_pieces = 3
config.king_can_capture = False
config.throne_is_hostile = True

train(config)
```

## Training Output
When training starts, the configuration now includes a "Game Rules" section:
```
--- Game Rules ---
King capture pieces: 2
  (standard custodian - 2 opposite attackers)
King can capture: True
Throne is hostile: False
```

## Backward Compatibility
All changes are fully backward compatible. Existing code that creates `Brandubh()`
without parameters will use the default rules (king_capture_pieces=2, 
king_can_capture=True, throne_is_hostile=False).

## Testing
Run the comprehensive test suite:
```bash
python test_tunable_rules.py
```

Expected output: 21 tests passed, 0 failed.
