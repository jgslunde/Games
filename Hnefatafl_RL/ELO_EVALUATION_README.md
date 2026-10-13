# ELO Evaluation System

This document explains how to use the ELO evaluation system to assess the relative strength of checkpoints during training.

## Overview

The ELO evaluation system (`evaluate_elo.py`) plays matches between different training checkpoints and computes their relative ELO ratings. This helps you understand:

- How quickly the model is improving during training
- Whether training progress is consistent or has plateaus
- Which checkpoints are strongest relative to others
- Whether earlier checkpoints are genuinely weaker or just different

**Key Feature:** The system uses **multiprocessing** to parallelize game execution across multiple CPU cores, making large-scale evaluations computationally feasible. By default, it uses all available CPU cores (typically 128 in your environment).

## Quick Start

### Basic Usage

```bash
python evaluate_elo.py checkpoints_brandubh/fatman_nothrone/
```

This will:
1. Find all checkpoints matching the pattern `checkpoint_iter_*.pth`
2. Play 20 games between each checkpoint and the next 10 checkpoints
3. Compute ELO ratings using maximum likelihood estimation
4. Save results to `elo_evaluation_results.json`
5. Display a table of ELO ratings

### Common Options

```bash
# Play more games per match for better accuracy
python evaluate_elo.py checkpoints_brandubh/fatman_nothrone/ --num-games 50

# Use fewer workers (default uses all CPUs)
python evaluate_elo.py checkpoints_brandubh/fatman_nothrone/ --num-workers 32

# Compare checkpoints that are further apart
python evaluate_elo.py checkpoints_brandubh/fatman_nothrone/ --checkpoint-distance 20

# Use more MCTS simulations for stronger play
python evaluate_elo.py checkpoints_brandubh/fatman_nothrone/ --simulations 400

# Evaluate fewer checkpoints (faster)
python evaluate_elo.py checkpoints_brandubh/fatman_nothrone/ --skip-checkpoints 5
python evaluate_elo.py checkpoints_brandubh/fatman_nothrone/ --max-checkpoints 30

# Resume from previous evaluation (useful if interrupted)
python evaluate_elo.py checkpoints_brandubh/fatman_nothrone/ --resume

# Evaluate Tablut or Hnefatafl checkpoints
python evaluate_elo.py checkpoints_tablut/jack_nothrone/ --game tablut
python evaluate_elo.py checkpoints_hnefatafl/jack_standard/ --game hnefatafl
```

## Understanding the Parameters

### `--num-workers` (default: all CPUs)
Number of parallel workers for playing games simultaneously. Each match between two checkpoints consists of multiple games that are played in parallel across available CPU cores.

**Recommendation:**
- Default (all CPUs): Best for maximum throughput on HPC systems
- Half of CPUs: Good balance if sharing resources
- 1 worker: Sequential execution for debugging only

**Performance impact:** With 128 CPUs and 20 games per match, all games in a match are played simultaneously, dramatically reducing evaluation time from hours to minutes.

### `--num-games` (default: 20)
Number of games to play per match. Higher values give more accurate ELO ratings but take longer. Each agent plays half the games as attacker and half as defender.

**Recommendation:** 
- Quick test: 10 games
- Normal evaluation: 20-50 games
- High confidence: 100+ games

### `--checkpoint-distance` (default: 10)
Maximum distance between checkpoints to match. Each checkpoint plays against the next N checkpoints. This prevents comparing very weak early checkpoints against very strong late checkpoints (where win rates would be 100% or 0%, providing little ELO information).

**Recommendation:**
- For smooth learning curves: 5-10
- For detecting plateaus: 10-20
- For sparse checkpoints: Increase accordingly

**Why this matters:** ELO is most reliable when win rates are between 20% and 80%. Matches with extreme win rates (0% or 100%) don't provide useful information about relative strength.

### `--simulations` (default: 200)
Number of MCTS simulations per move. Higher values make agents play stronger but slower.

**Recommendation:**
- Quick evaluation: 100-200
- Standard: 200-400
- High quality: 400-800

### `--skip-checkpoints` (default: 1)
Evaluate every Nth checkpoint. Use this to speed up evaluation on large training runs.

**Example:** With `--skip-checkpoints 5`, only iterations 0, 5, 10, 15, ... are evaluated.

### `--max-checkpoints` (default: all)
Limit the total number of checkpoints evaluated. Combined with checkpoint skipping, this lets you sample the training history.

## Output Format

### Console Output

The program displays:
1. Progress of matches being played
2. Final ELO ratings table sorted by strength
3. Statistics (mean, median, std, range, spread)

Example output:
```
================================================================================
ELO RATINGS
================================================================================
Rank   Iteration    ELO        Checkpoint                                        
--------------------------------------------------------------------------------
1      500          1682.3     checkpoint_iter_500.pth                           
2      490          1671.5     checkpoint_iter_490.pth                           
3      480          1656.2     checkpoint_iter_480.pth                           
...
================================================================================

ELO Statistics:
  Mean:   1549.3
  Median: 1547.8
  Std:    58.4
  Range:  1420.1 - 1682.3
  Spread: 262.2
```

### JSON Output

Results are saved to `elo_evaluation_results.json` in the checkpoint directory:

```json
{
  "metadata": {
    "checkpoint_dir": "checkpoints_brandubh/fatman_nothrone/",
    "game": "brandubh",
    "num_games_per_match": 20,
    "checkpoint_distance": 10,
    "simulations": 200,
    "base_elo": 1500.0,
    "total_matches": 245,
    "timestamp": "2025-11-15 14:30:45"
  },
  "checkpoints": [
    {
      "index": 0,
      "filename": "checkpoint_iter_0.pth",
      "iteration": 0,
      "elo": 1420.1
    },
    ...
  ],
  "matches": [
    {
      "checkpoint1_index": 0,
      "checkpoint2_index": 1,
      "checkpoint1_wins": 3,
      "checkpoint2_wins": 15,
      "draws": 2,
      "total_games": 20
    },
    ...
  ]
}
```

## Visualizing Results

Use `plot_elo.py` to visualize the ELO ratings:

```bash
# Plot ELO progression over training
python plot_elo.py checkpoints_brandubh/fatman_nothrone/elo_evaluation_results.json

# Plot match results matrix
python plot_elo.py checkpoints_brandubh/fatman_nothrone/elo_evaluation_results.json --matrix

# Plot both
python plot_elo.py checkpoints_brandubh/fatman_nothrone/elo_evaluation_results.json --both
```

This creates:
- `elo_progression.png`: ELO rating vs training iteration
- `match_matrix.png`: Heatmap of win percentages between all pairs

## How ELO Computation Works

### Traditional ELO
In chess, ELO ratings are updated incrementally after each game. This works well for ongoing tournaments but can be inconsistent when applied to a fixed set of match results.

### Maximum Likelihood ELO (Used Here)
This system computes ELO ratings that **best explain all match results simultaneously** using maximum likelihood estimation:

1. Start with initial ELO estimates for all checkpoints
2. Calculate predicted win probabilities for all matches using ELO formula: 
   `P(win) = 1 / (1 + 10^(-ΔE/400))`
3. Compute cross-entropy loss between predicted and actual results
4. Optimize ELO ratings to minimize this loss
5. Constraint: First checkpoint has ELO = base_elo (default 1500)

**Advantages:**
- More stable and consistent than incremental updates
- All match results contribute equally to final ratings
- No dependency on match order
- Minimizes overall prediction error

### Interpreting ELO Differences

- **50 ELO:** ~7% better win rate (57% vs 43%)
- **100 ELO:** ~14% better win rate (64% vs 36%)
- **200 ELO:** ~26% better win rate (76% vs 24%)
- **400 ELO:** ~48% better win rate (92% vs 8%)

Note: These are approximate and assume similar playing styles.

## Resuming Evaluations

If an evaluation is interrupted, use `--resume` to continue from where it left off:

```bash
python evaluate_elo.py checkpoints_brandubh/fatman_nothrone/ --resume
```

This loads existing match results from `elo_evaluation_results.json` and only plays new matches that haven't been completed.

## Tips for Efficient Evaluation

### Maximizing Parallelization

The system is designed to fully utilize multi-core systems:

1. **Match-level parallelization:** Each match plays all games simultaneously
2. **Optimal for HPC:** With 128 CPUs, a 20-game match completes as fast as a single game
3. **Memory efficient:** Each worker loads models independently, avoiding shared memory issues

**Example timing on 128 CPUs:**
- Single game: ~30 seconds
- 20-game match (sequential): ~10 minutes
- 20-game match (parallel): ~30 seconds (128x speedup)
- Full evaluation (50 matches): ~25 minutes vs ~8+ hours sequential

### For Large Training Runs (>100 checkpoints)

1. **Sample checkpoints:** Use `--skip-checkpoints 5` or `--max-checkpoints 50`
2. **Reduce games:** Use `--num-games 10` for initial analysis
3. **Use resume:** Run evaluation in stages with `--resume`

### For Accurate Rankings

1. **More games:** Use `--num-games 50` or higher
2. **Higher simulations:** Use `--simulations 400`
3. **Appropriate distance:** Ensure most matches have 20-80% win rates

### For Quick Analysis

1. **Fewer games:** `--num-games 10`
2. **Skip checkpoints:** `--skip-checkpoints 10`
3. **Lower simulations:** `--simulations 100`

## Troubleshooting

### "No checkpoints found"
- Check the checkpoint directory path
- Verify checkpoint naming matches `checkpoint_iter_*.pth`
- Use `--pattern` to specify a different naming pattern

### "Optimization did not converge"
- Not a critical error - results are still usable
- Can indicate insufficient match data
- Try increasing `--num-games` or reducing `--checkpoint-distance`

### Very long runtime
- System should scale well with 128 CPUs
- If still too slow, reduce number of checkpoints with `--skip-checkpoints`
- Or reduce `--num-games` (affects ELO accuracy)
- Check CPU utilization: `htop` should show ~100% usage across cores

### Multiprocessing issues
- Error about "spawn" method: Already handled in the script
- Out of memory: Reduce `--num-workers` (each worker loads a full model)
- Hanging processes: Check system resource limits (`ulimit -a`)

### Unexpected ELO patterns
- Ensure checkpoints are from consistent training (same rules/architecture)
- Check that you're using the correct game variant (`--game`)
- Verify match results aren't all one-sided (increase `--checkpoint-distance`)

## Example Workflows

### Initial Training Analysis
```bash
# Quick overview of first 50 checkpoints
python evaluate_elo.py checkpoints_brandubh/experiment1/ \
    --max-checkpoints 50 \
    --num-games 10 \
    --simulations 200

# Visualize
python plot_elo.py checkpoints_brandubh/experiment1/elo_evaluation_results.json
```

### Detailed Final Analysis
```bash
# Comprehensive evaluation of full training run
python evaluate_elo.py checkpoints_brandubh/experiment1/ \
    --skip-checkpoints 2 \
    --num-games 50 \
    --simulations 400 \
    --checkpoint-distance 15

# Create visualizations
python plot_elo.py checkpoints_brandubh/experiment1/elo_evaluation_results.json --both
```

### Comparing Different Experiments
Run evaluation on multiple checkpoint directories and compare the ELO progression plots to see which training configuration produces the strongest learning curve.

## Advanced Options

### Custom Game Rules
Override game rules from checkpoints:
```bash
python evaluate_elo.py checkpoints_brandubh/custom/ \
    --king-capture-pieces 4 \
    --throne-enabled true
```

### Custom Checkpoint Pattern
```bash
python evaluate_elo.py checkpoints_brandubh/custom/ \
    --pattern "model_*.pth"
```

### Different Base ELO
```bash
python evaluate_elo.py checkpoints_brandubh/experiment1/ \
    --base-elo 1000.0
```

Note: The base ELO is arbitrary - it's just a reference point. Only ELO *differences* are meaningful.
