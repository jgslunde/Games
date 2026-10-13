#!/bin/bash
# Example script demonstrating ELO evaluation usage

# This script shows various ways to use the ELO evaluation system

echo "=========================================="
echo "ELO Evaluation System - Example Usage"
echo "=========================================="
echo ""

# Check if checkpoint directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_directory>"
    echo ""
    echo "Example: $0 checkpoints_brandubh/fatman_nothrone/"
    echo ""
    echo "This script demonstrates various ELO evaluation commands."
    exit 1
fi

CHECKPOINT_DIR="$1"

# Validate checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Directory not found: $CHECKPOINT_DIR"
    exit 1
fi

echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Detect game type from directory name
GAME="brandubh"
if [[ "$CHECKPOINT_DIR" == *"tablut"* ]]; then
    GAME="tablut"
elif [[ "$CHECKPOINT_DIR" == *"hnefatafl"* ]]; then
    GAME="hnefatafl"
fi

echo "Detected game type: $GAME"
echo ""

# Example 1: Quick evaluation
echo "=========================================="
echo "Example 1: Quick Evaluation (Parallel)"
echo "=========================================="
echo "Command: python evaluate_elo.py $CHECKPOINT_DIR \\"
echo "           --game $GAME \\"
echo "           --max-checkpoints 10 \\"
echo "           --num-games 20 \\"
echo "           --simulations 100 \\"
echo "           --num-workers 64"
echo ""
echo "This runs a fast evaluation on the first 10 checkpoints."
echo "With 64 workers, 20 games per match are played in parallel."
echo "Press Enter to run, or Ctrl+C to skip..."
read

python evaluate_elo.py "$CHECKPOINT_DIR" \
    --game "$GAME" \
    --max-checkpoints 10 \
    --num-games 20 \
    --simulations 100 \
    --num-workers 64

echo ""
echo "Quick evaluation complete!"
echo ""

# Example 2: Standard evaluation
echo "=========================================="
echo "Example 2: Standard Evaluation (Full Parallelization)"
echo "=========================================="
echo "Command: python evaluate_elo.py $CHECKPOINT_DIR \\"
echo "           --game $GAME \\"
echo "           --skip-checkpoints 5 \\"
echo "           --num-games 20 \\"
echo "           --simulations 200 \\"
echo "           --checkpoint-distance 10 \\"
echo "           --num-workers 128"
echo ""
echo "This evaluates every 5th checkpoint with standard settings."
echo "Uses all 128 CPUs for maximum speed - games complete in ~30 seconds each!"
echo "Press Enter to run, or Ctrl+C to skip..."
read

python evaluate_elo.py "$CHECKPOINT_DIR" \
    --game "$GAME" \
    --skip-checkpoints 5 \
    --num-games 20 \
    --simulations 200 \
    --checkpoint-distance 10 \
    --num-workers 128

echo ""
echo "Standard evaluation complete!"
echo ""

# Example 3: Visualization
echo "=========================================="
echo "Example 3: Visualize Results"
echo "=========================================="

RESULTS_FILE="$CHECKPOINT_DIR/elo_evaluation_results.json"

if [ -f "$RESULTS_FILE" ]; then
    echo "Found results file: $RESULTS_FILE"
    echo ""
    echo "Creating visualizations..."
    
    python plot_elo.py "$RESULTS_FILE" --both
    
    echo ""
    echo "Visualizations saved to:"
    echo "  - $CHECKPOINT_DIR/elo_progression.png"
    echo "  - $CHECKPOINT_DIR/match_matrix.png"
else
    echo "No results file found at: $RESULTS_FILE"
    echo "Run an evaluation first!"
fi

echo ""
echo "=========================================="
echo "All examples complete!"
echo "=========================================="
echo ""
echo "For more options, see:"
echo "  python evaluate_elo.py --help"
echo "  python plot_elo.py --help"
echo ""
echo "Or read the documentation:"
echo "  cat ELO_EVALUATION_README.md"
