#!/bin/bash
# Quick benchmark script for neural network inference speed
# Tests single-position inference (batch_size=1) for realistic gameplay performance

echo "Running neural network inference benchmarks..."
echo ""
echo "Testing Brandubh (7x7)..."
python benchmark_network.py --game brandubh --batch-size 1 --iterations 500

echo ""
echo "========================================================================"
echo ""
echo "Testing Tablut (9x9)..."
python benchmark_network.py --game tablut --batch-size 1 --iterations 500

echo ""
echo "========================================================================"
echo ""
echo "Testing Hnefatafl (11x11)..."
python benchmark_network.py --game hnefatafl --batch-size 1 --iterations 500

echo ""
echo "Benchmarking complete!"
