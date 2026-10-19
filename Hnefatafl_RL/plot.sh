#!/bin/bash
# Plot the training output of every run in checkpoints/<rule set>/<run>/
for output_file in checkpoints/*/*/output.txt; do
    if [ -f "$output_file" ]; then  # the pattern stays as it is when nothing matches
        echo "Processing $output_file..."
        .venv/bin/python plot.py "$output_file"
        echo ""
    fi
done
echo "Done processing all output files."
