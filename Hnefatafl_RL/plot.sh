#!/bin/bash
# Plot all training outputs in checkpoint directories

# Find all output.txt files in checkpoints* directories
for output_file in checkpoints*/output.txt; do
    # Check if the file exists (in case no matches found)
    if [ -f "$output_file" ]; then
        echo "Processing $output_file..."
        python plot.py "$output_file"
        echo ""
    fi
done

echo "Done processing all output files."
