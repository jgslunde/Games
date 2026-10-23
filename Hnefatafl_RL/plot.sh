#!/bin/bash
# Plot the training progress of every run in checkpoints/<rule set>/<run>/, from its
# training_history.json, or from its output.txt if the JSON file was written by an older train.py
# (it lacks "eval_iterations" until the run is resumed once)
for run_dir in checkpoints/*/*/; do
    if grep -q '"eval_iterations"' "$run_dir/training_history.json" 2>/dev/null; then
        input_file="$run_dir/training_history.json"
    elif [ -f "$run_dir/output.txt" ]; then
        input_file="$run_dir/output.txt"
    else
        continue
    fi
    echo "Processing $input_file..."
    .venv/bin/python plot.py "$input_file"
    echo ""
done
echo "Done processing all runs."
