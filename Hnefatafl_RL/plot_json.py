#!/usr/bin/env python3
"""
Parse training_history.json and create matplotlib plots of training progress.

This script reads training_history.json file(s) and generates plots:
1. plot_elo.png - ELO vs first version (network vs network evaluation)
2. plot_win_rates.png - Win rates (attacker/defender/draw) for self-play and self-evaluation
3. plot_losses.png - Training and validation losses (policy, value, L2, total)
4. plot_gradients.png - Policy and value gradient norms
5. plot_buffer_time.png - Replay buffer size and learning rate

All plots are saved in the same directory as the input file.

Usage: python plot_json.py <path_to_training_history.json> [path_to_training_history2.json] [...]

Example: python plot_json.py checkpoints_brandubh/jack_nothrone/training_history.json
Example: python plot_json.py checkpoints_brandubh/jack_nothrone/training_history.json checkpoints_brandubh/jack_nothrone_v2/training_history.json
"""

import sys
import os
import json
import matplotlib.pyplot as plt


def load_json_file(filepath):
    """Load a training_history.json file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def merge_data_from_files(file_data_list):
    """
    Merge data from multiple training_history.json files.
    
    For cumulative ELO, adds the final ELO from each file to the values in subsequent files.
    For other metrics, simply concatenates the data.
    
    Args:
        file_data_list: List of data dictionaries from load_json_file()
    
    Returns:
        Merged data dictionary
    """
    if len(file_data_list) == 1:
        return file_data_list[0]
    
    # Initialize merged dictionary with all possible keys
    merged = {}
    all_keys = set()
    for data in file_data_list:
        all_keys.update(data.keys())
    
    for key in all_keys:
        if key == 'metadata':
            # Keep metadata from first file
            merged['metadata'] = file_data_list[0].get('metadata', {})
        elif key == 'selfplay_results' or key == 'timing':
            # These are nested dictionaries
            merged[key] = {}
            if key in file_data_list[0]:
                for subkey in file_data_list[0][key].keys():
                    merged[key][subkey] = []
        else:
            merged[key] = []
    
    # Handle cumulative ELO offset
    cumulative_elo_offset = 0.0
    
    for file_idx, data in enumerate(file_data_list):
        # For cumulative ELO, add offset from previous files
        if 'cumulative_elo' in data and data['cumulative_elo']:
            adjusted_elo = [elo + cumulative_elo_offset for elo in data['cumulative_elo']]
            merged['cumulative_elo'].extend(adjusted_elo)
            # Update offset for next file
            cumulative_elo_offset = adjusted_elo[-1]
        
        # Concatenate all other list fields
        for key in data:
            if key == 'metadata':
                continue
            elif key == 'selfplay_results' or key == 'timing':
                # Handle nested dictionaries
                if key in data:
                    for subkey in data[key]:
                        if subkey in merged[key]:
                            merged[key][subkey].extend(data[key][subkey])
            elif key != 'cumulative_elo' and isinstance(data[key], list):
                merged[key].extend(data[key])
    
    return merged


def plot_elo(data, output_dir):
    """Plot 1: ELO versus first version (network vs network)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = data.get('iterations', [])
    cumulative_elo = data.get('cumulative_elo', [])
    elo_gain = data.get('elo_gain', [])
    
    if cumulative_elo and len(cumulative_elo) == len(iterations):
        # Filter out None/null values
        valid_data = [(it, elo) for it, elo in zip(iterations, cumulative_elo) if elo is not None]
        if valid_data:
            valid_iters, valid_elos = zip(*valid_data)
            ax.plot(valid_iters, valid_elos, 's-', label='Cumulative ELO vs First Version', 
                   linewidth=2, markersize=4, color='tab:blue')
    
    if elo_gain:
        # Filter out None/null values and match with iterations
        # elo_gain might not be recorded for every iteration
        valid_gains = [(it, gain) for it, gain in zip(iterations, elo_gain) if gain is not None]
        if valid_gains:
            gain_iters, gains = zip(*valid_gains)
            ax2 = ax.twinx()
            ax2.plot(gain_iters, gains, 'o-', label='ELO Gain (vs previous)', 
                    linewidth=2, markersize=4, color='tab:orange', alpha=0.7)
            ax2.set_ylabel('ELO Gain (vs previous version)', fontsize=12, color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            
            # Combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
        else:
            ax.legend(loc='upper left', fontsize=11)
    else:
        ax.legend(loc='upper left', fontsize=11)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cumulative ELO Rating', fontsize=12, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_title('ELO Ratings During Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_elo.png'), dpi=150)
    plt.close()


def plot_win_rates(data, output_dir):
    """Plot 2: Win rates for self-play and self-evaluation (network vs network)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    iterations = data.get('iterations', [])
    win_rates = data.get('win_rates', [])
    
    # Self-play win rates
    ax = axes[0]
    selfplay = data.get('selfplay_results', {})
    if selfplay:
        attacker_wins = selfplay.get('attacker_wins', [])
        defender_wins = selfplay.get('defender_wins', [])
        draws = selfplay.get('draws', [])
        
        # Convert to percentages
        if attacker_wins and defender_wins:
            total_games = [a + d + dr for a, d, dr in zip(attacker_wins, defender_wins, draws)]
            attacker_pct = [100 * a / t if t > 0 else 0 for a, t in zip(attacker_wins, total_games)]
            defender_pct = [100 * d / t if t > 0 else 0 for d, t in zip(defender_wins, total_games)]
            draws_pct = [100 * dr / t if t > 0 else 0 for dr, t in zip(draws, total_games)]
            
            ax.plot(iterations[:len(attacker_pct)], attacker_pct, 'o-', label='Attacker', 
                   linewidth=2, markersize=4)
            ax.plot(iterations[:len(defender_pct)], defender_pct, 's-', label='Defender', 
                   linewidth=2, markersize=4)
            ax.plot(iterations[:len(draws_pct)], draws_pct, '^-', label='Draw', 
                   linewidth=2, markersize=4)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Self-Play Win Rates', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Self-evaluation win rates (network vs network)
    ax = axes[1]
    if win_rates:
        # Filter out None/null values
        valid_data = [(it, wr) for it, wr in zip(iterations, win_rates) if wr is not None]
        if valid_data:
            valid_iters, valid_rates = zip(*valid_data)
            ax.plot(valid_iters, [wr * 100 for wr in valid_rates], 'o-', 
                   label='New Network Win Rate', linewidth=2, markersize=4, color='tab:purple')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Self-Evaluation (New vs Old Network)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=55, color='green', linestyle='--', alpha=0.3, linewidth=1, label='Threshold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_win_rates.png'), dpi=150)
    plt.close()


def plot_losses(data, output_dir):
    """Plot 3: Training and validation losses (policy, value, L2, total)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = data.get('iterations', [])
    
    # Policy Loss
    ax = axes[0, 0]
    policy_loss = data.get('policy_loss', [])
    val_policy_loss = data.get('val_policy_loss', [])
    
    if policy_loss:
        ax.plot(iterations[:len(policy_loss)], policy_loss, 'o-', 
               label='Training', linewidth=2, markersize=4, color='tab:blue')
    if val_policy_loss:
        ax.plot(iterations[:len(val_policy_loss)], val_policy_loss, 's-', 
               label='Validation', linewidth=2, markersize=4, color='tab:orange')
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Policy Loss', fontsize=11)
    ax.set_title('Policy Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Value Loss
    ax = axes[0, 1]
    value_loss = data.get('value_loss', [])
    val_value_loss = data.get('val_value_loss', [])
    
    if value_loss:
        ax.plot(iterations[:len(value_loss)], value_loss, 'o-', 
               label='Training', linewidth=2, markersize=4, color='tab:blue')
    if val_value_loss:
        ax.plot(iterations[:len(val_value_loss)], val_value_loss, 's-', 
               label='Validation', linewidth=2, markersize=4, color='tab:orange')
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Value Loss', fontsize=11)
    ax.set_title('Value Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # L2 Loss
    ax = axes[1, 0]
    l2_loss = data.get('l2_loss', [])
    val_l2_loss = data.get('val_l2_loss', [])
    
    if l2_loss:
        ax.plot(iterations[:len(l2_loss)], l2_loss, 'o-', 
               label='Training', linewidth=2, markersize=4, color='tab:blue')
    if val_l2_loss:
        ax.plot(iterations[:len(val_l2_loss)], val_l2_loss, 's-', 
               label='Validation', linewidth=2, markersize=4, color='tab:orange')
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('L2 Regularization Loss', fontsize=11)
    ax.set_title('L2 Regularization Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Total Loss
    ax = axes[1, 1]
    total_loss = data.get('total_loss', [])
    val_total_loss = data.get('val_total_loss', [])
    
    if total_loss:
        ax.plot(iterations[:len(total_loss)], total_loss, 'o-', 
               label='Training', linewidth=2, markersize=4, color='tab:blue')
    if val_total_loss:
        ax.plot(iterations[:len(val_total_loss)], val_total_loss, 's-', 
               label='Validation', linewidth=2, markersize=4, color='tab:orange')
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Total Loss', fontsize=11)
    ax.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_losses.png'), dpi=150)
    plt.close()


def plot_gradients(data, output_dir):
    """Plot 4: Gradient norms for policy and value heads."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = data.get('iterations', [])
    policy_grad = data.get('policy_grad', [])
    value_grad = data.get('value_grad', [])
    
    if policy_grad:
        ax.plot(iterations[:len(policy_grad)], policy_grad, 'o-', 
               label='Policy Gradient Norm', linewidth=2, markersize=4, color='tab:blue')
    
    if value_grad:
        ax.plot(iterations[:len(value_grad)], value_grad, 's-', 
               label='Value Gradient Norm', linewidth=2, markersize=4, color='tab:red')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Gradient Norm (L2)', fontsize=12)
    ax.set_title('Gradient Norms During Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_gradients.png'), dpi=150)
    plt.close()


def plot_buffer_and_lr(data, output_dir):
    """Plot 5: Buffer size, learning rate, and iteration time."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    iterations = data.get('iterations', [])
    buffer_size = data.get('buffer_size', [])
    learning_rate = data.get('learning_rate', [])
    
    # Get timing data - could be in timing.total_time or just a list
    timing = data.get('timing', {})
    if isinstance(timing, dict):
        total_time = timing.get('total_time', [])
    else:
        total_time = []
    
    color1 = 'tab:green'
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Buffer Size', fontsize=12, color=color1)
    if buffer_size:
        ax1.plot(iterations[:len(buffer_size)], buffer_size, 'o-', 
                color=color1, label='Buffer Size', linewidth=2, markersize=4)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # Create second y-axis for learning rate and time
    ax2 = ax1.twinx()
    
    if learning_rate:
        color2 = 'tab:purple'
        # Filter out zero or negative values for log scale
        valid_lr = [(it, lr) for it, lr in zip(iterations[:len(learning_rate)], learning_rate) if lr > 0]
        if valid_lr:
            lr_iters, lr_vals = zip(*valid_lr)
            ax2.plot(lr_iters, lr_vals, 's-', 
                    color=color2, label='Learning Rate', linewidth=2, markersize=4)
            ax2.set_yscale('log')
    
    if total_time:
        color3 = 'tab:orange'
        ax2.plot(iterations[:len(total_time)], total_time, '^-', 
                color=color3, label='Iteration Time', linewidth=2, markersize=4, alpha=0.7)
        if not learning_rate:
            # If no learning rate, use linear scale
            ax2.set_ylim(bottom=0)
    
    if learning_rate or total_time:
        if learning_rate and total_time:
            ax2.set_ylabel('Learning Rate (log) / Iteration Time (s)', fontsize=12)
        elif learning_rate:
            ax2.set_ylabel('Learning Rate (log scale)', fontsize=12, color='tab:purple')
            ax2.tick_params(axis='y', labelcolor='tab:purple')
        else:
            ax2.set_ylabel('Iteration Time (s)', fontsize=12, color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    ax1.set_title('Buffer Size, Learning Rate, and Timing', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    if learning_rate or total_time:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    else:
        ax1.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_buffer_lr.png'), dpi=150)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_json.py <path_to_training_history.json> [path_to_training_history2.json] [...]")
        sys.exit(1)
    
    filepaths = sys.argv[1:]
    
    # Check that all files exist
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' not found.")
            sys.exit(1)
    
    # Load all files
    print(f"Loading {len(filepaths)} file(s)...")
    file_data_list = []
    for filepath in filepaths:
        print(f"  - {filepath}")
        data = load_json_file(filepath)
        file_data_list.append(data)
    
    # Merge data from multiple files if necessary
    if len(file_data_list) > 1:
        print(f"\nMerging data from {len(file_data_list)} files...")
        data = merge_data_from_files(file_data_list)
    else:
        data = file_data_list[0]
    
    # Output directory is the same as the first input file
    output_dir = os.path.dirname(filepaths[0])
    if not output_dir:
        output_dir = '.'
    
    print(f"\nCreating plots...")
    iterations = data.get('iterations', [])
    print(f"  - Found {len(iterations)} iterations")
    if iterations:
        print(f"  - Iteration range: {min(iterations)} to {max(iterations)}")
    
    # Count data points
    policy_loss = data.get('policy_loss', [])
    val_policy_loss = data.get('val_policy_loss', [])
    policy_grad = data.get('policy_grad', [])
    cumulative_elo = data.get('cumulative_elo', [])
    
    print(f"  - Training losses: {len(policy_loss)} data points")
    if val_policy_loss:
        print(f"  - Validation losses: {len(val_policy_loss)} data points")
    if policy_grad:
        print(f"  - Gradient norms: {len(policy_grad)} data points")
    if cumulative_elo:
        print(f"  - ELO evaluations: {sum(1 for x in cumulative_elo if x is not None)} data points")
    
    # Create plots
    if cumulative_elo and any(x is not None for x in cumulative_elo):
        plot_elo(data, output_dir)
        print(f"  ✓ Saved plot_elo.png")
    
    plot_win_rates(data, output_dir)
    print(f"  ✓ Saved plot_win_rates.png")
    
    plot_losses(data, output_dir)
    print(f"  ✓ Saved plot_losses.png")
    
    if policy_grad:
        plot_gradients(data, output_dir)
        print(f"  ✓ Saved plot_gradients.png")
    
    plot_buffer_and_lr(data, output_dir)
    print(f"  ✓ Saved plot_buffer_lr.png")
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
