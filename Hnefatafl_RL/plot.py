#!/usr/bin/env python3
"""
Parse training output and create matplotlib plots of training progress.

This script parses the output.txt file from train.py and generates five plots:
1. plot_elo.png - ELO vs random and ELO vs first version
2. plot_win_rates.png - Win rates (attacker/defender/draw) for self-play, 
   random evaluation, and self-evaluation
3. plot_losses.png - Policy and value losses per iteration (dual y-axes)
4. plot_buffer_time.png - Replay buffer size and iteration time (dual y-axes)
5. plot_epoch_losses.png - Policy and value losses per epoch across all iterations

All plots are saved in the same directory as the input file.

Usage: python plot.py <path_to_output.txt>

Example: python plot.py checkpoints_v05_fatman_asym_atkboost2_nodraw/output.txt
"""

import sys
import re
import os
import matplotlib.pyplot as plt
import numpy as np


def parse_output_file(filepath):
    """Parse the training output file and extract all relevant metrics."""
    
    # Store data per iteration, then compile at the end
    iterations_data = {}
    
    data = {
        'iterations': [],
        
        # Self-play results
        'selfplay_attacker_wins': [],
        'selfplay_defender_wins': [],
        'selfplay_draws': [],
        
        # Random evaluation results - with iteration tracking
        'random_iterations': [],
        'random_attacker_wins': [],
        'random_defender_wins': [],
        'random_total_wins': [],
        'random_elo': [],
        
        # Self-evaluation results (network vs network) - with iteration tracking
        'selfeval_iterations': [],
        'selfeval_attacker_wins': [],
        'selfeval_defender_wins': [],
        'selfeval_total_wins': [],
        'selfeval_win_rate': [],
        'cumulative_elo_vs_first': [],
        
        # Training losses
        'policy_loss': [],
        'value_loss': [],
        
        # Epoch-level losses
        'epoch_steps': [],  # Global step counter across all iterations
        'epoch_policy_loss': [],
        'epoch_value_loss': [],
        
        # Buffer and time
        'buffer_size': [],
        'total_time': [],
    }
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    current_iteration = None
    global_epoch_step = 0  # Track epochs across all iterations
    
    for i, line in enumerate(lines):
        # Extract iteration number
        match = re.match(r'^Iteration (\d+)/\d+', line)
        if match:
            current_iteration = int(match.group(1))
            if current_iteration not in iterations_data:
                iterations_data[current_iteration] = {}
            continue
        
        if current_iteration is None:
            continue
        
        # Self-play results
        match = re.search(r'Results: (\d+) attacker wins \(([\d.]+)%\), (\d+) defender wins \(([\d.]+)%\), (\d+) draws', line)
        if match:
            iterations_data[current_iteration]['selfplay_attacker_wins'] = float(match.group(2))
            iterations_data[current_iteration]['selfplay_defender_wins'] = float(match.group(4))
            iterations_data[current_iteration]['selfplay_draws'] = int(match.group(5)) / (int(match.group(1)) + int(match.group(3)) + int(match.group(5))) * 100
            continue
        
        # Replay buffer size
        match = re.search(r'Replay buffer size: (\d+)', line)
        if match:
            iterations_data[current_iteration]['buffer_size'] = int(match.group(1))
            continue
        
        # Final losses
        match = re.search(r'Final losses - Policy: ([\d.]+), Value: ([\d.]+)', line)
        if match:
            iterations_data[current_iteration]['policy_loss'] = float(match.group(1))
            iterations_data[current_iteration]['value_loss'] = float(match.group(2))
            continue
        
        # Random evaluation results
        match = re.search(r'As Attacker: (\d+)/(\d+) wins', line)
        if match and '[3/4] Evaluating vs Random' in ''.join(lines[max(0, i-10):i]):
            attacker_wins = int(match.group(1))
            attacker_total = int(match.group(2))
            # Look for defender wins in next few lines
            for j in range(i+1, min(i+5, len(lines))):
                defender_match = re.search(r'As Defender: (\d+)/(\d+) wins', lines[j])
                if defender_match:
                    defender_wins = int(defender_match.group(1))
                    defender_total = int(defender_match.group(2))
                    iterations_data[current_iteration]['random_attacker_wins'] = attacker_wins / attacker_total * 100
                    iterations_data[current_iteration]['random_defender_wins'] = defender_wins / defender_total * 100
                    break
            continue
        
        # Random evaluation total and ELO
        match = re.search(r'Total: (\d+)/(\d+) wins \(([\d.]+)%\)', line)
        if match and '[3/4] Evaluating vs Random' in ''.join(lines[max(0, i-15):i]):
            iterations_data[current_iteration]['random_total_wins'] = float(match.group(3))
            # Look for ELO in next few lines
            for j in range(i+1, min(i+5, len(lines))):
                elo_match = re.search(r'ELO vs Random: ([+-]?\d+)', lines[j])
                if elo_match:
                    iterations_data[current_iteration]['random_elo'] = int(elo_match.group(1))
                    break
            continue
        
        # Self-evaluation results (network vs network)
        match = re.search(r'As Attacker: (\d+)/(\d+) wins', line)
        if match and '[4/4] Evaluating new vs old' in ''.join(lines[max(0, i-10):i]):
            attacker_wins = int(match.group(1))
            attacker_total = int(match.group(2))
            # Look for defender wins in next few lines
            for j in range(i+1, min(i+5, len(lines))):
                defender_match = re.search(r'As Defender: (\d+)/(\d+) wins', lines[j])
                if defender_match:
                    defender_wins = int(defender_match.group(1))
                    defender_total = int(defender_match.group(2))
                    iterations_data[current_iteration]['selfeval_attacker_wins'] = attacker_wins / attacker_total * 100
                    iterations_data[current_iteration]['selfeval_defender_wins'] = defender_wins / defender_total * 100
                    break
            continue
        
        # Self-evaluation win rate
        match = re.search(r'New network win rate: ([\d.]+)% \((\d+)/(\d+)\)', line)
        if match:
            iterations_data[current_iteration]['selfeval_win_rate'] = float(match.group(1))
            iterations_data[current_iteration]['selfeval_total_wins'] = float(match.group(1))
            continue
        
        # Cumulative ELO vs iteration 0
        match = re.search(r'Cumulative ELO \(vs iteration 0\): ([+-]?[\d.]+)', line)
        if match:
            iterations_data[current_iteration]['cumulative_elo_vs_first'] = float(match.group(1))
            continue
        
        # Total time for iteration
        match = re.search(r'Total:\s+([\d.]+)s', line)
        if match and 'Timing Summary:' in ''.join(lines[max(0, i-10):i]):
            iterations_data[current_iteration]['total_time'] = float(match.group(1))
            continue
        
        # Epoch-level losses (e.g., "  Epoch 1/10: policy=2.3456, value=0.5678, total=2.9134")
        match = re.search(r'Epoch (\d+)/\d+: policy=([\d.]+), value=([\d.]+)', line)
        if match:
            policy_loss = float(match.group(2))
            value_loss = float(match.group(3))
            data['epoch_steps'].append(global_epoch_step)
            data['epoch_policy_loss'].append(policy_loss)
            data['epoch_value_loss'].append(value_loss)
            global_epoch_step += 1
            continue
    
    # Now compile data only for complete iterations (those with both results and losses)
    for iteration in sorted(iterations_data.keys()):
        iter_data = iterations_data[iteration]
        
        # Check if iteration has minimum required data (results and losses)
        if 'selfplay_attacker_wins' in iter_data and 'policy_loss' in iter_data:
            data['iterations'].append(iteration)
            data['selfplay_attacker_wins'].append(iter_data['selfplay_attacker_wins'])
            data['selfplay_defender_wins'].append(iter_data['selfplay_defender_wins'])
            data['selfplay_draws'].append(iter_data['selfplay_draws'])
            data['policy_loss'].append(iter_data['policy_loss'])
            data['value_loss'].append(iter_data['value_loss'])
            
            # Optional data - always append to keep arrays aligned
            if 'buffer_size' in iter_data:
                data['buffer_size'].append(iter_data['buffer_size'])
            else:
                # Use previous value if available, otherwise 0
                data['buffer_size'].append(data['buffer_size'][-1] if data['buffer_size'] else 0)
            
            if 'total_time' in iter_data:
                data['total_time'].append(iter_data['total_time'])
            else:
                # Use previous value if available, otherwise 0
                data['total_time'].append(data['total_time'][-1] if data['total_time'] else 0)
            
            # Random evaluation data (tracked separately)
            if 'random_elo' in iter_data:
                data['random_iterations'].append(iteration)
                data['random_elo'].append(iter_data['random_elo'])
                data['random_total_wins'].append(iter_data.get('random_total_wins', 0))
                data['random_attacker_wins'].append(iter_data.get('random_attacker_wins', 0))
                data['random_defender_wins'].append(iter_data.get('random_defender_wins', 0))
            
            # Self-evaluation data (tracked separately)
            if 'cumulative_elo_vs_first' in iter_data:
                data['selfeval_iterations'].append(iteration)
                data['cumulative_elo_vs_first'].append(iter_data['cumulative_elo_vs_first'])
                if 'selfeval_total_wins' in iter_data:
                    data['selfeval_total_wins'].append(iter_data['selfeval_total_wins'])
                    data['selfeval_attacker_wins'].append(iter_data.get('selfeval_attacker_wins', 0))
                    data['selfeval_defender_wins'].append(iter_data.get('selfeval_defender_wins', 0))
    
    return data


def fill_missing_values(data):
    """Fill in missing values for metrics that aren't recorded every iteration."""
    # This function is no longer needed since we'll plot only the iterations where data exists
    # We just need to make sure the iteration indices align with the data
    return data


def plot_elo(data, output_dir):
    """Plot 1: ELO versus random and ELO versus first version."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if data['random_elo']:
        ax.plot(data['random_iterations'], data['random_elo'], 'o-', label='ELO vs Random', linewidth=2, markersize=4)
    
    if data['cumulative_elo_vs_first']:
        ax.plot(data['selfeval_iterations'], data['cumulative_elo_vs_first'], 's-', label='ELO vs First Version', linewidth=2, markersize=4)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('ELO Rating', fontsize=12)
    ax.set_title('ELO Ratings During Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_elo.png'), dpi=150)
    plt.close()


def plot_win_rates(data, output_dir):
    """Plot 2: Win rates for self-play, random, and self-evaluation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    iterations = data['iterations']
    
    # Self-play win rates
    ax = axes[0]
    if data['selfplay_attacker_wins']:
        ax.plot(iterations, data['selfplay_attacker_wins'], 'o-', label='Attacker', linewidth=2, markersize=4)
    if data['selfplay_defender_wins']:
        ax.plot(iterations, data['selfplay_defender_wins'], 's-', label='Defender', linewidth=2, markersize=4)
    if data['selfplay_draws']:
        ax.plot(iterations, data['selfplay_draws'], '^-', label='Draw', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Self-Play Win Rates', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Random evaluation win rates
    ax = axes[1]
    if data['random_attacker_wins']:
        ax.plot(data['random_iterations'], data['random_attacker_wins'], 'o-', label='Attacker', linewidth=2, markersize=4)
    if data['random_defender_wins']:
        ax.plot(data['random_iterations'], data['random_defender_wins'], 's-', label='Defender', linewidth=2, markersize=4)
    # Draw rate for random is not explicitly stated, so we calculate it
    if data['random_attacker_wins'] and data['random_defender_wins']:
        draw_rate = [100 - (a + d) / 2 for a, d in zip(data['random_attacker_wins'], data['random_defender_wins'])]
        ax.plot(data['random_iterations'], draw_rate, '^-', label='Draw (approx)', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Random Evaluation Win Rates', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Self-evaluation win rates (network vs network)
    ax = axes[2]
    if data['selfeval_attacker_wins']:
        ax.plot(data['selfeval_iterations'], data['selfeval_attacker_wins'], 
                'o-', label='Attacker', linewidth=2, markersize=4)
    if data['selfeval_defender_wins']:
        ax.plot(data['selfeval_iterations'], data['selfeval_defender_wins'], 
                's-', label='Defender', linewidth=2, markersize=4)
    # Calculate draw rate
    if data['selfeval_attacker_wins'] and data['selfeval_defender_wins']:
        draw_rate = [100 - (a + d) / 2 for a, d in zip(data['selfeval_attacker_wins'], data['selfeval_defender_wins'])]
        ax.plot(data['selfeval_iterations'], draw_rate, '^-', label='Draw (approx)', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Self-Evaluation Win Rates', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_win_rates.png'), dpi=150)
    plt.close()


def plot_losses(data, output_dir):
    """Plot 3: Policy loss and value loss with dual y-axes."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    iterations = data['iterations']
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Policy Loss', fontsize=12, color=color1)
    if data['policy_loss']:
        ax1.plot(iterations, data['policy_loss'], 'o-', color=color1, label='Policy Loss', linewidth=2, markersize=4)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Value Loss', fontsize=12, color=color2)
    if data['value_loss']:
        ax2.plot(iterations, data['value_loss'], 's-', color=color2, label='Value Loss', linewidth=2, markersize=4)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title
    ax1.set_title('Training Losses (Per Iteration)', fontsize=14, fontweight='bold')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_losses.png'), dpi=150)
    plt.close()


def plot_epoch_losses(data, output_dir):
    """Plot 5: Epoch-level policy loss and value loss with dual y-axes across all iterations."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    epoch_steps = data['epoch_steps']
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Training Epoch (Cumulative)', fontsize=12)
    ax1.set_ylabel('Policy Loss', fontsize=12, color=color1)
    if data['epoch_policy_loss']:
        ax1.plot(epoch_steps, data['epoch_policy_loss'], '-', color=color1, 
                label='Policy Loss', linewidth=1.5, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Value Loss', fontsize=12, color=color2)
    if data['epoch_value_loss']:
        ax2.plot(epoch_steps, data['epoch_value_loss'], '-', color=color2, 
                label='Value Loss', linewidth=1.5, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title
    ax1.set_title('Training Losses (Per Epoch - All Iterations)', fontsize=14, fontweight='bold')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_epoch_losses.png'), dpi=150)
    plt.close()


def plot_buffer_and_time(data, output_dir):
    """Plot 4: Buffer size and total time with dual y-axes."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    iterations = data['iterations']
    
    color1 = 'tab:green'
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Buffer Size', fontsize=12, color=color1)
    if data['buffer_size']:
        ax1.plot(iterations, data['buffer_size'], 'o-', color=color1, label='Buffer Size', linewidth=2, markersize=4)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    # Format y-axis with thousands separator
    ax1.ticklabel_format(style='plain', axis='y')
    
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Total Time (s)', fontsize=12, color=color2)
    if data['total_time']:
        ax2.plot(iterations, data['total_time'], 's-', color=color2, label='Total Time', linewidth=2, markersize=4)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title
    ax1.set_title('Buffer Size and Iteration Time', fontsize=14, fontweight='bold')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_buffer_time.png'), dpi=150)
    plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot.py <path_to_output.txt>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    
    print(f"Parsing {filepath}...")
    data = parse_output_file(filepath)
    
    # Fill missing values for metrics not recorded every iteration
    data = fill_missing_values(data)
    
    # Output directory is the same as the input file
    output_dir = os.path.dirname(filepath)
    
    print(f"Creating plots...")
    print(f"  - Found {len(data['iterations'])} iterations")
    print(f"  - Random evaluations: {len(data['random_elo'])} data points")
    print(f"  - Self-evaluations: {len(data['cumulative_elo_vs_first'])} data points")
    print(f"  - Policy loss: {len(data['policy_loss'])} data points")
    print(f"  - Value loss: {len(data['value_loss'])} data points")
    print(f"  - Epoch-level losses: {len(data['epoch_policy_loss'])} epochs")
    
    plot_elo(data, output_dir)
    print(f"  ✓ Saved plot_elo.png")
    
    plot_win_rates(data, output_dir)
    print(f"  ✓ Saved plot_win_rates.png")
    
    plot_losses(data, output_dir)
    print(f"  ✓ Saved plot_losses.png")
    
    plot_buffer_and_time(data, output_dir)
    print(f"  ✓ Saved plot_buffer_time.png")
    
    if data['epoch_policy_loss']:
        plot_epoch_losses(data, output_dir)
        print(f"  ✓ Saved plot_epoch_losses.png")
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
