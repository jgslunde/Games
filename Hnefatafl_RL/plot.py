#!/usr/bin/env python3
"""
Parse training output and create matplotlib plots of training progress.

This script parses the output.txt file(s) from train.py and generates five plots:
1. plot_elo.png - ELO vs random and ELO vs first version
2. plot_win_rates.png - Win rates (attacker/defender/draw) for self-play, 
   random evaluation, and self-evaluation
3. plot_losses.png - Policy and value losses per iteration (dual y-axes)
4. plot_buffer_time.png - Replay buffer size and iteration time (dual y-axes)
5. plot_epoch_losses.png - Policy and value losses per epoch across all iterations

All plots are saved in the same directory as the input file.

Usage: python plot.py <path_to_output.txt> [path_to_output2.txt] [...]

Example: python plot.py checkpoints_v05_fatman_asym_atkboost2_nodraw/output.txt
Example: python plot.py checkpoints_brandubh/jack_nothrone/output.txt checkpoints_brandubh/jack_nothrone/output2.txt
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
        'elo_difference': [],  # Individual ELO improvement per iteration
        
        # Training losses
        'policy_loss': [],
        'value_loss': [],
        'l2_loss': [],
        'policy_grad': [],
        'value_grad': [],
        'val_policy_loss': [],
        'val_value_loss': [],
        'val_l2_loss': [],
        
        # Epoch-level losses and gradients
        'epoch_steps': [],  # Global step counter across all iterations
        'epoch_policy_loss': [],
        'epoch_value_loss': [],
        'epoch_l2_loss': [],
        'epoch_policy_grad': [],
        'epoch_value_grad': [],
        'epoch_val_policy_loss': [],
        'epoch_val_value_loss': [],
        'epoch_val_l2_loss': [],
        
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
        
        # Final losses (old format)
        match = re.search(r'Final losses - Policy: ([\d.]+), Value: ([\d.]+)', line)
        if match:
            iterations_data[current_iteration]['policy_loss'] = float(match.group(1))
            iterations_data[current_iteration]['value_loss'] = float(match.group(2))
            continue
        
        # Final Summary (new format) - Training losses
        match = re.search(r'Training\s+-\s+Policy:\s+([\d.]+),\s+Value:\s+([\d.]+),\s+L2:\s+([\d.]+)', line)
        if match and 'Final Summary' in ''.join(lines[max(0, i-5):i]):
            iterations_data[current_iteration]['policy_loss'] = float(match.group(1))
            iterations_data[current_iteration]['value_loss'] = float(match.group(2))
            iterations_data[current_iteration]['l2_loss'] = float(match.group(3))
            continue
        
        # Final Summary (new format) - Gradients
        match = re.search(r'Gradients\s+-\s+Policy:\s+([\d.]+),\s+Value:\s+([\d.]+)', line)
        if match and 'Final Summary' in ''.join(lines[max(0, i-5):i]):
            iterations_data[current_iteration]['policy_grad'] = float(match.group(1))
            iterations_data[current_iteration]['value_grad'] = float(match.group(2))
            continue
        
        # Final Summary (new format) - Validation losses
        match = re.search(r'Validation\s+-\s+Policy:\s+([\d.]+),\s+Value:\s+([\d.]+),\s+L2:\s+([\d.]+)', line)
        if match and 'Final Summary' in ''.join(lines[max(0, i-5):i]):
            iterations_data[current_iteration]['val_policy_loss'] = float(match.group(1))
            iterations_data[current_iteration]['val_value_loss'] = float(match.group(2))
            iterations_data[current_iteration]['val_l2_loss'] = float(match.group(3))
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
        
        # Self-evaluation win rate (old format)
        match = re.search(r'New network win rate: ([\d.]+)% \((\d+)/(\d+)\)', line)
        if match:
            iterations_data[current_iteration]['selfeval_win_rate'] = float(match.group(1))
            iterations_data[current_iteration]['selfeval_total_wins'] = float(match.group(1))
            continue
        
        # Self-evaluation results (new format: "82W-46L-0D (64.1% score)")
        match = re.search(r'New network results: (\d+)W-(\d+)L-(\d+)D \(([\d.]+)% score\)', line)
        if match:
            wins = int(match.group(1))
            losses = int(match.group(2))
            draws = int(match.group(3))
            total_games = wins + losses + draws
            win_rate = float(match.group(4))
            iterations_data[current_iteration]['selfeval_win_rate'] = win_rate
            iterations_data[current_iteration]['selfeval_total_wins'] = win_rate
            continue
        
        # Cumulative ELO vs iteration 0
        match = re.search(r'Cumulative ELO \(vs iteration 0\): ([+-]?[\d.]+)', line)
        if match:
            iterations_data[current_iteration]['cumulative_elo_vs_first'] = float(match.group(1))
            continue
        
        # ELO difference (individual improvement when network is NOT accepted)
        match = re.search(r'ELO difference: ([+-]?[\d.]+)', line)
        if match:
            iterations_data[current_iteration]['elo_difference'] = float(match.group(1))
            continue
        
        # ELO gain (individual improvement when network IS accepted)
        match = re.search(r'ELO gain: ([+-]?[\d.]+)', line)
        if match:
            iterations_data[current_iteration]['elo_difference'] = float(match.group(1))
            continue
        
        # Total time for iteration
        match = re.search(r'Total:\s+([\d.]+)s', line)
        if match and 'Timing Summary:' in ''.join(lines[max(0, i-10):i]):
            iterations_data[current_iteration]['total_time'] = float(match.group(1))
            continue
        
        # Epoch-level losses (e.g., "  Epoch 1/10: policy=2.3456, value=0.5678, total=2.9134")
        # Old format
        match = re.search(r'Epoch (\d+)/\d+: policy=([\d.]+), value=([\d.]+)', line)
        if match:
            policy_loss = float(match.group(2))
            value_loss = float(match.group(3))
            data['epoch_steps'].append(global_epoch_step)
            data['epoch_policy_loss'].append(policy_loss)
            data['epoch_value_loss'].append(value_loss)
            data['epoch_policy_grad'].append(0.0)  # Not available in old format
            data['epoch_value_grad'].append(0.0)
            data['epoch_val_policy_loss'].append(0.0)  # Not available in old format
            data['epoch_val_value_loss'].append(0.0)
            global_epoch_step += 1
            continue
        
        # Epoch-level losses (new columnar format)
        # e.g., "      1 |   3.5152   0.5707   0.368899   3.6578 |   1.5566   0.7306 |   3.4958   0.5636   0.373004   3.6367"
        match = re.match(r'\s+(\d+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
        if match:
            epoch_num = int(match.group(1))
            policy_loss = float(match.group(2))
            value_loss = float(match.group(3))
            l2_loss = float(match.group(4))
            total_loss = float(match.group(5))
            policy_grad = float(match.group(6))
            value_grad = float(match.group(7))
            val_policy_loss = float(match.group(8))
            val_value_loss = float(match.group(9))
            val_l2_loss = float(match.group(10))
            val_total_loss = float(match.group(11))
            
            data['epoch_steps'].append(global_epoch_step)
            data['epoch_policy_loss'].append(policy_loss)
            data['epoch_value_loss'].append(value_loss)
            data['epoch_l2_loss'].append(l2_loss)
            data['epoch_policy_grad'].append(policy_grad)
            data['epoch_value_grad'].append(value_grad)
            data['epoch_val_policy_loss'].append(val_policy_loss)
            data['epoch_val_value_loss'].append(val_value_loss)
            data['epoch_val_l2_loss'].append(val_l2_loss)
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
            
            # Gradients and validation losses (new format)
            data['policy_grad'].append(iter_data.get('policy_grad', 0))
            data['value_grad'].append(iter_data.get('value_grad', 0))
            data['l2_loss'].append(iter_data.get('l2_loss', 0))
            data['val_policy_loss'].append(iter_data.get('val_policy_loss', 0))
            data['val_value_loss'].append(iter_data.get('val_value_loss', 0))
            data['val_l2_loss'].append(iter_data.get('val_l2_loss', 0))
            
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
            # Only include iterations that have both ELO and win rate data
            if 'cumulative_elo_vs_first' in iter_data and 'selfeval_attacker_wins' in iter_data:
                data['selfeval_iterations'].append(iteration)
                data['cumulative_elo_vs_first'].append(iter_data['cumulative_elo_vs_first'])
                if 'elo_difference' in iter_data:
                    data['elo_difference'].append(iter_data['elo_difference'])
                else:
                    data['elo_difference'].append(0.0)  # Default to 0 if not present
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


def merge_data_from_files(file_data_list):
    """
    Merge data from multiple output files.
    
    For cumulative ELO, adds the final ELO from each file to the values in subsequent files.
    For other metrics, simply concatenates the data.
    
    Args:
        file_data_list: List of data dictionaries from parse_output_file()
    
    Returns:
        Merged data dictionary
    """
    if len(file_data_list) == 1:
        return file_data_list[0]
    
    merged = {
        'iterations': [],
        'selfplay_attacker_wins': [],
        'selfplay_defender_wins': [],
        'selfplay_draws': [],
        'random_iterations': [],
        'random_attacker_wins': [],
        'random_defender_wins': [],
        'random_total_wins': [],
        'random_elo': [],
        'selfeval_iterations': [],
        'selfeval_attacker_wins': [],
        'selfeval_defender_wins': [],
        'selfeval_total_wins': [],
        'selfeval_win_rate': [],
        'cumulative_elo_vs_first': [],
        'elo_difference': [],
        'policy_loss': [],
        'value_loss': [],
        'l2_loss': [],
        'policy_grad': [],
        'value_grad': [],
        'val_policy_loss': [],
        'val_value_loss': [],
        'val_l2_loss': [],
        'epoch_steps': [],
        'epoch_policy_loss': [],
        'epoch_value_loss': [],
        'epoch_l2_loss': [],
        'epoch_policy_grad': [],
        'epoch_value_grad': [],
        'epoch_val_policy_loss': [],
        'epoch_val_value_loss': [],
        'epoch_val_l2_loss': [],
        'buffer_size': [],
        'total_time': [],
    }
    
    cumulative_elo_offset = 0.0
    
    for file_idx, data in enumerate(file_data_list):
        # For cumulative ELO, we need to add the offset from previous files
        if data['cumulative_elo_vs_first']:
            # Add offset to all cumulative ELO values
            adjusted_elo = [elo + cumulative_elo_offset for elo in data['cumulative_elo_vs_first']]
            merged['cumulative_elo_vs_first'].extend(adjusted_elo)
            merged['selfeval_iterations'].extend(data['selfeval_iterations'])
            merged['elo_difference'].extend(data['elo_difference'])
            
            # Update offset for next file (use the last ELO value from current file)
            cumulative_elo_offset = adjusted_elo[-1]
        
        # For other self-evaluation metrics
        merged['selfeval_attacker_wins'].extend(data['selfeval_attacker_wins'])
        merged['selfeval_defender_wins'].extend(data['selfeval_defender_wins'])
        merged['selfeval_total_wins'].extend(data['selfeval_total_wins'])
        
        # Simple concatenation for other metrics
        merged['iterations'].extend(data['iterations'])
        merged['selfplay_attacker_wins'].extend(data['selfplay_attacker_wins'])
        merged['selfplay_defender_wins'].extend(data['selfplay_defender_wins'])
        merged['selfplay_draws'].extend(data['selfplay_draws'])
        
        merged['random_iterations'].extend(data['random_iterations'])
        merged['random_attacker_wins'].extend(data['random_attacker_wins'])
        merged['random_defender_wins'].extend(data['random_defender_wins'])
        merged['random_total_wins'].extend(data['random_total_wins'])
        merged['random_elo'].extend(data['random_elo'])
        
        merged['policy_loss'].extend(data['policy_loss'])
        merged['value_loss'].extend(data['value_loss'])
        merged['l2_loss'].extend(data['l2_loss'])
        merged['policy_grad'].extend(data['policy_grad'])
        merged['value_grad'].extend(data['value_grad'])
        merged['val_policy_loss'].extend(data['val_policy_loss'])
        merged['val_value_loss'].extend(data['val_value_loss'])
        merged['val_l2_loss'].extend(data['val_l2_loss'])
        
        merged['epoch_policy_loss'].extend(data['epoch_policy_loss'])
        merged['epoch_value_loss'].extend(data['epoch_value_loss'])
        merged['epoch_l2_loss'].extend(data['epoch_l2_loss'])
        merged['epoch_policy_grad'].extend(data['epoch_policy_grad'])
        merged['epoch_value_grad'].extend(data['epoch_value_grad'])
        merged['epoch_val_policy_loss'].extend(data['epoch_val_policy_loss'])
        merged['epoch_val_value_loss'].extend(data['epoch_val_value_loss'])
        merged['epoch_val_l2_loss'].extend(data['epoch_val_l2_loss'])
        # Adjust epoch steps to be cumulative across files
        if merged['epoch_steps']:
            max_step = merged['epoch_steps'][-1] + 1
            adjusted_steps = [step + max_step for step in data['epoch_steps']]
            merged['epoch_steps'].extend(adjusted_steps)
        else:
            merged['epoch_steps'].extend(data['epoch_steps'])
        
        merged['buffer_size'].extend(data['buffer_size'])
        merged['total_time'].extend(data['total_time'])
    
    return merged


def plot_elo(data, output_dir):
    """Plot 1: ELO versus first version and individual ELO differences."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot cumulative ELO on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Cumulative ELO vs First Version', fontsize=12, color=color1)
    if data['cumulative_elo_vs_first']:
        ax1.plot(data['selfeval_iterations'], data['cumulative_elo_vs_first'], 's-', 
                color=color1, label='Cumulative ELO', linewidth=2, markersize=5)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot individual ELO differences on right y-axis
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Individual ELO Difference', fontsize=12, color=color2)
    if data['elo_difference']:
        ax2.plot(data['selfeval_iterations'], data['elo_difference'], 'o-', 
                color=color2, label='ELO Difference', linewidth=2, markersize=4, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(y=0, color=color2, linestyle='--', linewidth=1, alpha=0.5)
    
    ax1.set_title('ELO Ratings During Training', fontsize=14, fontweight='bold')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_elo.png'), dpi=150)
    plt.close()


def plot_win_rates(data, output_dir):
    """Plot 2: Win rates for self-play and self-evaluation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
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
    
    # Self-evaluation win rates (network vs network)
    ax = axes[1]
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
    """Plot 3: Policy, value, and L2 losses with triple y-axes, showing training and validation."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    iterations = data['iterations']
    
    # Policy loss on left y-axis
    color1 = 'tab:blue'
    color1_val = 'lightblue'
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Policy Loss', fontsize=12, color=color1)
    if data['policy_loss']:
        ax1.plot(iterations, data['policy_loss'], 'o-', color=color1, label='Policy Loss (Train)', linewidth=2, markersize=4)
    if data['val_policy_loss']:
        ax1.plot(iterations, data['val_policy_loss'], 'o--', color=color1_val, label='Policy Loss (Val)', linewidth=2, markersize=4, alpha=1.0)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Value loss on right y-axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    color2_val = 'lightcoral'
    ax2.set_ylabel('Value Loss', fontsize=12, color=color2)
    if data['value_loss']:
        ax2.plot(iterations, data['value_loss'], 's-', color=color2, label='Value Loss (Train)', linewidth=2, markersize=4)
    if data['val_value_loss']:
        ax2.plot(iterations, data['val_value_loss'], 's--', color=color2_val, label='Value Loss (Val)', linewidth=2, markersize=4, alpha=1.0)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # L2 loss on far right y-axis
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color3 = 'tab:green'
    color3_val = 'lightgreen'
    ax3.set_ylabel('L2 Loss', fontsize=12, color=color3)
    if data['l2_loss']:
        ax3.plot(iterations, data['l2_loss'], '^-', color=color3, label='L2 Loss (Train)', linewidth=2, markersize=4)
    if data['val_l2_loss']:
        ax3.plot(iterations, data['val_l2_loss'], '^--', color=color3_val, label='L2 Loss (Val)', linewidth=2, markersize=4, alpha=1.0)
    ax3.tick_params(axis='y', labelcolor=color3)
    
    # Add title
    ax1.set_title('Training Losses (Per Iteration)', fontsize=14, fontweight='bold')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_losses.png'), dpi=150)
    plt.close()


def plot_epoch_losses(data, output_dir):
    """Plot 5: Epoch-level policy, value, and L2 losses with triple y-axes across all iterations, showing training and validation."""
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    epoch_steps = data['epoch_steps']
    
    # Policy loss on left y-axis
    color1 = 'tab:blue'
    color1_val = 'lightblue'
    ax1.set_xlabel('Training Epoch (Cumulative)', fontsize=12)
    ax1.set_ylabel('Policy Loss', fontsize=12, color=color1)
    if data['epoch_policy_loss']:
        ax1.plot(epoch_steps, data['epoch_policy_loss'], '-', color=color1, 
                label='Policy Loss (Train)', linewidth=1.5, alpha=0.8)
    if data['epoch_val_policy_loss']:
        ax1.plot(epoch_steps, data['epoch_val_policy_loss'], '--', color=color1_val, 
                label='Policy Loss (Val)', linewidth=1.5, alpha=1.0)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Value loss on right y-axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    color2_val = 'lightcoral'
    ax2.set_ylabel('Value Loss', fontsize=12, color=color2)
    if data['epoch_value_loss']:
        ax2.plot(epoch_steps, data['epoch_value_loss'], '-', color=color2, 
                label='Value Loss (Train)', linewidth=1.5, alpha=0.8)
    if data['epoch_val_value_loss']:
        ax2.plot(epoch_steps, data['epoch_val_value_loss'], '--', color=color2_val, 
                label='Value Loss (Val)', linewidth=1.5, alpha=1.0)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # L2 loss on far right y-axis
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color3 = 'tab:green'
    color3_val = 'lightgreen'
    ax3.set_ylabel('L2 Loss', fontsize=12, color=color3)
    if data['epoch_l2_loss']:
        ax3.plot(epoch_steps, data['epoch_l2_loss'], '-', color=color3, 
                label='L2 Loss (Train)', linewidth=1.5, alpha=0.8)
    if data['epoch_val_l2_loss']:
        ax3.plot(epoch_steps, data['epoch_val_l2_loss'], '--', color=color3_val, 
                label='L2 Loss (Val)', linewidth=1.5, alpha=1.0)
    ax3.tick_params(axis='y', labelcolor=color3)
    
    # Add title
    ax1.set_title('Training Losses (Per Epoch - All Iterations)', fontsize=14, fontweight='bold')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_epoch_losses.png'), dpi=150)
    plt.close()


def plot_gradients(data, output_dir):
    """Plot 6: Per-iteration gradient norms for policy and value heads."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = data['iterations']
    
    if data['policy_grad']:
        ax.plot(iterations, data['policy_grad'], 'o-', color='tab:blue', 
               label='Policy Gradient Norm', linewidth=2, markersize=4)
    if data['value_grad']:
        ax.plot(iterations, data['value_grad'], 's-', color='tab:red', 
               label='Value Gradient Norm', linewidth=2, markersize=4)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norms (Per Iteration)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_gradients.png'), dpi=150)
    plt.close()


def plot_epoch_gradients(data, output_dir):
    """Plot 7: Per-epoch gradient norms for policy and value heads across all iterations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epoch_steps = data['epoch_steps']
    
    if data['epoch_policy_grad']:
        ax.plot(epoch_steps, data['epoch_policy_grad'], '-', color='tab:blue', 
               label='Policy Gradient Norm', linewidth=1.5, alpha=0.8)
    if data['epoch_value_grad']:
        ax.plot(epoch_steps, data['epoch_value_grad'], '-', color='tab:red', 
               label='Value Gradient Norm', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Training Epoch (Cumulative)', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norms (Per Epoch - All Iterations)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_epoch_gradients.png'), dpi=150)
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
    ax1.set_ylim(bottom=0)
    # Format y-axis with thousands separator
    ax1.ticklabel_format(style='plain', axis='y')
    
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Total Time (s)', fontsize=12, color=color2)
    if data['total_time']:
        ax2.plot(iterations, data['total_time'], 's-', color=color2, label='Total Time', linewidth=2, markersize=4)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)
    
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
    if len(sys.argv) < 2:
        print("Usage: python plot.py <path_to_output.txt> [path_to_output2.txt] [...]")
        sys.exit(1)
    
    filepaths = sys.argv[1:]
    
    # Check that all files exist
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' not found.")
            sys.exit(1)
    
    # Parse all files
    print(f"Parsing {len(filepaths)} file(s)...")
    file_data_list = []
    for filepath in filepaths:
        print(f"  - {filepath}")
        data = parse_output_file(filepath)
        file_data_list.append(data)
    
    # Merge data from multiple files if necessary
    if len(file_data_list) > 1:
        print(f"\nMerging data from {len(file_data_list)} files...")
        data = merge_data_from_files(file_data_list)
    else:
        data = file_data_list[0]
    
    # Fill missing values for metrics not recorded every iteration
    data = fill_missing_values(data)
    
    # Output directory is the same as the first input file
    output_dir = os.path.dirname(filepaths[0])
    
    print(f"\nCreating plots...")
    print(f"  - Found {len(data['iterations'])} iterations")
    print(f"  - Random evaluations: {len(data['random_elo'])} data points")
    print(f"  - Self-evaluations: {len(data['cumulative_elo_vs_first'])} data points")
    print(f"  - Policy loss: {len(data['policy_loss'])} data points")
    print(f"  - Value loss: {len(data['value_loss'])} data points")
    print(f"  - Gradients: {len(data['policy_grad'])} data points")
    print(f"  - Epoch-level losses: {len(data['epoch_policy_loss'])} epochs")
    print(f"  - Epoch-level gradients: {len(data['epoch_policy_grad'])} epochs")
    
    plot_elo(data, output_dir)
    print("  ✓ Saved plot_elo.png")
    
    plot_win_rates(data, output_dir)
    print("  ✓ Saved plot_win_rates.png")
    
    plot_losses(data, output_dir)
    print("  ✓ Saved plot_losses.png")
    
    if data['policy_grad']:
        plot_gradients(data, output_dir)
        print("  ✓ Saved plot_gradients.png")
    
    plot_buffer_and_time(data, output_dir)
    print("  ✓ Saved plot_buffer_time.png")
    
    if data['epoch_policy_loss']:
        plot_epoch_losses(data, output_dir)
        print("  ✓ Saved plot_epoch_losses.png")
    
    if data['epoch_policy_grad']:
        plot_epoch_gradients(data, output_dir)
        print("  ✓ Saved plot_epoch_gradients.png")
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
