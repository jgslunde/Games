#!/usr/bin/env python3
"""
Plot ELO ratings from evaluation results.

Usage:
    python plot_elo.py <results_json>
    python plot_elo.py checkpoints_brandubh/fatman_nothrone/elo_evaluation_results.json
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_elo_progression(results_path: str):
    """
    Plot ELO ratings as a function of training iteration.
    
    Args:
        results_path: Path to elo_evaluation_results.json
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract checkpoint data
    checkpoints = data['checkpoints']
    
    # Sort by iteration (if available)
    checkpoints_with_iter = [cp for cp in checkpoints if cp['iteration'] is not None]
    
    if not checkpoints_with_iter:
        print("Warning: No checkpoints have iteration numbers. Plotting by index instead.")
        iterations = list(range(len(checkpoints)))
        elo_ratings = [cp['elo'] for cp in checkpoints]
        xlabel = "Checkpoint Index"
    else:
        checkpoints_with_iter.sort(key=lambda x: x['iteration'])
        iterations = [cp['iteration'] for cp in checkpoints_with_iter]
        elo_ratings = [cp['elo'] for cp in checkpoints_with_iter]
        xlabel = "Training Iteration"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot ELO ratings
    ax.plot(iterations, elo_ratings, 'o-', linewidth=2, markersize=6, 
            color='steelblue', label='ELO Rating')
    
    # Add horizontal line at base ELO
    base_elo = data['metadata'].get('base_elo', 1500)
    ax.axhline(y=base_elo, color='gray', linestyle='--', linewidth=1, 
               label=f'Base ELO ({base_elo:.0f})')
    
    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('ELO Rating', fontsize=12, fontweight='bold')
    
    game = data['metadata'].get('game', 'Unknown')
    ax.set_title(f'ELO Rating Progression - {game.capitalize()}', 
                 fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add statistics text
    mean_elo = np.mean(elo_ratings)
    std_elo = np.std(elo_ratings)
    min_elo = np.min(elo_ratings)
    max_elo = np.max(elo_ratings)
    
    stats_text = (
        f"Statistics:\n"
        f"Mean: {mean_elo:.1f}\n"
        f"Std:  {std_elo:.1f}\n"
        f"Min:  {min_elo:.1f}\n"
        f"Max:  {max_elo:.1f}\n"
        f"Spread: {max_elo - min_elo:.1f}"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add metadata text
    num_games = data['metadata'].get('num_games_per_match', 'N/A')
    checkpoint_dist = data['metadata'].get('checkpoint_distance', 'N/A')
    total_matches = data['metadata'].get('total_matches', 'N/A')
    
    metadata_text = (
        f"Evaluation Settings:\n"
        f"Games per match: {num_games}\n"
        f"Checkpoint distance: {checkpoint_dist}\n"
        f"Total matches: {total_matches}"
    )
    
    ax.text(0.98, 0.02, metadata_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(results_path).parent / 'elo_progression.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show plot
    plt.show()


def plot_match_matrix(results_path: str):
    """
    Plot match results as a matrix/heatmap.
    
    Args:
        results_path: Path to elo_evaluation_results.json
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    checkpoints = data['checkpoints']
    matches = data['matches']
    
    n = len(checkpoints)
    
    # Create matrix of win percentages
    # win_matrix[i][j] = percentage of games won by checkpoint i against checkpoint j
    win_matrix = np.full((n, n), np.nan)
    
    for match in matches:
        i = match['checkpoint1_index']
        j = match['checkpoint2_index']
        wins1 = match['checkpoint1_wins']
        wins2 = match['checkpoint2_wins']
        total = match['total_games']
        
        if total > 0:
            win_matrix[i, j] = 100 * wins1 / total
            win_matrix[j, i] = 100 * wins2 / total
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(win_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Win Percentage (%)', rotation=270, labelpad=20, fontweight='bold')
    
    # Add labels
    ax.set_xlabel('Opponent Checkpoint Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Checkpoint Index', fontsize=12, fontweight='bold')
    
    game = data['metadata'].get('game', 'Unknown')
    ax.set_title(f'Match Results Matrix - {game.capitalize()}', 
                 fontsize=14, fontweight='bold')
    
    # Set tick labels
    if n <= 30:
        # Show all ticks for small matrices
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        
        # Use iteration numbers if available
        checkpoints_sorted = sorted(checkpoints, key=lambda x: x['index'])
        labels = []
        for cp in checkpoints_sorted:
            if cp['iteration'] is not None:
                labels.append(str(cp['iteration']))
            else:
                labels.append(str(cp['index']))
        
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        # Show fewer ticks for large matrices
        step = n // 10
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], rotation=90)
        ax.set_yticklabels([str(t) for t in ticks])
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(results_path).parent / 'match_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot ELO evaluation results")
    parser.add_argument("results_file", type=str,
                       help="Path to elo_evaluation_results.json")
    parser.add_argument("--matrix", action="store_true",
                       help="Plot match results matrix instead of ELO progression")
    parser.add_argument("--both", action="store_true",
                       help="Plot both ELO progression and match matrix")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"Error: File not found: {args.results_file}")
        return
    
    # Plot requested figures
    if args.both:
        plot_elo_progression(args.results_file)
        plot_match_matrix(args.results_file)
    elif args.matrix:
        plot_match_matrix(args.results_file)
    else:
        plot_elo_progression(args.results_file)


if __name__ == "__main__":
    main()
