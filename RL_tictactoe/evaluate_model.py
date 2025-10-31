#!/usr/bin/env python3
"""
Evaluate a trained model against a random player.
Usage: python evaluate_model.py [checkpoint_path]
"""

import sys
import torch
from TTTnet import TicTacToeNet
from evaluate import evaluate_against_random, print_elo_report


def load_model(checkpoint_path):
    """Load a model from checkpoint."""
    network = TicTacToeNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    return network, checkpoint


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <checkpoint_path>")
        print("\nExample:")
        print("  python evaluate_model.py checkpoints/final_model.pt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    print(f"Loading model from {checkpoint_path}...")
    network, checkpoint = load_model(checkpoint_path)
    
    # Print checkpoint info
    iteration = checkpoint.get('iteration', 'Unknown')
    print(f"Model iteration: {iteration}")
    
    if 'metadata' in checkpoint:
        metadata = checkpoint['metadata']
        if 'estimated_elo' in metadata:
            print(f"Previous ELO estimate: {metadata['estimated_elo']:.0f}")
    
    print("\n" + "="*60)
    print("EVALUATION AGAINST RANDOM PLAYER")
    print("="*60)
    
    # Evaluate
    results = evaluate_against_random(
        network,
        num_games=200,  # More games for better estimate
        num_mcts_sims=50,
        verbose=True
    )
    
    # Print ELO report
    estimated_elo = print_elo_report(results, opponent_elo=1000)
    
    print(f"\nFinal ELO estimate: {estimated_elo:.0f}")


if __name__ == "__main__":
    main()
