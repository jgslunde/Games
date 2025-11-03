"""
Visualize training progress from training_history.json
"""

import json
import os

def print_training_summary(history_file="checkpoints/training_history.json"):
    """Print a summary of training progress."""
    
    if not os.path.exists(history_file):
        print(f"No training history found at {history_file}")
        print("Train a model first with: python3 train.py")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    if not history['iterations']:
        print("No training data found")
        return
    
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    # Basic info
    num_iters = len(history['iterations'])
    print(f"\nTotal iterations: {num_iters}")
    
    # Loss progression
    if history['policy_loss']:
        print("\nLoss progression:")
        print(f"  Policy loss:  {history['policy_loss'][0]:.4f} → {history['policy_loss'][-1]:.4f}")
        print(f"  Value loss:   {history['value_loss'][0]:.4f} → {history['value_loss'][-1]:.4f}")
        print(f"  Total loss:   {history['total_loss'][0]:.4f} → {history['total_loss'][-1]:.4f}")
    
    # Buffer size
    if history['buffer_size']:
        print(f"\nReplay buffer: {history['buffer_size'][-1]} samples")
    
    # Performance vs random
    if history['vs_random_win_rate']:
        print("\nPerformance vs Random Player:")
        print(f"  Win rate:     {100*history['vs_random_win_rate'][0]:.1f}% → {100*history['vs_random_win_rate'][-1]:.1f}%")
        print(f"  ELO rating:   {history['vs_random_elo'][0]:+.0f} → {history['vs_random_elo'][-1]:+.0f}")
        
        if history['vs_random_attacker_wins']:
            latest_attacker = history['vs_random_attacker_wins'][-1]
            latest_defender = history['vs_random_defender_wins'][-1]
            print(f"  Latest games: {latest_attacker} attacker wins, {latest_defender} defender wins")
    
    # Best network updates
    if history['win_rates']:
        updates = sum(1 for wr in history['win_rates'] if wr >= 0.55)
        print(f"\nBest network updates: {updates}/{len(history['win_rates'])} evaluations")
    
    print("\n" + "=" * 70)
    
    # Show ELO progression in detail
    if history['vs_random_elo'] and len(history['vs_random_elo']) > 1:
        print("\nELO Rating Progression:")
        print("  Iter | ELO  | Win% | Attacker | Defender")
        print("  " + "-" * 44)
        
        # Show first, middle, and last few
        indices = []
        if len(history['vs_random_elo']) <= 10:
            indices = list(range(len(history['vs_random_elo'])))
        else:
            indices.extend([0, 1, 2])  # First 3
            mid = len(history['vs_random_elo']) // 2
            indices.extend([mid - 1, mid, mid + 1])  # Middle 3
            indices.extend(range(len(history['vs_random_elo']) - 3, len(history['vs_random_elo'])))  # Last 3
        
        prev_idx = -10
        for i in indices:
            if i - prev_idx > 1:
                print("  ...")
            
            # Find corresponding iteration
            iter_num = (i + 1)
            elo = history['vs_random_elo'][i]
            win_rate = history['vs_random_win_rate'][i]
            att_wins = history['vs_random_attacker_wins'][i] if history['vs_random_attacker_wins'] else 0
            def_wins = history['vs_random_defender_wins'][i] if history['vs_random_defender_wins'] else 0
            
            print(f"  {iter_num:4d} | {elo:+4.0f} | {100*win_rate:4.0f}% | {att_wins:8d} | {def_wins:8d}")
            prev_idx = i
        
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        history_file = sys.argv[1]
    else:
        history_file = "checkpoints/training_history.json"
    
    print_training_summary(history_file)
