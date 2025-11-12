#!/usr/bin/env python3
"""
Benchmark inference speed across all checkpoints in a directory.

This script helps identify if/when trained weights cause slower inference
compared to randomly initialized networks.

Usage:
    python benchmark_checkpoints.py <checkpoint_dir> [--game {brandubh,tablut,hnefatafl}] [--iterations N]

Example:
    python benchmark_checkpoints.py checkpoints_brandubh/fatman_nothrone/
    python benchmark_checkpoints.py checkpoints_tablut/jack_nothrone/ --game tablut --iterations 20
"""

import argparse
import time
import torch
import numpy as np
import os
import glob
from network import BrandubhNet
from network_tablut import TablutNet
from network_hnefatafl import HnefataflNet


def count_parameters(network):
    """Count the total number of trainable parameters in a network."""
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def benchmark_checkpoint(checkpoint_path, network_class, board_size, num_iterations=20, device='cpu'):
    """
    Benchmark a single checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        network_class: Network class to instantiate
        board_size: Size of the board
        num_iterations: Number of forward passes to average
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        dict with timing statistics
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'network_state_dict' in checkpoint:
            state_dict = checkpoint['network_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Maybe the checkpoint is just the state dict itself
            state_dict = checkpoint
        
        # Get network configuration from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            num_channels = config.get('num_channels', 64)
            num_res_blocks = config.get('num_res_blocks', 4)
        else:
            num_channels = checkpoint.get('num_channels', 64)
            num_res_blocks = checkpoint.get('num_res_blocks', 4)
        
        # Create network
        network = network_class(num_res_blocks=num_res_blocks, num_channels=num_channels).to(device)
        network.load_state_dict(state_dict)
        network.eval()
        
        # Count parameters
        num_params = count_parameters(network)
        
        # Create dummy input (batch size 1)
        dummy_input = torch.randn(1, 4, board_size, board_size, device=device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(3):
                _ = network(dummy_input)
        
        # Benchmark runs
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                policy_logits, value = network(dummy_input)
                # Ensure computation is complete
                if device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                times.append(elapsed)
        
        # Calculate statistics
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        median_time = np.median(times)
        
        # Get training metadata if available
        iteration = checkpoint.get('iteration', None)
        elo = checkpoint.get('elo', None)
        
        return {
            'success': True,
            'mean_time': mean_time,
            'std_time': std_time,
            'median_time': median_time,
            'num_params': num_params,
            'num_channels': num_channels,
            'num_res_blocks': num_res_blocks,
            'iteration': iteration,
            'elo': elo,
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed across checkpoints")
    parser.add_argument("checkpoint_dir", type=str,
                       help="Directory containing checkpoint files")
    parser.add_argument("--game", type=str, default="brandubh",
                       choices=["brandubh", "tablut", "hnefatafl"],
                       help="Game variant (default: brandubh)")
    parser.add_argument("--iterations", type=int, default=20,
                       help="Number of forward passes per checkpoint (default: 20)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run on (default: cpu)")
    parser.add_argument("--pattern", type=str, default="*.pth",
                       help="Checkpoint file pattern (default: *.pth)")
    parser.add_argument("--no-flush-denormal", action="store_true",
                       help="Disable denormal flushing (to reproduce the slowdown issue)")
    
    args = parser.parse_args()
    
    # Enable denormal flushing by default for fast inference
    # This eliminates the 2-3x slowdown caused by denormal numbers
    if not args.no_flush_denormal:
        torch.set_flush_denormal(True)
        print("Denormal flushing: ENABLED (eliminates slowdown)")
    else:
        torch.set_flush_denormal(False)
        print("Denormal flushing: DISABLED (will reproduce 2-3x slowdown with trained weights)")
    
    args = parser.parse_args()
    
    # Select network class and board size
    if args.game == "tablut":
        network_class = TablutNet
        board_size = 9
    elif args.game == "hnefatafl":
        network_class = HnefataflNet
        board_size = 11
    else:  # brandubh
        network_class = BrandubhNet
        board_size = 7
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(args.checkpoint_dir, args.pattern)
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))
    
    if not checkpoint_files:
        print(f"Error: No checkpoint files found matching pattern: {checkpoint_pattern}")
        return
    
    print("=" * 100)
    print(f"Checkpoint Inference Speed Benchmark - {args.game.capitalize()} ({board_size}x{board_size})")
    print("=" * 100)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Number of checkpoints: {len(checkpoint_files)}")
    print(f"Iterations per checkpoint: {args.iterations}")
    print(f"Device: {args.device}")
    print("=" * 100)
    print()
    
    results = []
    
    # Benchmark random initialization for comparison
    print("Benchmarking random initialization (baseline)...")
    random_net = network_class(num_res_blocks=4, num_channels=64).to(args.device)
    random_net.eval()
    dummy_input = torch.randn(1, 4, board_size, board_size, device=args.device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = random_net(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(args.iterations):
            start = time.time()
            _ = random_net(dummy_input)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)
    
    random_mean = np.mean(times)
    random_std = np.std(times)
    print(f"Random weights: {random_mean*1000:.3f} ms (±{random_std*1000:.3f} ms)\n")
    
    # Benchmark each checkpoint
    for i, checkpoint_path in enumerate(checkpoint_files):
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"[{i+1}/{len(checkpoint_files)}] {checkpoint_name}... ", end='', flush=True)
        
        result = benchmark_checkpoint(
            checkpoint_path=checkpoint_path,
            network_class=network_class,
            board_size=board_size,
            num_iterations=args.iterations,
            device=args.device
        )
        
        if result['success']:
            print(f"{result['mean_time']*1000:.3f} ms (±{result['std_time']*1000:.3f} ms) - {result['num_channels']}ch/{result['num_res_blocks']}blocks")
            results.append({
                'checkpoint': checkpoint_name,
                'path': checkpoint_path,
                **result
            })
        else:
            print(f"FAILED: {result['error']}")
    
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    if not results:
        print("No successful benchmark results.")
        return
    
    # Sort by mean time
    results_sorted = sorted(results, key=lambda x: x['mean_time'])
    
    print(f"\n{'Checkpoint':<40} {'Iteration':<12} {'Channels':<10} {'Blocks':<8} {'Mean (ms)':<12} {'vs Random':<12}")
    print("-" * 100)
    
    for r in results_sorted[:10]:  # Show 10 fastest
        checkpoint = r['checkpoint'][:38]
        iteration = str(r['iteration']) if r['iteration'] is not None else 'N/A'
        mean_ms = r['mean_time'] * 1000
        vs_random = r['mean_time'] / random_mean
        
        print(f"{checkpoint:<40} {iteration:<12} {r['num_channels']:<10} {r['num_res_blocks']:<8} {mean_ms:<12.3f} {vs_random:.2f}x")
    
    print("\n... (showing 10 fastest)")
    print()
    
    # Show slowest
    print(f"\n{'Checkpoint':<40} {'Iteration':<12} {'Channels':<10} {'Blocks':<8} {'Mean (ms)':<12} {'vs Random':<12}")
    print("-" * 100)
    
    for r in results_sorted[-10:]:  # Show 10 slowest
        checkpoint = r['checkpoint'][:38]
        iteration = str(r['iteration']) if r['iteration'] is not None else 'N/A'
        mean_ms = r['mean_time'] * 1000
        vs_random = r['mean_time'] / random_mean
        
        print(f"{checkpoint:<40} {iteration:<12} {r['num_channels']:<10} {r['num_res_blocks']:<8} {mean_ms:<12.3f} {vs_random:.2f}x")
    
    print("\n... (showing 10 slowest)")
    print()
    
    # Statistics
    all_times = [r['mean_time'] for r in results]
    mean_all = np.mean(all_times)
    std_all = np.std(all_times)
    min_time = np.min(all_times)
    max_time = np.max(all_times)
    
    print("=" * 100)
    print("STATISTICS")
    print("=" * 100)
    print(f"Random initialization:  {random_mean*1000:.3f} ms")
    print(f"Checkpoint mean:        {mean_all*1000:.3f} ms (±{std_all*1000:.3f} ms)")
    print(f"Checkpoint min:         {min_time*1000:.3f} ms ({min_time/random_mean:.2f}x vs random)")
    print(f"Checkpoint max:         {max_time*1000:.3f} ms ({max_time/random_mean:.2f}x vs random)")
    print(f"Slowdown range:         {min_time/random_mean:.2f}x to {max_time/random_mean:.2f}x")
    
    # Find checkpoints that are significantly slower
    threshold = 2.0  # 2x slower than random
    slow_checkpoints = [r for r in results if r['mean_time'] / random_mean > threshold]
    
    if slow_checkpoints:
        print(f"\nCheckpoints >2x slower than random: {len(slow_checkpoints)}/{len(results)}")
        print(f"This suggests the issue affects {100*len(slow_checkpoints)/len(results):.1f}% of checkpoints")
    else:
        print(f"\nNo checkpoints found that are significantly slower (>2x) than random initialization")
    
    print("=" * 100)
    
    # Save detailed results to CSV
    csv_path = os.path.join(args.checkpoint_dir, "inference_benchmark_results.csv")
    print(f"\nSaving detailed results to: {csv_path}")
    
    with open(csv_path, 'w') as f:
        f.write("checkpoint,iteration,channels,blocks,parameters,mean_ms,std_ms,median_ms,vs_random\n")
        for r in results:
            iteration = r['iteration'] if r['iteration'] is not None else ''
            vs_random = r['mean_time'] / random_mean
            f.write(f"{r['checkpoint']},{iteration},{r['num_channels']},{r['num_res_blocks']},{r['num_params']},{r['mean_time']*1000:.6f},{r['std_time']*1000:.6f},{r['median_time']*1000:.6f},{vs_random:.4f}\n")
    
    print("Done!")


if __name__ == "__main__":
    main()
