#!/usr/bin/env python3
"""
Benchmark neural network inference speed for different configurations.

Tests the impact of channel width and torch.compile on inference performance.

Usage:
    python benchmark_network.py [--game {brandubh,tablut,hnefatafl}] [--batch-size N] [--iterations N]

Example:
    python benchmark_network.py --game brandubh
    python benchmark_network.py --game tablut --batch-size 256 --iterations 1000
"""

import argparse
import time
import torch
import numpy as np
from network import BrandubhNet
from network_tablut import TablutNet
from network_hnefatafl import HnefataflNet


def count_parameters(network):
    """Count the total number of trainable parameters in a network."""
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def benchmark_network(network_class, board_size, num_channels, num_res_blocks, batch_size, num_iterations, use_compile=False, device='cpu'):
    """
    Benchmark a network configuration.
    
    Args:
        network_class: Network class to instantiate
        board_size: Size of the board
        num_channels: Number of channels in the network
        num_res_blocks: Number of residual blocks (depth)
        batch_size: Batch size for inference
        num_iterations: Number of forward passes to average
        use_compile: Whether to use torch.compile
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        dict with timing statistics and parameter count
    """
    # Create network
    network = network_class(num_res_blocks=num_res_blocks, num_channels=num_channels).to(device)
    network.eval()
    
    # Count parameters
    num_params = count_parameters(network)
    
    # Compile if requested
    compile_time = 0.0
    if use_compile:
        if hasattr(torch, 'compile'):
            print("    Compiling network... ", end='', flush=True)
            compile_start = time.time()
            network = torch.compile(network)
            compile_time = time.time() - compile_start
            print(f"done ({compile_time:.2f}s)")
        else:
            print("    Warning: torch.compile not available (requires PyTorch 2.0+), skipping compilation")
            return None  # Signal that this configuration should be skipped
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 4, board_size, board_size, device=device)
    
    # Warmup runs (important for compiled networks)
    print(f"    Warming up... ", end='', flush=True)
    warmup_iterations = 10 if use_compile else 3
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = network(dummy_input)
    print("done")
    
    # Benchmark runs
    print(f"    Benchmarking {num_iterations} iterations... ", end='', flush=True)
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            policy_logits, value = network(dummy_input)
            # Ensure computation is complete (important for GPU)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)
    
    print("done")
    
    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    
    # Calculate throughput
    throughput = batch_size / mean_time  # positions/second
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'median_time': median_time,
        'throughput': throughput,
        'compile_time': compile_time,
        'num_params': num_params,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark neural network inference speed")
    parser.add_argument("--game", type=str, default="brandubh", 
                       choices=["brandubh", "tablut", "hnefatafl"],
                       help="Game variant (default: brandubh)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for inference (default: 1 for single position gameplay)")
    parser.add_argument("--iterations", type=int, default=500,
                       help="Number of forward passes to benchmark (default: 500)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run on (default: cpu)")
    parser.add_argument("--no-flush-denormal", action="store_true",
                       help="Disable denormal flushing (for comparison - will be slower!)")
    
    args = parser.parse_args()
    
    # Enable denormal flushing by default for fast inference
    # Denormal (subnormal) floating point numbers are 10-100x slower on CPU
    if not args.no_flush_denormal:
        torch.set_flush_denormal(True)
        print("Denormal flushing: ENABLED (fast mode)")
    else:
        torch.set_flush_denormal(False)
        print("Denormal flushing: DISABLED (slow mode - for comparison)")
    
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
    
    # Test configurations
    channel_sizes = [64, 96, 128]
    depth_options = [4, 6]  # Number of residual blocks
    compile_options = [False, True]
    
    print("=" * 100)
    print(f"Neural Network Inference Benchmark - {args.game.capitalize()} ({board_size}x{board_size})")
    print("=" * 100)
    print("Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size} {'(single position - realistic for gameplay)' if args.batch_size == 1 else '(batch processing - for training)'}")
    print(f"  Benchmark iterations: {args.iterations}")
    print("  Channel widths: 64, 96, 128")
    print("  Network depths: 4, 6 residual blocks")
    
    # Check torch.compile availability
    if not hasattr(torch, 'compile'):
        print("\nNote: torch.compile not available (requires PyTorch 2.0+)")
        print("      Only testing without compilation.")
    
    print("=" * 80)
    
    results = []
    
    for num_channels in channel_sizes:
        for num_res_blocks in depth_options:
            for use_compile in compile_options:
                compile_str = "compiled" if use_compile else "no compile"
                print(f"\n[{num_channels} channels, {num_res_blocks} blocks, {compile_str}]")
                
                stats = benchmark_network(
                    network_class=network_class,
                    board_size=board_size,
                    num_channels=num_channels,
                    num_res_blocks=num_res_blocks,
                    batch_size=args.batch_size,
                    num_iterations=args.iterations,
                    use_compile=use_compile,
                    device=args.device
                )
                
                # Skip if compilation was requested but not available
                if stats is None:
                    continue
                
                results.append({
                    'channels': num_channels,
                    'blocks': num_res_blocks,
                    'compiled': use_compile,
                    'stats': stats
                })
                
                print("  Results:")
                print(f"    Parameters:    {stats['num_params']:,}")
                print(f"    Mean time:     {stats['mean_time']*1000:.3f} ms  (±{stats['std_time']*1000:.3f} ms)")
                print(f"    Median time:   {stats['median_time']*1000:.3f} ms")
                print(f"    Min time:      {stats['min_time']*1000:.3f} ms")
                print(f"    Max time:      {stats['max_time']*1000:.3f} ms")
                if args.batch_size > 1:
                    print(f"    Throughput:    {stats['throughput']:.1f} positions/sec")
                else:
                    print(f"    Inference rate: {stats['throughput']:.1f} evaluations/sec")
                if use_compile:
                    print(f"    Compile time:  {stats['compile_time']:.2f} s")
    
    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    throughput_label = "Throughput (pos/s)" if args.batch_size > 1 else "Evals/sec"
    print(f"{'Channels':<10} {'Blocks':<8} {'Parameters':<12} {'Compiled':<12} {'Mean (ms)':<12} {'Speedup':<10} {throughput_label:<20}")
    print("-" * 100)
    
    # Use 64 channels, 4 blocks without compilation as baseline
    baseline = None
    for r in results:
        if r['channels'] == 64 and r['blocks'] == 4 and not r['compiled']:
            baseline = r['stats']['mean_time']
            break
    
    for r in results:
        channels = r['channels']
        blocks = r['blocks']
        num_params = r['stats']['num_params']
        compiled = "Yes" if r['compiled'] else "No"
        mean_ms = r['stats']['mean_time'] * 1000
        throughput = r['stats']['throughput']
        
        if baseline:
            speedup = baseline / r['stats']['mean_time']
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "—"
        
        print(f"{channels:<10} {blocks:<8} {num_params:<12,} {compiled:<12} {mean_ms:<12.3f} {speedup_str:<10} {throughput:<20.1f}")
    
    print("=" * 100)
    
    # Key findings
    print("\nKey Findings:")
    
    # Compare compilation speedup for each channel/depth combination
    for num_channels in channel_sizes:
        for num_res_blocks in depth_options:
            no_compile = None
            with_compile = None
            
            for r in results:
                if r['channels'] == num_channels and r['blocks'] == num_res_blocks:
                    if r['compiled']:
                        with_compile = r['stats']['mean_time']
                    else:
                        no_compile = r['stats']['mean_time']
            
            if no_compile and with_compile:
                speedup = no_compile / with_compile
                print(f"  • {num_channels} ch, {num_res_blocks} blocks: torch.compile provides {speedup:.2f}x speedup")
    
    # Compare depth impact for each channel size (without compilation)
    print()
    for num_channels in channel_sizes:
        depth4 = None
        depth6 = None
        for r in results:
            if r['channels'] == num_channels and not r['compiled']:
                if r['blocks'] == 4:
                    depth4 = r['stats']['mean_time']
                elif r['blocks'] == 6:
                    depth6 = r['stats']['mean_time']
        
        if depth4 and depth6:
            slowdown = depth6 / depth4
            print(f"  • {num_channels} channels: 6 blocks vs 4 blocks = {slowdown:.2f}x slower")
    
    # Compare channel width impact for each depth (without compilation)
    print()
    for num_res_blocks in depth_options:
        ch64 = None
        ch128 = None
        for r in results:
            if r['blocks'] == num_res_blocks and not r['compiled']:
                if r['channels'] == 64:
                    ch64 = r['stats']['mean_time']
                elif r['channels'] == 128:
                    ch128 = r['stats']['mean_time']
        
        if ch64 and ch128:
            slowdown = ch128 / ch64
            print(f"  • {num_res_blocks} blocks: 128 ch vs 64 ch = {slowdown:.2f}x slower")
    
    print()


if __name__ == "__main__":
    main()
