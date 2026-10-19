#!/usr/bin/env python3
"""
Time one network call (single thread) for the networks of run configs, as the self-play workers
make it: batch 1, compiled kernels called from generated C++. This is the cost per search
simulation that a network adds.

Usage:
    python benchmark_network.py <run config.json> [<run config.json> ...] [--batch-size N]
                                [--iterations N] [--eager]

Example:
    python benchmark_network.py configs/copenhagen/*.json
"""

import argparse
import time
import warnings

import numpy as np
import torch

from run_config import load_config
from tafl_network import TaflNet, count_parameters


def main():
    parser = argparse.ArgumentParser(description="Time the networks of run configs")
    parser.add_argument("configs", nargs="+", help="Run config JSON files")
    parser.add_argument("--batch-size", type=int, default=1, help="Positions per call (default: 1)")
    parser.add_argument("--iterations", type=int, default=300, help="Timed calls (default: 300)")
    parser.add_argument("--eager", action="store_true", help="Time without torch.compile")
    args = parser.parse_args()

    # As in training: subnormal weights flushed to zero, one thread, C++ wrapper for compiled kernels
    torch.set_flush_denormal(True)
    warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')
    torch.set_num_threads(1)
    from torch._inductor import config as inductor_config
    inductor_config.cpp_wrapper = True

    for path in args.configs:
        config = load_config(path)
        n = config.board_size
        network = TaflNet(config.network, n).eval()
        model = network if args.eager else torch.compile(network, mode='default')
        x = (torch.rand(args.batch_size, config.network.features.num_planes, n, n) < 0.2).float()
        with torch.no_grad():
            for _ in range(20):  # warm-up (and compilation)
                model(x)
            times = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                model(x)
                times.append(time.perf_counter() - start)
        print(f"{path}: {config.network.describe()}, {count_parameters(network):,} parameters, "
              f"{n}x{n}, batch {args.batch_size}: median {1000 * np.median(times):.2f} ms", flush=True)


if __name__ == "__main__":
    main()
