#!/usr/bin/env python3
"""
Start a training run from a JSON config file.

Usage:
    python run_training.py configs/brandubh/baseline.json
    python run_training.py configs/tablut/line_6x48_gpool.json --set system.num_workers=16
    python run_training.py configs/brandubh/baseline.json --set self_play.num_iterations=50 \
        --set network.channels=48 --checkpoint-dir checkpoints_brandubh/test
    python run_training.py configs/brandubh/baseline.json --resume checkpoints/.../checkpoint_iter_40.pth
    python run_training.py configs/brandubh/baseline.json --show      # print the resolved config only

The config file format is described in run_config.py; the network section in network_config.py.
The resolved config (file + overrides) is printed at the start and written to
<checkpoint_dir>/run_config.json, and every checkpoint also stores it.
"""

import argparse
import sys

from run_config import apply_overrides, describe, load_config


def main():
    parser = argparse.ArgumentParser(description="Train a Tafl network from a JSON config",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("config", help="Path of the run config (JSON)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="Override one setting, e.g. system.num_workers=16, network.channels=48, "
                             "network.features.history_length=4, self_play.num_iterations=5. "
                             "Values are read as JSON where possible. Repeatable.")
    parser.add_argument("--checkpoint-dir", help="Shorthand for --set system.checkpoint_dir=...")
    parser.add_argument("--num-workers", type=int, help="Shorthand for --set system.num_workers=...")
    parser.add_argument("--resume", metavar="CHECKPOINT",
                        help="Resume from this checkpoint (weights, optimizer state and training history)")
    parser.add_argument("--load-weights", metavar="CHECKPOINT",
                        help="Start a new run from the weights of this checkpoint")
    parser.add_argument("--show", action="store_true", help="Print the resolved config and exit")
    args = parser.parse_args()

    if args.resume and args.load_weights:
        parser.error("use either --resume or --load-weights, not both")

    overrides = list(args.set)
    if args.checkpoint_dir:
        overrides.append(f"system.checkpoint_dir={args.checkpoint_dir}")
    if args.num_workers is not None:
        overrides.append(f"system.num_workers={args.num_workers}")

    config = load_config(args.config)
    if overrides:
        config = apply_overrides(config, overrides)

    print(f"Config file: {args.config}")
    if overrides:
        print(f"Overrides:   {' '.join(overrides)}")
    print("Resolved configuration:")
    print(describe(config))
    if args.show:
        return

    from train import train  # needs torch; imported here so that --show works without it
    train(config, resume_from=args.resume, load_weights_from=args.load_weights)


if __name__ == "__main__":
    sys.exit(main())
