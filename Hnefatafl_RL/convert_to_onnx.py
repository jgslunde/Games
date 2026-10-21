#!/usr/bin/env python3
"""
Convert a training checkpoint to ONNX for deployment in the browser (onnxruntime-web) or elsewhere.

The network architecture and its input planes are read from the checkpoint. Next to the .onnx file
a .json file is written that says exactly what the model expects and returns: board size, the input
planes in order, the policy index layout and the complete rule set the network was trained with
(see tafl.Rules). A browser implementation must build the same planes in the same order (see
features.py) and play by the same rules.

Usage:
    python convert_to_onnx.py <checkpoint_path> [--output OUTPUT] [--opset N] [--no-verify]

Examples:
    python convert_to_onnx.py checkpoints/brandubh/line_4x32/best_model.pth
    python convert_to_onnx.py checkpoints/copenhagen/line_10x64_gpool/best_model.pth --output model.onnx
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from tafl import ENDINGS, policy_size
from tafl_network import load_network, describe_network, count_parameters

# The network's outputs in order (see tafl_network.py); clients can read them by name
OUTPUT_NAMES = ['policy_logits', 'value', 'ending_logits', 'moves_left']


def export_to_onnx(network, output_path, board_size, num_planes, opset_version=12):
    """
    Export the network; the batch axis is dynamic, the input is [batch, num_planes, N, N].

    Weights and BatchNorm statistics below the smallest normal float32 (about 1.2e-38) are set to 0
    first (in place). Such "subnormal" numbers make CPUs 10-100x slower unless the program flushes
    them to zero, which training does but WebAssembly in the browser cannot. Zeroing them changes
    the outputs by far less than float32 precision.
    """
    print(f"\nExporting to ONNX format...")
    tiny = torch.finfo(torch.float32).tiny
    flushed = 0
    with torch.no_grad():
        for tensor in list(network.parameters()) + list(network.buffers()):
            if tensor.is_floating_point():
                subnormal = (tensor != 0) & (tensor.abs() < tiny)
                flushed += int(subnormal.sum())
                tensor[subnormal] = 0.0
    print(f"  Subnormal weights set to 0: {flushed}")
    dummy_input = torch.randn(1, num_planes, board_size, board_size)
    torch.onnx.export(
        network,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['board_state'],
        output_names=OUTPUT_NAMES,
        dynamic_axes={name: {0: 'batch_size'} for name in ['board_state'] + OUTPUT_NAMES},
        dynamo=False,
        verbose=False
    )
    print(f"  ✓ ONNX model saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


def write_model_info(network, checkpoint, onnx_path):
    """Write <model>.json: everything a client needs to build the input and read the output."""
    info_path = os.path.splitext(onnx_path)[0] + '.json'
    config, features, board_size = checkpoint['config'], network.feature_spec, network.board_size
    info = {
        'board_size': board_size,
        'input': {
            'shape': [1, features.num_planes, board_size, board_size],
            'planes': features.plane_names(),
            'description': 'See features.py: piece planes of the last history_length positions (newest '
                           'first, zero planes before the start of the game), the player plane (0 '
                           'attackers, 1 defenders), then the optional constant, reach and defender-region '
                           'planes. All values are 0 or 1.',
        },
        'policy': {
            'size': policy_size(board_size),
            'index': 'from_square * 4 * (N-1) + direction * (N-1) + (distance - 1), with '
                     'from_square = row * N + col and directions 0 up, 1 down, 2 left, 3 right',
            'note': 'logits; take the softmax over the legal moves only',
        },
        'value': 'tanh output in [-1, 1] for the player to move',
        'ending': {'classes': list(ENDINGS),
                   'note': 'logits of how the game will end; take the softmax over the endings the '
                           'rules allow (exit_fort and encircled only where those rules are on)'},
        'moves_left': 'the expected number of plies until the game ends',
        'network': network.config.to_dict(),
        'network_description': describe_network(network),
        'num_parameters': count_parameters(network),
        # The complete rule set (see tafl.Rules; the setup rows use A attacker, D defender, K king)
        'ruleset': config['ruleset'],
        'rules': config['rules'],
        'max_game_length': config['max_game_length'],
        # The search settings of the training run's evaluation games. An unvisited move is valued
        # at the mean value of its parent node, seen by the player to move there, plus
        # fpu_reduction (negative = pessimistic). During backup the value is multiplied by
        # -discount at every ply, as the training targets were.
        'search': {'c_puct': config['eval_c_puct'],
                   'fpu_reduction': config['eval_fpu_reduction'],
                   'discount': config['search_discount']},
        'run_name': config['name'],
        'iteration': checkpoint.get('iteration'),
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  ✓ Model description saved to: {info_path}")


def verify_onnx_model(onnx_path, network, board_size, num_planes):
    """Check that the ONNX model gives the same outputs as the PyTorch network."""
    print(f"\nVerifying ONNX model...")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠ Warning: onnxruntime not installed, skipping verification")
        print("  Install with: pip install onnxruntime")
        return None

    test_input = (torch.rand(1, num_planes, board_size, board_size) < 0.2).float()
    network.eval()
    with torch.no_grad():
        pytorch_outputs = [output.numpy() for output in network(test_input)]

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    onnx_outputs = ort_session.run(OUTPUT_NAMES, {input_name: test_input.numpy()})

    # Relative to the size of each output (moves_left is in plies, the others are of order 1)
    tolerance = 1e-4
    diffs = [np.abs(a - b).max() / max(1.0, np.abs(a).max())
             for a, b in zip(pytorch_outputs, onnx_outputs)]
    for name, diff in zip(OUTPUT_NAMES, diffs):
        print(f"  {name} difference: {diff:.2e} (tolerance: {tolerance:.2e})")
    if max(diffs) < tolerance:
        print("  ✓ Verification passed! ONNX model outputs match PyTorch model")
        return True
    print("  ✗ Verification failed! Outputs differ beyond tolerance")
    return False


def print_model_info(onnx_path):
    """Print the ONNX model's opset, inputs and outputs."""
    try:
        import onnx
    except ImportError:
        print("\n  (Install 'onnx' package to see detailed model info: pip install onnx)")
        return
    model = onnx.load(onnx_path)
    print(f"\nONNX Model Information:")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Operators: {len(model.graph.node)}")
    for label, tensors in (("Input", model.graph.input), ("Output", model.graph.output)):
        print(f"  {label}:")
        for tensor in tensors:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic'
                     for dim in tensor.type.tensor_type.shape.dim]
            print(f"    - {tensor.name}: {shape}")


def main():
    parser = argparse.ArgumentParser(description='Convert a PyTorch checkpoint to ONNX',
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file (.pth)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path (default: the checkpoint path with .onnx)')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version (default: 12)')
    parser.add_argument('--no-verify', action='store_true', help='Skip the verification step')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found")
        sys.exit(1)
    if args.output is None:
        args.output = os.path.splitext(args.checkpoint)[0] + '.onnx'

    print("=" * 70)
    print("PyTorch to ONNX Converter")
    print("=" * 70)
    print(f"Loading checkpoint from {args.checkpoint}...")
    network, checkpoint = load_network(args.checkpoint)
    board_size, num_planes = network.board_size, network.feature_spec.num_planes
    print(f"  Network: {describe_network(network)}")
    print(f"  Rule set: {checkpoint['config']['ruleset']}; input planes: "
          f"{', '.join(network.feature_spec.plane_names())}")
    export_to_onnx(network, args.output, board_size, num_planes, args.opset)
    write_model_info(network, checkpoint, args.output)
    if not args.no_verify:
        verify_onnx_model(args.output, network, board_size, num_planes)
    print_model_info(args.output)
    print("\n" + "=" * 70)
    print("Conversion completed successfully!")
    print("=" * 70)
    print(f"\nONNX file: {args.output}")
    print(f"Model description: {os.path.splitext(args.output)[0] + '.json'}")


if __name__ == "__main__":
    main()
