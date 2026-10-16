#!/usr/bin/env python3
"""
Convert a training checkpoint to ONNX for deployment in the browser (onnxruntime-web) or elsewhere.

The network architecture and its input planes are read from the checkpoint, so this works for the
configurable networks (tafl_network.py) as well as the older fixed classes. Next to the .onnx file a
.json file is written that says exactly what the model expects and returns: board size, the input
planes in order, the policy index layout and the rules the network was trained with. A browser
implementation must build the same planes in the same order (see features.py).

Usage:
    python convert_to_onnx.py <checkpoint_path> [--output OUTPUT] [--board-size N] [--opset N] [--no-verify]

Examples:
    python convert_to_onnx.py checkpoints_brandubh/line_4x32/best_model.pth
    python convert_to_onnx.py checkpoints_tablut/best_model.pth --output model.onnx
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from tafl_network import load_network, network_config_from_checkpoint, describe_network, count_parameters


def load_checkpoint(checkpoint_path, board_size=None):
    """The network of a checkpoint (eval mode), its NetworkConfig, the checkpoint dict and board size."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    network, checkpoint = load_network(checkpoint_path, board_size=board_size)
    board_size = network.board_size if hasattr(network, 'board_size') else board_size
    if board_size is None:
        board_size = checkpoint.get('config', {}).get('board_size')
    net_config = network_config_from_checkpoint(checkpoint, board_size)
    print(f"  Network: {describe_network(network)}")
    print(f"  Board: {board_size}x{board_size}; input planes: {', '.join(net_config.features.plane_names())}")
    return network, net_config, checkpoint, board_size


def export_to_onnx(network, output_path, board_size, num_planes, opset_version=12):
    """Export the network; the batch axis is dynamic, the input is [batch, num_planes, N, N]."""
    print(f"\nExporting to ONNX format...")
    dummy_input = torch.randn(1, num_planes, board_size, board_size)
    torch.onnx.export(
        network,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['board_state'],
        output_names=['policy_logits', 'value'],
        dynamic_axes={
            'board_state': {0: 'batch_size'},
            'policy_logits': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        },
        dynamo=False,
        verbose=False
    )
    print(f"  ✓ ONNX model saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


def write_model_info(network, net_config, checkpoint, board_size, onnx_path):
    """Write <model>.json: everything a client needs to build the input and read the output."""
    info_path = os.path.splitext(onnx_path)[0] + '.json'
    config = checkpoint.get('config', {}) if isinstance(checkpoint.get('config'), dict) else {}
    rule_names = ['king_capture_pieces', 'king_can_capture', 'throne_is_hostile', 'throne_enabled']
    info = {
        'board_size': board_size,
        'input': {
            'shape': [1, net_config.features.num_planes, board_size, board_size],
            'planes': net_config.features.plane_names(),
            'description': 'See features.py: piece planes of the last history_length positions (newest '
                           'first, zero planes before the start of the game), the player plane (0 '
                           'attackers, 1 defenders), then the optional constant, reach and king-region '
                           'planes. All values are 0 or 1.',
        },
        'policy': {
            'size': board_size * board_size * 4 * (board_size - 1),
            'index': 'from_square * 4 * (N-1) + direction * (N-1) + (distance - 1), with '
                     'from_square = row * N + col and directions 0 up, 1 down, 2 left, 3 right',
            'note': 'logits; take the softmax over the legal moves only',
        },
        'value': 'tanh output in [-1, 1] for the player to move',
        'network': net_config.to_dict(),
        'network_description': describe_network(network),
        'num_parameters': count_parameters(network),
        'rules': {name: config[name] for name in rule_names if name in config},
        'search_discount': config.get('search_discount'),
        'run_name': config.get('name'),
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
        pytorch_policy, pytorch_value = network(test_input)
        pytorch_policy = pytorch_policy.numpy()
        pytorch_value = pytorch_value.numpy()

    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_policy, onnx_value = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input.numpy()})

    policy_diff = np.abs(pytorch_policy - onnx_policy).max()
    value_diff = np.abs(pytorch_value - onnx_value).max()
    tolerance = 1e-4
    print(f"  Policy output difference: {policy_diff:.2e} (tolerance: {tolerance:.2e})")
    print(f"  Value output difference: {value_diff:.2e} (tolerance: {tolerance:.2e})")
    if policy_diff < tolerance and value_diff < tolerance:
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
    parser.add_argument('--board-size', type=int, default=None, choices=[7, 9, 11],
                        help='Board size, only needed for very old checkpoints that do not record it')
    parser.add_argument('--game', type=str, choices=['brandubh', 'tablut', 'hnefatafl'], default=None,
                        help='Alternative to --board-size')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version (default: 12)')
    parser.add_argument('--no-verify', action='store_true', help='Skip the verification step')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found")
        sys.exit(1)
    board_size = args.board_size
    if board_size is None and args.game is not None:
        board_size = {'brandubh': 7, 'tablut': 9, 'hnefatafl': 11}[args.game]
    if args.output is None:
        args.output = os.path.splitext(args.checkpoint)[0] + '.onnx'

    print("=" * 70)
    print("PyTorch to ONNX Converter")
    print("=" * 70)
    network, net_config, checkpoint, board_size = load_checkpoint(args.checkpoint, board_size)
    num_planes = net_config.features.num_planes
    export_to_onnx(network, args.output, board_size, num_planes, args.opset)
    write_model_info(network, net_config, checkpoint, board_size, args.output)
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
