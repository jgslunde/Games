#!/usr/bin/env python3
"""
Convert PyTorch checkpoint to ONNX format for deployment in JavaScript/web environments.

This script loads a trained model checkpoint and exports it to ONNX format,
which can be used with ONNX Runtime in JavaScript, C++, or other environments.

Usage:
    python convert_to_onnx.py <checkpoint_path> [--output OUTPUT] [--game {brandubh,tablut,hnefatafl}]

Examples:
    python convert_to_onnx.py checkpoints_brandubh/jack_nothrone/best_model.pth
    python convert_to_onnx.py checkpoints_tablut/best_model.pth --game tablut --output model.onnx
    python convert_to_onnx.py checkpoints_hnefatafl/best_model.pth --game hnefatafl
"""

import argparse
import os
import sys
import torch
import numpy as np

# Import network architectures
from network import BrandubhNet
from network_tablut import TablutNet
from network_hnefatafl import HnefataflNet


def load_checkpoint(checkpoint_path, game_type='brandubh'):
    """
    Load a checkpoint and create the appropriate network.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        game_type: Type of game ('brandubh', 'tablut', or 'hnefatafl')
    
    Returns:
        network: The loaded neural network
        config: Configuration dictionary from checkpoint (if available)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract configuration
    config = checkpoint.get('config', {})
    num_res_blocks = config.get('num_res_blocks', 4)
    num_channels = config.get('num_channels', 64)
    
    print(f"  Architecture: {num_res_blocks} residual blocks, {num_channels} channels")
    
    # Create appropriate network
    if game_type == 'tablut':
        network = TablutNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
        board_size = 9
    elif game_type == 'hnefatafl':
        network = HnefataflNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
        board_size = 11
    else:  # brandubh
        network = BrandubhNet(num_res_blocks=num_res_blocks, num_channels=num_channels)
        board_size = 7
    
    # Load state dict
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    
    print(f"  Game type: {game_type} ({board_size}x{board_size})")
    print(f"  Network loaded successfully")
    
    return network, config, board_size


def export_to_onnx(network, output_path, board_size, opset_version=12):
    """
    Export network to ONNX format.
    
    Args:
        network: PyTorch network to export
        output_path: Path to save ONNX file
        board_size: Size of the game board (7, 9, or 11)
        opset_version: ONNX opset version (default: 12 for broad compatibility)
    """
    print(f"\nExporting to ONNX format...")
    
    # Create dummy input for tracing
    # Input format: [batch_size, channels, height, width]
    # Channels: 4 planes representing game state (attackers, defenders, king, current_player)
    dummy_input = torch.randn(1, 4, board_size, board_size)
    
    # Export to ONNX
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
        verbose=False
    )
    
    print(f"  ✓ ONNX model saved to: {output_path}")
    
    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"  File size: {file_size:.2f} MB")


def verify_onnx_model(onnx_path, network, board_size):
    """
    Verify that the ONNX model produces the same outputs as the PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model
        network: Original PyTorch network
        board_size: Size of the game board
    
    Returns:
        bool: True if outputs match within tolerance
    """
    print(f"\nVerifying ONNX model...")
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠ Warning: onnxruntime not installed, skipping verification")
        print("  Install with: pip install onnxruntime")
        return None
    
    # Create test input
    test_input = torch.randn(1, 4, board_size, board_size)
    
    # Get PyTorch output
    network.eval()
    with torch.no_grad():
        pytorch_policy, pytorch_value = network(test_input)
        pytorch_policy = pytorch_policy.numpy()
        pytorch_value = pytorch_value.numpy()
    
    # Get ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    onnx_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_outputs = ort_session.run(None, onnx_inputs)
    onnx_policy, onnx_value = onnx_outputs
    
    # Compare outputs
    policy_diff = np.abs(pytorch_policy - onnx_policy).max()
    value_diff = np.abs(pytorch_value - onnx_value).max()
    
    tolerance = 1e-5
    policy_match = policy_diff < tolerance
    value_match = value_diff < tolerance
    
    print(f"  Policy output difference: {policy_diff:.2e} (tolerance: {tolerance:.2e})")
    print(f"  Value output difference: {value_diff:.2e} (tolerance: {tolerance:.2e})")
    
    if policy_match and value_match:
        print("  ✓ Verification passed! ONNX model outputs match PyTorch model")
        return True
    else:
        print("  ✗ Verification failed! Outputs differ beyond tolerance")
        if not policy_match:
            print(f"    Policy mismatch: {policy_diff:.2e}")
        if not value_match:
            print(f"    Value mismatch: {value_diff:.2e}")
        return False


def print_model_info(onnx_path):
    """Print information about the ONNX model structure."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        
        print(f"\nONNX Model Information:")
        print(f"  Opset version: {model.opset_import[0].version}")
        print(f"  Input:")
        for input_tensor in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"    - {input_tensor.name}: {shape}")
        print(f"  Output:")
        for output_tensor in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"    - {output_tensor.name}: {shape}")
    except ImportError:
        print("\n  (Install 'onnx' package to see detailed model info: pip install onnx)")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch checkpoint to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Brandubh model (7x7)
  python convert_to_onnx.py checkpoints_brandubh/jack_nothrone/best_model.pth
  
  # Convert Tablut model (9x9) with custom output name
  python convert_to_onnx.py checkpoints_tablut/best_model.pth --game tablut --output tablut.onnx
  
  # Convert Hnefatafl model (11x11)
  python convert_to_onnx.py checkpoints_hnefatafl/best_model.pth --game hnefatafl

Notes:
  - The ONNX model can be used with ONNX Runtime in JavaScript, Python, C++, etc.
  - For JavaScript: Use onnxruntime-web (https://github.com/microsoft/onnxruntime)
  - Input shape: [batch_size, 4, board_size, board_size]
  - 4 channels: attackers, defenders, king, current_player
  - Outputs: policy_logits and value
        """
    )
    
    parser.add_argument('checkpoint', type=str,
                       help='Path to PyTorch checkpoint file (.pth)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for ONNX file (default: same name as checkpoint with .onnx extension)')
    parser.add_argument('--game', type=str, choices=['brandubh', 'tablut', 'hnefatafl'],
                       default='brandubh',
                       help='Game type (default: brandubh)')
    parser.add_argument('--opset', type=int, default=12,
                       help='ONNX opset version (default: 12 for broad compatibility)')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification step')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found")
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = os.path.join(checkpoint_dir, f"{checkpoint_name}.onnx")
    
    print("=" * 70)
    print("PyTorch to ONNX Converter")
    print("=" * 70)
    
    # Load checkpoint
    network, config, board_size = load_checkpoint(args.checkpoint, args.game)
    
    # Export to ONNX
    export_to_onnx(network, args.output, board_size, args.opset)
    
    # Verify if requested
    if not args.no_verify:
        verify_onnx_model(args.output, network, board_size)
    
    # Print model info
    print_model_info(args.output)
    
    print("\n" + "=" * 70)
    print("Conversion completed successfully!")
    print("=" * 70)
    print(f"\nYou can now use the ONNX model in:")
    print(f"  - JavaScript (onnxruntime-web)")
    print(f"  - Python (onnxruntime)")
    print(f"  - C++ (onnxruntime)")
    print(f"  - Mobile (iOS/Android)")
    print(f"\nONNX file: {args.output}")


if __name__ == "__main__":
    main()
