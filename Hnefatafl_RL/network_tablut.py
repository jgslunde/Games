"""
Compact AlphaZero-style neural network for Tablut (9x9).
Architecture optimized for training on personal computers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class TablutNet(nn.Module):
    """
    Compact neural network for Tablut (9x9).
    
    Input: 4 planes (9x9) - [attackers, defenders, king, current_player]
    Output: 
        - Policy: probability distribution over all possible moves
        - Value: estimated win probability for current player
    
    Policy encoding:
    Each move is encoded as: from_square (81 positions) * direction (4 directions) * distance (1-8)
    Total policy size: 81 * 4 * 8 = 2592 possible moves (many will be illegal)
    
    Architecture:
    - Small number of residual blocks for efficiency
    - Compact channel count for faster training
    """
    
    def __init__(self, num_res_blocks: int = 4, num_channels: int = 64):
        super().__init__()
        
        self.num_channels = num_channels
        
        # Initial convolution
        self.conv_input = nn.Conv2d(4, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head - compact AlphaZero design
        # Use tiny 1x1 conv (2 channels is standard) to compress features
        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        # Policy output: from_square (81) * direction (4) * distance (8)
        # FC layer: 2*9*9=162 inputs
        self.fc_policy = nn.Linear(2 * 9 * 9, 81 * 4 * 8)
        
        # Value head - compact AlphaZero design
        # Use tiny 1x1 conv (1 channel is standard) to compress features
        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        # FC layers: 1*9*9=81 inputs
        self.fc_value1 = nn.Linear(1 * 9 * 9, 64)
        self.fc_value2 = nn.Linear(64, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: batch of board states, shape (batch, 4, 9, 9)
        
        Returns:
            policy_logits: shape (batch, 2592)
            value: shape (batch, 1)
        """
        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.reshape(p.size(0), -1)  # Use reshape instead of view for memory format compatibility
        policy_logits = self.fc_policy(p)
        
        # Value head
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        value = torch.tanh(self.fc_value2(v))
        
        return policy_logits, value
    
    def optimize_for_inference(self, use_compile: bool = True, compile_mode: str = 'default'):
        """
        Optimize network for fast CPU inference using torch.compile.
        
        Args:
            use_compile: whether to use torch.compile (PyTorch 2.0+)
            compile_mode: compilation mode ('default', 'reduce-overhead', 'max-autotune')
        
        Returns:
            optimized network (may be the same object or compiled version)
        """
        self.eval()
        
        # Use torch.compile if available and requested (PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            self = torch.compile(self, mode=compile_mode)
        
        return self


class TablutMoveEncoder:
    """
    Encode/decode moves for Tablut (9x9 board).
    
    Policy encoding:
    - from_square: 81 positions (9x9 board)
    - direction: 4 directions (up, down, left, right)
    - distance: 8 distances (1-8 squares)
    
    Total policy size: 81 * 4 * 8 = 2592
    
    Move encoding formula:
        policy_index = from_square * 32 + direction * 8 + (distance - 1)
    
    where:
        - from_square = from_row * 9 + from_col
        - direction: 0 (up), 1 (down), 2 (left), 3 (right)
        - distance: 1-8 (number of squares to move)
    """
    
    # Direction vectors: (delta_row, delta_col)
    DIRECTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }
    
    @staticmethod
    def encode_move(move: Tuple[int, int, int, int]) -> int:
        """
        Encode a move as a policy index.
        
        Args:
            move: (from_row, from_col, to_row, to_col)
        
        Returns:
            policy_index: integer in [0, 2591]
        """
        from_r, from_c, to_r, to_c = move
        from_square = from_r * 9 + from_c
        
        # Determine direction and distance
        dr = to_r - from_r
        dc = to_c - from_c
        
        if dr != 0:  # vertical move
            direction = 0 if dr < 0 else 1  # up or down
            distance = abs(dr)
        else:  # horizontal move
            direction = 2 if dc < 0 else 3  # left or right
            distance = abs(dc)
        
        policy_index = from_square * 32 + direction * 8 + (distance - 1)
        return policy_index
    
    @staticmethod
    def decode_move(policy_index: int) -> Tuple[int, int, int, int]:
        """
        Decode a policy index to a move.
        
        Args:
            policy_index: integer in [0, 2591]
        
        Returns:
            move: (from_row, from_col, to_row, to_col)
        """
        from_square = policy_index // 32
        remainder = policy_index % 32
        direction = remainder // 8
        distance = (remainder % 8) + 1
        
        from_r = from_square // 9
        from_c = from_square % 9
        
        dr, dc = TablutMoveEncoder.DIRECTIONS[direction]
        to_r = from_r + dr * distance
        to_c = from_c + dc * distance
        
        return (from_r, from_c, to_r, to_c)
    
    @staticmethod
    def get_legal_move_mask(game) -> np.ndarray:
        """
        Get a mask of legal moves for the current game state.
        
        Args:
            game: Tablut game instance
        
        Returns:
            mask: binary array of shape (2592,) where 1 = legal, 0 = illegal
        """
        mask = np.zeros(2592, dtype=np.float32)
        legal_moves = game.get_legal_moves()
        
        for move in legal_moves:
            idx = TablutMoveEncoder.encode_move(move)
            mask[idx] = 1.0
        
        return mask
    
    @staticmethod
    def moves_to_policy(legal_moves) -> np.ndarray:
        """
        Convert list of legal moves to policy indices.
        
        Args:
            legal_moves: list of (from_row, from_col, to_row, to_col)
        
        Returns:
            indices: array of policy indices
        """
        return np.array([TablutMoveEncoder.encode_move(move) for move in legal_moves])


def test_move_encoding():
    """Test move encoding/decoding."""
    print("Testing Tablut move encoding...")
    
    # Test some moves
    test_moves = [
        (4, 4, 0, 4),  # King up 4 squares
        (4, 4, 4, 8),  # King right 4 squares
        (0, 4, 0, 0),  # Attacker left 4 squares
        (4, 0, 4, 1),  # Attacker right 1 square
        (2, 2, 6, 2),  # Move down 4 squares
        (7, 7, 7, 0),  # Move left 7 squares
    ]
    
    for move in test_moves:
        idx = TablutMoveEncoder.encode_move(move)
        decoded = TablutMoveEncoder.decode_move(idx)
        print(f"Move {move} -> index {idx} -> decoded {decoded}")
        assert decoded == move, f"Encoding/decoding failed: {move} != {decoded}"
    
    print("All encoding tests passed!")
    
    # Test full range
    print(f"\nTesting full policy range (0-2591)...")
    for idx in range(2592):
        move = TablutMoveEncoder.decode_move(idx)
        encoded = TablutMoveEncoder.encode_move(move)
        assert encoded == idx, f"Round-trip failed at index {idx}: {encoded} != {idx}"
    
    print("Full range test passed!")


def test_network():
    """Test network forward pass."""
    print("\nTesting TablutNet...")
    
    net = TablutNet(num_res_blocks=4, num_channels=64)
    
    # Create dummy input (batch_size=2, 4 planes, 9x9)
    x = torch.randn(2, 4, 9, 9)
    
    # Forward pass
    policy_logits, value = net(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Policy logits shape: {policy_logits.shape} (expected: [2, 2592])")
    print(f"Value shape: {value.shape} (expected: [2, 1])")
    
    assert policy_logits.shape == (2, 2592), f"Wrong policy shape: {policy_logits.shape}"
    assert value.shape == (2, 1), f"Wrong value shape: {value.shape}"
    assert value.min() >= -1.0 and value.max() <= 1.0, f"Value out of range: [{value.min()}, {value.max()}]"
    
    print("Network test passed!")


if __name__ == "__main__":
    test_move_encoding()
    test_network()
