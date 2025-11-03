"""
Compact AlphaZero-style neural network for Brandubh.
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


class BrandubhNet(nn.Module):
    """
    Compact neural network for Brandubh.
    
    Input: 4 planes (7x7) - [attackers, defenders, king, current_player]
    Output: 
        - Policy: probability distribution over all possible moves
        - Value: estimated win probability for current player
    
    Policy encoding:
    Each move is encoded as: from_square (49 positions) * direction (4 directions) * distance (1-6)
    Total policy size: 49 * 4 * 6 = 1176 possible moves (many will be illegal)
    
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
        # Policy output: from_square (49) * direction (4) * distance (6)
        # FC layer is now much smaller: 2*7*7=98 inputs instead of 32*7*7=1568
        self.fc_policy = nn.Linear(2 * 7 * 7, 49 * 4 * 6)
        
        # Value head - compact AlphaZero design
        # Use tiny 1x1 conv (1 channel is standard) to compress features
        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        # FC layers are now much smaller: 1*7*7=49 inputs instead of 16*7*7=784
        self.fc_value1 = nn.Linear(1 * 7 * 7, 64)
        self.fc_value2 = nn.Linear(64, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: batch of board states, shape (batch, 4, 7, 7)
        
        Returns:
            policy_logits: shape (batch, 1176)
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
        v = v.reshape(v.size(0), -1)  # Use reshape instead of view for memory format compatibility
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


class MoveEncoder:
    """
    Encodes and decodes moves to/from policy vector indices.
    
    Policy encoding scheme:
    - Each square (49 total) can initiate a move
    - From each square, can move in 4 directions (up, down, left, right)
    - Can move 1-6 squares in each direction
    - Total: 49 * 4 * 6 = 1176 possible move encodings
    
    Move format: (from_row, from_col, to_row, to_col)
    Policy index: from_square * 24 + direction * 6 + (distance - 1)
    """
    
    # Direction mappings
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
            policy_index: integer in [0, 1175]
        """
        from_r, from_c, to_r, to_c = move
        from_square = from_r * 7 + from_c
        
        # Determine direction and distance
        dr = to_r - from_r
        dc = to_c - from_c
        
        if dr != 0:  # vertical move
            direction = 0 if dr < 0 else 1  # up or down
            distance = abs(dr)
        else:  # horizontal move
            direction = 2 if dc < 0 else 3  # left or right
            distance = abs(dc)
        
        policy_index = from_square * 24 + direction * 6 + (distance - 1)
        return policy_index
    
    @staticmethod
    def decode_move(policy_index: int) -> Tuple[int, int, int, int]:
        """
        Decode a policy index to a move.
        
        Args:
            policy_index: integer in [0, 1175]
        
        Returns:
            move: (from_row, from_col, to_row, to_col)
        """
        from_square = policy_index // 24
        remainder = policy_index % 24
        direction = remainder // 6
        distance = (remainder % 6) + 1
        
        from_r = from_square // 7
        from_c = from_square % 7
        
        dr, dc = MoveEncoder.DIRECTIONS[direction]
        to_r = from_r + dr * distance
        to_c = from_c + dc * distance
        
        return (from_r, from_c, to_r, to_c)
    
    @staticmethod
    def get_legal_move_mask(game) -> np.ndarray:
        """
        Get a mask of legal moves for the current game state.
        
        Args:
            game: Brandubh game instance
        
        Returns:
            mask: binary array of shape (1176,) where 1 = legal, 0 = illegal
        """
        mask = np.zeros(1176, dtype=np.float32)
        legal_moves = game.get_legal_moves()
        
        for move in legal_moves:
            idx = MoveEncoder.encode_move(move)
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
        return np.array([MoveEncoder.encode_move(move) for move in legal_moves])


def test_move_encoding():
    """Test move encoding/decoding."""
    print("Testing move encoding...")
    
    # Test some moves
    test_moves = [
        (3, 3, 0, 3),  # King up 3 squares
        (3, 3, 3, 6),  # King right 3 squares
        (0, 3, 0, 0),  # Attacker left 3 squares
        (3, 0, 3, 1),  # Attacker right 1 square
    ]
    
    for move in test_moves:
        encoded = MoveEncoder.encode_move(move)
        decoded = MoveEncoder.decode_move(encoded)
        print(f"Move: {move} -> Encoded: {encoded} -> Decoded: {decoded}")
        assert move == decoded, f"Encoding failed: {move} != {decoded}"
    
    print("All tests passed!")


if __name__ == "__main__":
    # Test the network
    print("Creating network...")
    net = BrandubhNet(num_res_blocks=4, num_channels=64)
    
    print("\nNetwork architecture:")
    print(net)
    
    print("\nParameter count:")
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    x = torch.randn(batch_size, 4, 7, 7)
    policy_logits, value = net(x)
    print(f"Input shape: {x.shape}")
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test move encoding
    print("\n" + "="*50)
    test_move_encoding()
