"""
Compact AlphaZero-style neural network for Brandubh.
Architecture optimized for training on personal computers.

History planes support: The network can accept T timesteps of history,
where each timestep has 3 piece planes (attackers, defenders, king) plus
1 current player plane. Total input planes = T * 3 + 1.
When T=1, this equals 4 planes (backward compatible with older checkpoints).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class SEResidualBlock(nn.Module):
    """Residual block with batch normalization and squeeze-and-excitation."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Squeeze and Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE scaling
        b, c, _, _ = out.size()
        y = self.se(out).view(b, c, 1, 1)
        out = out * y
        
        out += residual
        out = F.relu(out)
        return out


class BrandubhNet(nn.Module):
    """
    Compact neural network for Brandubh with optional history planes.
    
    Input: (T * 3 + 1) planes (7x7) where T is history length:
        - T * 3 planes for piece positions (attackers, defenders, king) for each timestep
        - 1 plane for current player
        When T=1 (default), this is 4 planes: [attackers, defenders, king, current_player]
    
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
    
    def __init__(self, num_res_blocks: int = 4, num_channels: int = 64, value_head_hidden_size: int = 64,
                 history_length: int = 1):
        """
        Initialize the network.
        
        Args:
            num_res_blocks: Number of residual blocks
            num_channels: Number of channels in convolutional layers
            value_head_hidden_size: Hidden size of value head FC layer
            history_length: Number of timesteps of history (T). 
                           Input planes = T * 3 + 1 (piece planes for each timestep + player plane)
                           T=1 gives 4 planes (backward compatible)
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.value_head_hidden_size = value_head_hidden_size
        self.history_length = history_length
        
        # Calculate input channels: T * 3 piece planes + 1 player plane
        self.input_channels = history_length * 3 + 1
        
        # Initial convolution - input channels depends on history length
        self.conv_input = nn.Conv2d(self.input_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            SEResidualBlock(num_channels) for _ in range(num_res_blocks)
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
        self.fc_value1 = nn.Linear(1 * 7 * 7, value_head_hidden_size)
        self.fc_value2 = nn.Linear(value_head_hidden_size, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: batch of board states, shape (batch, input_channels, 7, 7)
               where input_channels = history_length * 3 + 1
        
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
    
    def load_state_dict_compatible(self, state_dict, strict=True):
        """
        Load state dict with backward compatibility for older checkpoints.
        
        Older checkpoints (history_length=1, input_channels=4) can be loaded into
        networks with history_length>1 by expanding the input convolution weights.
        
        Args:
            state_dict: State dictionary to load
            strict: If True, raise error on missing/unexpected keys (default True)
        """
        # Check if we need to adapt input convolution weights
        saved_input_weight = state_dict.get('conv_input.weight', None)
        
        if saved_input_weight is not None:
            saved_channels = saved_input_weight.shape[1]
            expected_channels = self.input_channels
            
            if saved_channels != expected_channels:
                # Need to adapt weights
                if saved_channels == 4 and expected_channels > 4:
                    # Loading old checkpoint (4 channels) into new network (more channels)
                    # Strategy: Copy the 4 learned weights to the first 4 channels,
                    # initialize the remaining channels to zero
                    print(f"Adapting checkpoint: {saved_channels} input channels -> {expected_channels} input channels")
                    
                    new_weight = torch.zeros(
                        saved_input_weight.shape[0],  # out_channels (num_channels)
                        expected_channels,             # in_channels
                        saved_input_weight.shape[2],  # kernel_h
                        saved_input_weight.shape[3],  # kernel_w
                        dtype=saved_input_weight.dtype
                    )
                    
                    # Copy existing weights to first 4 channels
                    # Old format: [attackers, defenders, king, player]
                    # New format: [attackers_t0, defenders_t0, king_t0, attackers_t1, defenders_t1, king_t1, ..., player]
                    # Map old[0:3] -> new[0:3] (piece planes for t=0)
                    # Map old[3] -> new[-1] (player plane)
                    new_weight[:, 0:3, :, :] = saved_input_weight[:, 0:3, :, :]  # Piece planes
                    new_weight[:, -1, :, :] = saved_input_weight[:, 3, :, :]     # Player plane
                    
                    state_dict['conv_input.weight'] = new_weight
                    
                elif saved_channels > expected_channels:
                    # Loading newer checkpoint into older network - truncate
                    print(f"Truncating checkpoint: {saved_channels} input channels -> {expected_channels} input channels")
                    state_dict['conv_input.weight'] = saved_input_weight[:, :expected_channels, :, :]
        
        # Use standard load_state_dict
        return super().load_state_dict(state_dict, strict=strict)
    
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


class StateHistory:
    """
    Manages game state history for neural network input with history planes.
    
    This class maintains a rolling buffer of piece planes from recent game states,
    which can be combined with the current player plane to create the full network input.
    
    For history_length=T, the output has T*3 + 1 planes:
    - Planes 0-2: Current state piece planes (attackers, defenders, king)
    - Planes 3-5: Previous state piece planes (t-1)
    - ...
    - Planes (T-1)*3 to T*3-1: Oldest state piece planes (t-T+1)
    - Plane T*3: Current player plane
    
    When history_length=1, this produces 4 planes (backward compatible).
    """
    
    def __init__(self, history_length: int = 1, board_size: int = 7):
        """
        Initialize state history tracker.
        
        Args:
            history_length: Number of timesteps to track (T)
            board_size: Size of the game board
        """
        self.history_length = history_length
        self.board_size = board_size
        self.piece_planes_per_state = 3  # attackers, defenders, king
        
        # Buffer to store piece planes (most recent first)
        # Each entry is shape (3, board_size, board_size)
        self.history = []
    
    def reset(self):
        """Clear the history buffer."""
        self.history = []
    
    def push(self, piece_planes: np.ndarray):
        """
        Add piece planes to history.
        
        Args:
            piece_planes: Shape (3, board_size, board_size) - [attackers, defenders, king]
        """
        # Add to front of list (most recent first)
        self.history.insert(0, piece_planes.copy())
        
        # Keep only the last history_length entries
        if len(self.history) > self.history_length:
            self.history = self.history[:self.history_length]
    
    def get_state_with_history(self, current_player: int) -> np.ndarray:
        """
        Get full state representation with history planes.
        
        Args:
            current_player: Current player (0 for attacker, 1 for defender)
        
        Returns:
            state: Shape (history_length * 3 + 1, board_size, board_size)
                   Most recent state first, then progressively older states,
                   finally current player plane.
        """
        num_planes = self.history_length * 3 + 1
        state = np.zeros((num_planes, self.board_size, self.board_size), dtype=np.float32)
        
        # Fill in history planes
        for t in range(min(len(self.history), self.history_length)):
            start_idx = t * 3
            state[start_idx:start_idx+3] = self.history[t]
        
        # For missing history (e.g., at start of game), planes remain zero
        # This is the standard approach used in AlphaGo/AlphaZero
        
        # Add current player plane at the end
        state[-1] = np.full((self.board_size, self.board_size), current_player, dtype=np.float32)
        
        return state
    
    def clone(self) -> 'StateHistory':
        """Create a deep copy of the history tracker."""
        new_history = StateHistory(self.history_length, self.board_size)
        new_history.history = [h.copy() for h in self.history]
        return new_history
    
    @staticmethod
    def from_game(game, history_length: int = 1) -> 'StateHistory':
        """
        Create a StateHistory initialized with the current game state.
        
        Note: This only captures the current state. For full history,
        you need to track states as moves are made.
        
        Args:
            game: Game instance with get_piece_planes() method
            history_length: Number of timesteps to track
        
        Returns:
            StateHistory instance with current state pushed
        """
        history = StateHistory(history_length, board_size=game.board.shape[0])
        history.push(game.get_piece_planes())
        return history


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
