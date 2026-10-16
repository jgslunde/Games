"""
Configurable AlphaZero-style network for all three Tafl board sizes, built from a NetworkConfig
(see network_config.py for the options and what they are for), plus helpers that build or load a
network from a checkpoint, including checkpoints of the older fixed classes.

Input:  (batch, FeatureSpec.num_planes, N, N), see features.py
Output: policy logits (batch, N*N*4*(N-1)) in MoveEncoder index order
        value (batch, 1) in [-1, 1] for the player to move
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FeatureSpec
from network_config import NetworkConfig


def _conv_bn(cin: int, cout: int, kernel, padding) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=padding, bias=False), nn.BatchNorm2d(cout))


# ---------------------------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """Per-channel gates from the board-wide mean (as in the old BrandubhNet)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean((2, 3))))))
        return x * gate[:, :, None, None]


class GlobalPoolBias(nn.Module):
    """
    KataGo-style global pooling: a few pooled planes (board-wide average and maximum) become a
    per-channel bias added to every square. Cheap, and the only way for a convolution-only
    network to use whole-board facts such as material.
    """

    def __init__(self, channels: int, pool_channels: int):
        super().__init__()
        self.pool_conv = _conv_bn(channels, pool_channels, 1, 0)
        self.fc = nn.Linear(2 * pool_channels, channels)

    def forward(self, x):
        pooled = F.relu(self.pool_conv(x))
        features = torch.cat([pooled.mean((2, 3)), pooled.amax((2, 3))], dim=1)
        return x + self.fc(features)[:, :, None, None]


class ResBlock(nn.Module):
    """3x3 conv, 3x3 conv, residual add; optional global-pool bias and squeeze-excitation."""

    def __init__(self, channels: int, squeeze_excitation: bool = False, global_pool_channels: int = 0):
        super().__init__()
        self.conv1 = _conv_bn(channels, channels, 3, 1)
        self.global_pool = GlobalPoolBias(channels, global_pool_channels) if global_pool_channels else None
        self.conv2 = _conv_bn(channels, channels, 3, 1)
        self.se = SqueezeExcitation(channels) if squeeze_excitation else None

    def forward(self, x):
        h = F.relu(self.conv1(x))
        if self.global_pool is not None:
            h = self.global_pool(h)
        h = self.conv2(h)
        if self.se is not None:
            h = self.se(h)
        return F.relu(x + h)


class LineBlock(nn.Module):
    """
    Bottleneck block whose kernels span a whole row and a whole column:
        1x1 conv down -> [3x3 conv (line3)] -> (1x(2N-1) conv + (2N-1)x1 conv) -> 1x1 conv up
        -> residual add
    "thick" uses 3x(2N-1) and (2N-1)x3 kernels instead.

    The line kernels are 2N-1 long and centred, so every square sees every other square of its
    row and column, with one weight per offset (-(N-1) ... N-1). A kernel only N long would let a
    square see just N//2 squares to each side: from a corner, only half of its edge line.
    """

    def __init__(self, channels: int, board_size: int, kind: str, bottleneck: float = 0.5,
                 global_pool_channels: int = 0):
        super().__init__()
        inner = max(4, int(round(channels * bottleneck)))
        rows = 3 if kind == "thick" else 1
        length = 2 * board_size - 1
        self.down = _conv_bn(channels, inner, 1, 0)
        self.local = _conv_bn(inner, inner, 3, 1) if kind == "line3" else None
        self.row = nn.Conv2d(inner, inner, (rows, length), padding=(rows // 2, board_size - 1),
                             bias=False)
        self.col = nn.Conv2d(inner, inner, (length, rows), padding=(board_size - 1, rows // 2),
                             bias=False)
        self.bn_line = nn.BatchNorm2d(inner)
        self.global_pool = GlobalPoolBias(inner, global_pool_channels) if global_pool_channels else None
        self.up = _conv_bn(inner, channels, 1, 0)

    def forward(self, x):
        h = F.relu(self.down(x))
        if self.local is not None:
            h = F.relu(self.local(h))
        h = F.relu(self.bn_line(self.row(h) + self.col(h)))
        if self.global_pool is not None:
            h = self.global_pool(h)
        return F.relu(x + self.up(h))


def make_block(kind: str, config: NetworkConfig, board_size: int, global_pool_channels: int) -> nn.Module:
    if kind == "res":
        return ResBlock(config.channels, config.squeeze_excitation, global_pool_channels)
    return LineBlock(config.channels, board_size, kind, config.bottleneck, global_pool_channels)


# ---------------------------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------------------------

def move_index_tables(board_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every policy index (from_square * 4(N-1) + direction * (N-1) + distance - 1): the index
    of the landing square, and whether the move stays on the board. Directions as in MoveEncoder:
    0 up, 1 down, 2 left, 3 right.
    """
    n = board_size
    to_square = np.zeros(n * n * 4 * (n - 1), dtype=np.int64)
    on_board = np.zeros_like(to_square, dtype=bool)
    for r in range(n):
        for c in range(n):
            for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                for dist in range(1, n):
                    idx = (r * n + c) * 4 * (n - 1) + d * (n - 1) + dist - 1
                    tr, tc = r + dr * dist, c + dc * dist
                    if 0 <= tr < n and 0 <= tc < n:
                        to_square[idx] = tr * n + tc
                        on_board[idx] = True
                    else:
                        to_square[idx] = r * n + c  # placeholder, masked out
    return to_square, on_board


class DensePolicyHead(nn.Module):
    """The old head: 1x1 conv to a few planes, then one dense layer over the whole board."""

    def __init__(self, channels: int, board_size: int, planes: int = 2):
        super().__init__()
        n = board_size
        self.conv = _conv_bn(channels, planes, 1, 0)
        self.fc = nn.Linear(planes * n * n, n * n * 4 * (n - 1))

    def forward(self, x):
        return self.fc(F.relu(self.conv(x)).flatten(1))


class ConvPolicyHead(nn.Module):
    """One logit plane per (direction, distance); flattened in MoveEncoder order."""

    def __init__(self, channels: int, board_size: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 4 * (board_size - 1), 1)

    def forward(self, x):
        # (B, 4(N-1), N, N) -> (B, N, N, 4(N-1)) -> (B, N*N*4(N-1)): from_square major
        return self.conv(x).permute(0, 2, 3, 1).flatten(1)


class BilinearPolicyHead(nn.Module):
    """logit(from, to) = q_from . k_to / sqrt(d) + u_from + v_to, gathered into MoveEncoder order."""

    def __init__(self, channels: int, board_size: int, dim: int = 32):
        super().__init__()
        self.n, self.dim = board_size, dim
        self.query = nn.Conv2d(channels, dim, 1)
        self.key = nn.Conv2d(channels, dim, 1)
        self.from_bias = nn.Conv2d(channels, 1, 1)
        self.to_bias = nn.Conv2d(channels, 1, 1)
        to_square, on_board = move_index_tables(board_size)
        from_square = np.arange(len(to_square)) // (4 * (board_size - 1))
        self.register_buffer("flat_index", torch.from_numpy(from_square * board_size ** 2 + to_square))
        self.register_buffer("off_board", torch.from_numpy(~on_board))

    def forward(self, x):
        squares = self.n * self.n
        q = self.query(x).flatten(2)  # (B, d, N²)
        k = self.key(x).flatten(2)
        scores = torch.bmm(q.transpose(1, 2), k) / math.sqrt(self.dim)  # (B, from, to)
        scores = scores + self.from_bias(x).flatten(1)[:, :, None] + self.to_bias(x).flatten(1)[:, None, :]
        logits = torch.index_select(scores.reshape(-1, squares * squares), 1, self.flat_index)
        return logits.masked_fill(self.off_board, -1e4)


class DenseValueHead(nn.Module):
    """The old head: 1x1 conv to one plane, then two dense layers."""

    def __init__(self, channels: int, board_size: int, hidden: int = 64):
        super().__init__()
        self.conv = _conv_bn(channels, 1, 1, 0)
        self.fc1 = nn.Linear(board_size * board_size, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.conv(x)).flatten(1)
        return torch.tanh(self.fc2(F.relu(self.fc1(h))))


class PooledValueHead(nn.Module):
    """Board-wide average and maximum of a few planes, plus one spatial plane, then two dense layers."""

    def __init__(self, channels: int, board_size: int, hidden: int = 64, pool_planes: int = 16):
        super().__init__()
        self.pool_conv = _conv_bn(channels, pool_planes, 1, 0)
        self.spatial_conv = _conv_bn(channels, 1, 1, 0)
        self.fc1 = nn.Linear(2 * pool_planes + board_size * board_size, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        pooled = F.relu(self.pool_conv(x))
        spatial = F.relu(self.spatial_conv(x)).flatten(1)
        h = torch.cat([pooled.mean((2, 3)), pooled.amax((2, 3)), spatial], dim=1)
        return torch.tanh(self.fc2(F.relu(self.fc1(h))))


# ---------------------------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------------------------

class TaflNet(nn.Module):
    """Network built from a NetworkConfig for one board size."""

    def __init__(self, config: NetworkConfig, board_size: int):
        super().__init__()
        if config.arch != "tafl":
            raise ValueError("TaflNet needs arch 'tafl' (use build_network for legacy configs)")
        self.config = config
        self.board_size = board_size
        self.feature_spec: FeatureSpec = config.features
        self.history_length = config.features.history_length
        self.input_channels = config.features.num_planes
        c, n = config.channels, board_size

        self.stem = _conv_bn(self.input_channels, c, 3, 1)

        def pool_channels(i: int) -> int:
            use = (config.global_pool == "all"
                   or (config.global_pool == "every_other" and i % 2 == 0))
            return config.global_pool_channels if use else 0

        if config.tie_blocks:
            # One block, applied num_blocks times (every application shares the weights)
            shared = make_block(config.blocks[0], config, n,
                                config.global_pool_channels if config.global_pool != "none" else 0)
            self.blocks = nn.ModuleList([shared])
            self.block_order = [0] * config.num_blocks
        else:
            self.blocks = nn.ModuleList([make_block(kind, config, n, pool_channels(i))
                                         for i, kind in enumerate(config.blocks)])
            self.block_order = list(range(config.num_blocks))

        # Module names contain "policy" / "value": train_network splits gradient norms by them
        if config.policy_head == "dense":
            self.policy_head = DensePolicyHead(c, n, config.policy_dense_channels)
        elif config.policy_head == "conv":
            self.policy_head = ConvPolicyHead(c, n)
        else:
            self.policy_head = BilinearPolicyHead(c, n, config.policy_bilinear_dim)
        if config.value_head == "dense":
            self.value_head = DenseValueHead(c, n, config.value_hidden)
        else:
            self.value_head = PooledValueHead(c, n, config.value_hidden, config.value_pool_channels)

    def forward(self, x):
        h = F.relu(self.stem(x))
        for i in self.block_order:
            h = self.blocks[i](h)
        return self.policy_head(h), self.value_head(h)

    def load_state_dict_compatible(self, state_dict, strict: bool = True, saved_plane_names=None):
        """
        Load weights, adapting the stem to another set of input planes: planes of the same name
        keep their weights, new planes start at zero, dropped planes are left out.

        Args:
            saved_plane_names: plane names of the saved network (FeatureSpec.plane_names()). If
                               None, the saved network is assumed to have had only history planes.
        """
        key = "stem.0.weight"
        saved = state_dict.get(key)
        if saved is not None and saved.shape[1] != self.input_channels:
            if saved_plane_names is None:
                if (saved.shape[1] - 1) % 3 != 0:
                    raise ValueError(f"Cannot guess the plane layout of a {saved.shape[1]}-plane checkpoint")
                saved_plane_names = FeatureSpec(history_length=(saved.shape[1] - 1) // 3).plane_names()
            print(f"Adapting checkpoint: {saved.shape[1]} input planes -> {self.input_channels} input planes")
            new_weight = torch.zeros(saved.shape[0], self.input_channels, *saved.shape[2:], dtype=saved.dtype)
            for i, name in enumerate(self.feature_spec.plane_names()):
                if name in saved_plane_names:
                    new_weight[:, i] = saved[:, saved_plane_names.index(name)]
            state_dict = {**state_dict, key: new_weight}
        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    def optimize_for_inference(self, use_compile: bool = True, compile_mode: str = 'default'):
        """eval() and, if requested and available, torch.compile; returns the network to call."""
        self.eval()
        if use_compile and hasattr(torch, 'compile'):
            return torch.compile(self, mode=compile_mode)
        return self


# ---------------------------------------------------------------------------------------------
# Building and loading
# ---------------------------------------------------------------------------------------------

def build_network(config: NetworkConfig, board_size: int) -> nn.Module:
    """A network for `config` on an NxN board (a legacy class instance when arch is 'legacy')."""
    if config.arch == "legacy":
        from games import game_spec_for_board
        legacy_class = game_spec_for_board(board_size).legacy_network_class
        return legacy_class(num_res_blocks=config.num_blocks, num_channels=config.channels,
                            value_head_hidden_size=config.value_hidden,
                            history_length=config.features.history_length)
    return TaflNet(config, board_size)


def count_parameters(network: nn.Module) -> int:
    return sum(p.numel() for p in network.parameters())


def describe_network(network: nn.Module) -> str:
    """One line with the architecture and the parameter count."""
    if isinstance(network, TaflNet):
        label = network.config.describe()
    else:
        blocks = len(getattr(network, 'res_blocks', []))
        label = (f"{blocks}x{getattr(network, 'num_channels', '?')} legacy {type(network).__name__}, "
                 f"{getattr(network, 'input_channels', 4)} planes")
    return f"{label}, {count_parameters(network):,} parameters"


def _board_size_from_state_dict(state_dict) -> Optional[int]:
    """Board size from the policy output size of the old dense head, if present."""
    weight = state_dict.get('fc_policy.weight')
    if weight is None:
        return None
    for n in (7, 9, 11):
        if weight.shape[0] == n * n * 4 * (n - 1):
            return n
    return None


def legacy_network_config(config: dict, board_size: int, state_dict=None) -> NetworkConfig:
    """
    The NetworkConfig equivalent of an old checkpoint: sizes from its flat config dict, or, when
    the state dict is given, from the weight shapes (which also covers checkpoints without a config).
    """
    sizes = {}
    if state_dict is not None and 'conv_input.weight' in state_dict:
        stem = state_dict['conv_input.weight']
        sizes = dict(num_channels=stem.shape[0], history_length=(stem.shape[1] - 1) // 3,
                     num_res_blocks=len({k.split('.')[1] for k in state_dict if k.startswith('res_blocks.')}))
        if 'fc_value1.weight' in state_dict:
            sizes['value_head_hidden_size'] = state_dict['fc_value1.weight'].shape[0]
    get = lambda name, default: sizes.get(name, config.get(name, default))
    return NetworkConfig(arch="legacy", channels=get('num_channels', 64),
                         blocks=["res"] * get('num_res_blocks', 4),
                         squeeze_excitation=(board_size == 7),  # BrandubhNet has SE blocks
                         policy_head="dense", value_head="dense",
                         value_hidden=get('value_head_hidden_size', 64),
                         features=FeatureSpec(history_length=get('history_length', 1)))


def checkpoint_board_size(checkpoint: dict) -> Optional[int]:
    """The board size a checkpoint was trained for, if it can be told."""
    config = checkpoint.get('config', {})
    if isinstance(config, dict) and 'board_size' in config:
        return config['board_size']
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('network_state_dict') or checkpoint
    return _board_size_from_state_dict(state_dict)


def network_config_from_checkpoint(checkpoint: dict, board_size: Optional[int] = None) -> NetworkConfig:
    """The NetworkConfig stored in a checkpoint, or the legacy equivalent for older checkpoints."""
    if 'network_config' in checkpoint:
        return NetworkConfig.from_dict(checkpoint['network_config'])
    board_size = board_size or checkpoint_board_size(checkpoint) or 7
    config = checkpoint.get('config', checkpoint)
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('network_state_dict') or checkpoint
    return legacy_network_config(config if isinstance(config, dict) else {}, board_size, state_dict)


def load_weights(network: nn.Module, checkpoint: dict):
    """Load a checkpoint's weights into `network`, adapting the input planes where needed."""
    state_dict = checkpoint.get('network_state_dict') or checkpoint.get('model_state_dict') or checkpoint
    if isinstance(network, TaflNet):
        saved_names = None
        if 'network_config' in checkpoint:
            saved_names = NetworkConfig.from_dict(checkpoint['network_config']).features.plane_names()
        network.load_state_dict_compatible(state_dict, saved_plane_names=saved_names)
    else:
        network.load_state_dict_compatible(state_dict)


def load_network(checkpoint_path: str, board_size: Optional[int] = None, device: str = 'cpu'):
    """
    Build the network a checkpoint was trained with and load its weights.

    Returns:
        (network in eval mode, the checkpoint dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    board_size = board_size or checkpoint_board_size(checkpoint)
    if board_size is None:
        raise ValueError(f"{checkpoint_path} does not say which board it is for; pass board_size")
    network = build_network(network_config_from_checkpoint(checkpoint, board_size), board_size)
    load_weights(network, checkpoint)
    network.to(device)
    network.eval()
    return network, checkpoint
