"""
Network architecture configuration (torch-free, so configs can be read and validated anywhere).

A NetworkConfig describes one of the architectures in tafl_network.py:

    arch                 "tafl": the configurable network of tafl_network.py
                         "legacy": the fixed BrandubhNet / TablutNet / HnefataflNet classes
                         (only for checkpoints written before the config system)
    channels             width of the residual trunk
    blocks               list of block kinds, in order (or an int: that many "res" blocks):
                           "res"    3x3 conv, 3x3 conv (the classic AlphaZero block)
                           "line"   1x1 conv down to `bottleneck` x channels, then a row conv
                                    (1 x 2N-1) and a column conv (2N-1 x 1) added together, then
                                    1x1 up. The kernels are centred and 2N-1 long, so one block
                                    gives every square a view of its whole row and column, which
                                    is where all Tafl rules act.
                           "line3"  like "line" with a 3x3 conv before the row/column convs
                                    (local shapes and lines in one block)
                           "thick"  like "line" with 3 x 2N-1 and 2N-1 x 3 kernels (sees the
                                    neighbouring rows and columns too; about three times the cost
                                    of "line")
                           "attention"  transformer block over the squares (self-attention with
                                    a learned bias per square offset, then a feed-forward
                                    layer): every square sees every square in one block, also
                                    diagonally and far away. Ignores global_pool.
    attention_heads      heads of the attention blocks (channels must be divisible by it)
    bottleneck           inner width of line/line3/thick blocks as a fraction of `channels`
    squeeze_excitation   squeeze-and-excitation in "res" blocks (the old BrandubhNet had it)
    global_pool          "none", "every_other" or "all": blocks that add a bias computed from
                         board-wide average and maximum features (KataGo style), for material,
                         encirclement and other whole-board facts
    global_pool_channels pooled channels of that bias
    tie_blocks           all blocks share one set of weights (same compute, fewer parameters;
                         the block is applied len(blocks) times)
    policy_head          "dense":    1x1 conv to a few planes, then one dense layer to all moves
                                     (the old head; most of the parameters, no weight sharing
                                     between squares)
                         "conv":     1x1 conv to 4(N-1) planes = (direction, distance) logits for
                                     every from-square, in MoveEncoder index order
                         "bilinear": logit(from, to) = q_from . k_to + u_from + v_to with q, k, u, v
                                     from 1x1 convs; the landing square is judged directly
    value_head           "dense":  1x1 conv to one plane, then two dense layers (the old head)
                         "pooled": 1x1 conv to a few planes, board-wide average and maximum, plus
                                   one spatial plane, then two dense layers
    features             which input planes the network gets (features.FeatureSpec)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Union

from features import FeatureSpec

BLOCK_KINDS = ("res", "line", "line3", "thick", "attention")
POLICY_HEADS = ("dense", "conv", "bilinear")
VALUE_HEADS = ("dense", "pooled")
GLOBAL_POOL_MODES = ("none", "every_other", "all")
ARCHS = ("tafl", "legacy")


@dataclass
class NetworkConfig:
    arch: str = "tafl"
    channels: int = 64
    blocks: Union[List[str], int] = field(default_factory=lambda: ["res"] * 6)
    bottleneck: float = 0.5
    squeeze_excitation: bool = False
    global_pool: str = "none"
    global_pool_channels: int = 16
    tie_blocks: bool = False
    attention_heads: int = 4
    policy_head: str = "conv"
    policy_dense_channels: int = 2
    policy_bilinear_dim: int = 32
    value_head: str = "pooled"
    value_hidden: int = 64
    value_pool_channels: int = 16
    features: FeatureSpec = field(default_factory=FeatureSpec)

    def __post_init__(self):
        if isinstance(self.blocks, int):
            self.blocks = ["res"] * self.blocks
        self.blocks = list(self.blocks)
        if isinstance(self.features, dict):
            self.features = FeatureSpec.from_dict(self.features)
        self.validate()

    def validate(self):
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {self.arch!r}")
        if self.channels < 2:
            raise ValueError("channels must be at least 2")
        if not self.blocks:
            raise ValueError("blocks must name at least one block")
        bad = [b for b in self.blocks if b not in BLOCK_KINDS]
        if bad:
            raise ValueError(f"Unknown block kinds {bad}; choose from {BLOCK_KINDS}")
        if not 0 < self.bottleneck <= 1:
            raise ValueError("bottleneck must be in (0, 1]")
        if self.global_pool not in GLOBAL_POOL_MODES:
            raise ValueError(f"global_pool must be one of {GLOBAL_POOL_MODES}, got {self.global_pool!r}")
        if self.policy_head not in POLICY_HEADS:
            raise ValueError(f"policy_head must be one of {POLICY_HEADS}, got {self.policy_head!r}")
        if self.value_head not in VALUE_HEADS:
            raise ValueError(f"value_head must be one of {VALUE_HEADS}, got {self.value_head!r}")
        if self.tie_blocks and len(set(self.blocks)) > 1:
            raise ValueError("tie_blocks needs blocks of one kind only")
        if "attention" in self.blocks and self.channels % self.attention_heads:
            raise ValueError(f"channels ({self.channels}) must be divisible by attention_heads "
                             f"({self.attention_heads})")
        if self.arch == "legacy":
            legacy_ok = (all(b == "res" for b in self.blocks) and self.policy_head == "dense"
                         and self.value_head == "dense" and self.global_pool == "none"
                         and not self.tie_blocks and not self.features.constant_planes
                         and not self.features.mobility_planes and not self.features.region_plane)
            if not legacy_ok:
                raise ValueError("arch 'legacy' supports only 'res' blocks, dense heads, no global "
                                 "pooling, no tied blocks and only history planes as features")

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NetworkConfig":
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown network settings: {sorted(unknown)}; "
                             f"valid ones are {sorted(cls.__dataclass_fields__)}")
        return cls(**data)

    def describe(self) -> str:
        """Short label, e.g. '6x64 line (tied), gpool every_other, conv policy, pooled value, 13 planes'."""
        kinds = sorted(set(self.blocks), key=self.blocks.index)
        trunk = f"{self.num_blocks}x{self.channels} {'/'.join(kinds)}"
        if self.arch == "legacy":
            trunk += " legacy"
        if self.squeeze_excitation and "res" in kinds:
            trunk += " SE"
        if self.tie_blocks:
            trunk += " (tied)"
        parts = [trunk]
        if self.global_pool != "none":
            parts.append(f"gpool {self.global_pool}")
        parts += [f"{self.policy_head} policy", f"{self.value_head} value",
                  f"{self.features.num_planes} planes"]
        return ", ".join(parts)
