# Network architectures for the Tafl games

Notes on which neural networks fit Brandubh (7x7), Tablut (9x9) and Hnefatafl (11x11), what was
implemented, and which runs to make. Written for the setup in this directory: AlphaZero-style
self-play, and play in the browser with `onnxruntime-web`, where a slow network means fewer search
nodes per move.

## 1. Summary

- Every Tafl rule acts along a straight line: pieces move like rooks, a capture needs a piece that
  can reach the square behind the victim, the king wins through an open line to a corner, a block
  needs a piece that can reach that line. A 3x3 convolution sees one square in each direction, so
  it needs many layers to follow a line. A kernel that spans a whole row or column sees the line in
  one layer, and on a board of 7 to 11 squares it costs no more than a 3x3 kernel.
- The old dense policy head (one big linear layer from the whole board to every move) holds half
  to two thirds of all parameters and has to learn every from-square separately. A 1x1 convolution
  that outputs one plane per (direction, distance) gives the same output in the same index order
  with a few hundred parameters.
- Measured in the browser (WASM, one thread): the new networks evaluate 4.5x (7x7), 2.3x (9x9) and
  4.5x (11x11) faster than the current ones at the same depth, with 5 to 12 times fewer parameters.
  Playing strength is not measured yet; that is what the configs in section 5 are for.
- Transformers, dilated kernels, 5x5 kernels, depthwise convolutions and squeeze-excitation were
  considered and rejected for this setup (section 3).

## 2. What the network has to compute

| Fact the evaluator needs | Shape of the computation |
|---|---|
| Reach: can the piece on square s move d squares in direction dir? | a line segment: all d squares must be empty |
| Arrival: can some piece of a side move to square t this turn? | the first piece met along each of the 4 lines through t |
| Threat: can piece p be captured next move? | enemy adjacent on one side, empty square on the other side, and Arrival of an enemy on that empty square |
| Escape: does the king have a clear line to a corner? Two of them? | line segments on fixed rows and columns |
| Block: can an attacker land on the king's escape line? | Arrival restricted to a line |
| Encirclement (11x11): do the attackers form a closed ring? | connectivity: a flood fill, iterative |
| Material, side to move, repetition | whole-board counts and history |

A full-row kernel computes Reach exactly in one layer: with an "occupied" input plane,
`Reach(s, right, d) = ReLU(1 - sum of occupied(s+1..s+d))` is one weight pattern, one output
channel per (direction, distance). Arrival is one more such layer, Threat a third. With 3x3
kernels, Reach at distance d needs d layers, because every layer can extend an "all empty so far"
chain by one square. The old 6-block networks cannot represent a 10-square line at all.

Two consequences:

- Zero padding reads an off-board square as "empty". A constant all-ones plane (0 outside the
  board after padding) plus corner and throne planes let the network tell edges and special squares
  apart in the first layer. These planes are free.
- The policy head needs line information at the from-square. A line block as the last block gives
  exactly that; a (direction, distance) head on top of it is "Reach times desirability".

Encirclement is the one thing lines do not solve: whether a ring is closed is an iterative
question. Rings are mostly straight walls, which line kernels see, but 11x11 wants depth more than
width for this. The king-region plane (a flood fill done by the game engine, see section 4) hands
the network the answer directly.

## 3. Architecture choices

### Trunk blocks (`network_config.py`: `blocks`)

| Block | What one layer sees | Verdict | Cost of a 6-block trunk, relative to `res` (browser) |
|---|---|---|---|
| `res`: 3x3, 3x3 | 8 neighbours | local shapes only (sandwiches, forts, diagonal guards) | 1.0 |
| `line`: 1x1 down, 1xN + Nx1, 1x1 up | whole row and column | the main bet; line facts are few, so the narrow inner width is enough | 0.37 to 0.43 |
| `line3`: like `line` with a 3x3 first | local shape, then lines over it | best "complete" block; use on 9x9 and 11x11 if diagonal shapes matter | 0.48 to 0.56 |
| `thick`: 3xN + Nx3 | row, column and the neighbouring rows and columns | merges two thin line layers into one, but loses the nonlinearity between them; same cost as `res` | 0.76 to 1.16 |
| 5x5 | 24 neighbours | still local, +1 reach, 1.7x to 1.9x the cost | rejected |
| dilated 3x3 | taps two squares apart | skips squares, so it cannot check "nothing in between" | rejected |
| depthwise separable | 8 neighbours, 4x to 8x fewer multiply-adds | only 30 % faster in practice, worse per multiply-add than line blocks | rejected |
| transformer (attention over squares) | any square | right for a large GPU network (Leela Chess Zero); at this size and batch 1 it is 2x to 6x slower per evaluation than a line network with the same parameters, and it has to learn the line structure that the line kernel has built in | rejected for the browser |

Other options: `global_pool` adds a KataGo-style bias from board-wide average and maximum
features (material, how many attackers near each corner); it costs about 8 % and is on in the 9x9
and 11x11 configs. `tie_blocks` applies one block several times with shared weights: same compute,
fewer parameters, and an iterative computation, which fits encirclement; untested. Squeeze-
excitation (`squeeze_excitation`) costs 5 to 10 % at 7x7 for little gain; the global-pool bias
does the same job in one place.

Width versus depth: the distinct line facts (nearest piece in each direction, its colour and
distance) are few, so the inner width of a line block can be half the trunk width. What grows with
board size is the number of facts built on other facts, so add blocks before channels: 4x32 for
7x7, 6x48 for 9x9, 8 to 10 x 64 for 11x11.

### Heads

- `policy_head: conv` (new default): 1x1 conv to 4(N-1) planes, flattened in `MoveEncoder` order.
  No other code changes; the move encoding stays the same.
- `policy_head: bilinear`: logit(from, to) = q_from . k_to + u_from + v_to. It judges the landing
  square directly, where captures happen. Same cost as `conv`; the second thing to try.
- `policy_head: dense`: the old head, kept for the baselines.
- `value_head: pooled` (new default): board-wide average and maximum of 16 planes plus one spatial
  plane, then two dense layers. `dense` is the old head.

### Input planes (`features.py`)

| Planes | What they give the network | Cost |
|---|---|---|
| `constant_planes`: ones, corners, throne | edges and special squares in the first layer | none |
| `mobility_planes`: squares each side can move to, squares the king can move to | Reach and Arrival, computed exactly by the move generator | a few microseconds per node in the engine |
| `region_plane`: squares connected to the king without crossing an attacker | the encirclement question, answered | a flood fill per node |
| `history_length`: piece planes of earlier positions | repetition and tempo | none |

The reach and region planes hand the network its hardest facts. In training they are built in
Python with numpy (`reach_planes`, `king_region_plane`, checked against the move generator in
`test_tafl_network.py`). A browser build must compute the same planes in the same order; the
`.json` file written by `convert_to_onnx.py` lists them.

## 4. Measured inference cost

Random weights, batch 1, one thread. "Browser" is `onnxruntime-web` 1.24 (WASM + SIMD) in headless
Chrome 141 on a 2.1 GHz Xeon core; a laptop is typically 1.5x to 2x faster, the ratios carry over.
Browser numbers vary by about 20 % between runs. (Benchmark script: `bench_arch.py` in the
session scratchpad; it builds the candidates directly as ONNX graphs.)

| Board | Network | Parameters | Million multiply-adds | Browser ms | Native ORT ms |
|---|---|---|---|---|---|
| 7x7 | current: 6x32 SE, dense heads | 243 k | 5.6 | 0.94 | 0.33 |
| 7x7 | 4 line blocks x 32, conv policy, pooled value | 23 k | 1.0 | 0.21 | 0.09 |
| 7x7 | 4 line3 blocks x 32 | 33 k | 1.5 | 0.29 | 0.13 |
| 7x7 | 6 line blocks x 32 | 33 k | 1.5 | 0.28 | 0.14 |
| 9x9 | current: 6x48, dense heads | 684 k | 20.7 | 2.17 | 0.72 |
| 9x9 | 6 line blocks x 48 + global pool | 85 k | 6.5 | 0.95 | 0.34 |
| 9x9 | 6 line3 blocks x 48 + global pool | 116 k | 9.1 | 1.20 | 0.47 |
| 9x9 | 6 line blocks x 64 + global pool | 146 k | 11.4 | 1.37 | 0.49 |
| 11x11 | current: 6x96, dense heads | 2,184 k | 122 | 10.1 | 2.74 |
| 11x11 | 6 line blocks x 64 + global pool | 171 k | 20 | 2.27 | 0.81 |
| 11x11 | 10 line blocks x 64 + global pool | 280 k | 33 | 3.35 | 1.26 |
| 11x11 | 6 line3 blocks x 64 + global pool | 227 k | 27 | 2.94 | 0.96 |
| 11x11 | transformer, 4 layers, width 64 | 150 k | 24 | 3.48 | 1.20 |

Leaf batching does not help on the CPU: at batch 8 the cost per position drops only 5 to 10 %.
It only pays with a GPU backend (WebGPU), where width is almost free and 8 to 16 leaves should be
evaluated per call.

Self-play runs the networks in PyTorch, not in ONNX Runtime, and PyTorch pays more per operation.
The line networks have about twice as many (small) operations as the old ones, so the self-play
gain is smaller than the browser gain, and in eager mode there is none at 9x9. Measured with
`benchmark_network.py --network-config ... --compile` (batch 1, one thread, `torch.compile` with
the C++ wrapper, as the self-play workers run it), on the shipped configs:

| Board | Baseline config | Line config | Self-play speed-up | Browser speed-up |
|---|---|---|---|---|
| 7x7 | 0.57 ms | `line_4x32`: 0.27 ms | 2.1x | 4.5x |
| 9x9 | 0.92 ms | `line_6x48_gpool`: 0.77 ms | 1.2x | 2.3x |
| 11x11 | 3.12 ms | `line_6x64_gpool`: 1.07 ms | 2.9x | 4.5x |

So the browser, which is the target, gains the most; training still gets faster, just less so.

## 5. The twelve configs (`configs/<game>/<name>.json`)

Each board has a baseline that reproduces the removed `train_<game>.py` defaults (the same training
settings, the same network as a `tafl_network` config), and three experiments that change only
the `network` section, so a comparison measures the network alone. Parameter counts are of the
shipped configs.

| Game | Config | Network | Parameters | Hypothesis |
|---|---|---|---|---|
| brandubh | `baseline` | 6x32 res + SE, dense heads, 4 planes | 243,911 | the current network |
| brandubh | `line_4x32` | 4 line blocks x 32, conv policy, pooled value, 7 planes | 27,707 | equal or better per node, 4x the nodes |
| brandubh | `line_4x32_mobility` | as above + reach planes, 10 planes | 28,571 | faster learning; if large, 3 blocks are enough |
| brandubh | `line3_4x32` | 4 line3 blocks x 32, 7 planes | 37,051 | tells whether diagonal shapes matter at 7x7 |
| tablut | `baseline` | 6x48 res, dense heads, 4 planes | 685,079 | the current network |
| tablut | `line_6x48_gpool` | 6 line blocks x 48, global pool, 7 planes | 93,707 | 2.3x faster at equal depth |
| tablut | `line3_6x48_gpool` | 6 line3 blocks x 48, global pool, 7 planes | 125,099 | my guess for the eventual winner at 9x9 |
| tablut | `line_6x48_gpool_mobility` | line_6x48_gpool + reach planes, 10 planes | 95,003 | the engine-computed planes |
| hnefatafl | `baseline` | 6x96 res, dense heads, 4 planes | 2,185,567 | the current network |
| hnefatafl | `line_6x64_gpool` | 6 line blocks x 64, global pool, 7 planes | 183,883 | 4.5x faster; the base for the others |
| hnefatafl | `line_10x64_gpool` | 10 line blocks x 64, global pool, 7 planes | 294,603 | depth for encirclement; expected to beat 6x96 at equal cost |
| hnefatafl | `line_6x64_gpool_mobility_region` | line_6x64_gpool + reach planes + king-region plane, 11 planes | 186,187 | the encirclement question answered by the engine |

Not in the configs, but one `--set` away: `network.policy_head=bilinear`,
`network.tie_blocks=true`, `network.features.history_length=4`, `network.blocks=["thick",...]`.

Note on the hnefatafl baseline: it keeps the old `value_loss_weight` of 40 and no simulation
balancing, because that is what the old script did. The experiments keep them too, for a fair
comparison; they are probably not good settings.

## 6. How to run and compare

```bash
python run_training.py configs/brandubh/line_4x32.json
python run_training.py configs/tablut/line_6x48_gpool.json --num-workers 64
python run_training.py configs/hnefatafl/baseline.json --set self_play.num_iterations=200
python run_training.py configs/brandubh/baseline.json --show        # print the resolved config
python run_training.py configs/brandubh/baseline.json --resume checkpoints_brandubh/baseline/checkpoint_iter_40.pth
```

The resolved config is written to `<checkpoint_dir>/run_config.json` and stored in every
checkpoint (`run_config`, `network_config`, and the flat `config` that older tools read).

Comparing networks:

1. Train the baseline and one experiment for the same wall-clock time (the line networks also play
   self-play faster, so they see more games).
2. Compare at a fixed time per move, not a fixed number of simulations:
   `python play.py A/best_model.pth B/best_model.pth --time-per-move 0.5 --num-games 100 --num-workers 16`.
   Different architectures can play each other; each network is rebuilt from its own checkpoint.
3. Watch the policy loss floor: the conv head should reach a lower floor sooner than the dense head.
4. A search-free probe: collect positions where the mate-in-1 scan found a win and positions with
   a capture available, and report the raw policy's top-1 accuracy. That separates what the network
   sees from what the search finds.
5. 11x11: depth (`line_10x64_gpool`) against width (`--set network.channels=96`) at equal cost.

Timing a config natively: `python benchmark_network.py --game tablut --network-config configs/tablut/line_6x48_gpool.json`.

## 7. The browser side

`python convert_to_onnx.py <checkpoint>` writes `model.onnx` and `model.json`. The JSON lists the
board size, the input planes in order, the policy index formula and the rules the network was
trained with. The planes are (see `features.py`):

```
[attackers, defenders, king] x history_length positions (newest first, zero before the game start)
[player]                                            0 attackers, 1 defenders
[ones, corners, throne]                             if constant_planes (throne plane empty when disabled)
[reach_attackers, reach_defenders, reach_king]      if mobility_planes (destination squares of the
                                                     pseudo-legal moves; defenders include the king)
[king_region]                                       if region_plane (flood fill from the king through
                                                     non-attacker squares)
```

The policy output keeps the old index: `from_square * 4(N-1) + direction * (N-1) + distance - 1`,
directions 0 up, 1 down, 2 left, 3 right; take the softmax over the legal moves only.

## 8. Code map

| File | Role |
|---|---|
| `network_config.py` | `NetworkConfig`: the architecture options, validation, JSON form (no torch) |
| `features.py` | `FeatureSpec` and the input planes; `build_features` is used by self-play, search and training |
| `tafl_network.py` | the blocks, heads, `TaflNet`, and `build_network` / `load_network` / `load_weights` for checkpoints of any format |
| `games.py` | registry: game name -> game class, board size, move encoder, legacy network class |
| `run_config.py` | `TrainingConfig` and the JSON run config: sections, `--set` overrides, typo detection |
| `run_training.py` | the entry point: `python run_training.py <config.json> [--set ...]` |
| `train.py` | the pipeline; workers receive the network config and build the network from it |
| `configs/` | the twelve run configs |
| `test_tafl_network.py` | planes vs move generator, head index order, every block kind through ONNX, checkpoint loading, configs, a short end-to-end run |
| `network.py`, `network_tablut.py`, `network_hnefatafl.py` | the old fixed classes; still loaded for old checkpoints (`arch: legacy`) |
