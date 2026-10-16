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
  one layer. It is 2N-1 long and centred, so that every square, also a corner, sees all of its row
  and column (a kernel only N long reaches N//2 squares to each side).
- The old dense policy head (one big linear layer from the whole board to every move) holds half
  to two thirds of all parameters and has to learn every from-square separately. A 1x1 convolution
  that outputs one plane per (direction, distance) gives the same output in the same index order
  with a few hundred parameters.
- Measured in the browser (WASM, one thread): the new networks evaluate 4.5x (7x7), 2.3x (9x9) and
  4.5x (11x11) faster than the current ones at the same depth, with 5 to 12 times fewer parameters.
  These numbers are for line kernels only N long. The 2N-1 kernels now used cost about 1.5x more
  per line network in PyTorch (section 4), so the speed advantage in self-play is now 1.2x to 1.8x.
  Playing strength is not measured yet; that is what the configs in section 5 are for.
- Dilated kernels, 5x5 kernels, depthwise convolutions and squeeze-excitation were considered and
  rejected for this setup (section 3). A pure transformer was rejected too, but an `attention`
  block exists for hybrids: one Hnefatafl config replaces two of its ten line blocks with it.

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
| `line`: 1x1 down, 1x(2N-1) + (2N-1)x1, 1x1 up | whole row and column | the main bet; line facts are few, so the narrow inner width is enough | 0.37 to 0.43 |
| `line3`: like `line` with a 3x3 first | local shape, then lines over it | best "complete" block; use on 9x9 and 11x11 if diagonal shapes matter | 0.48 to 0.56 |
| `thick`: 3x(2N-1) + (2N-1)x3 | row, column and the neighbouring rows and columns | merges two thin line layers into one, but loses the nonlinearity between them; same cost as `res` | 0.76 to 1.16 |
| 5x5 | 24 neighbours | still local, +1 reach, 1.7x to 1.9x the cost | rejected |
| dilated 3x3 | taps two squares apart | skips squares, so it cannot check "nothing in between" | rejected |
| depthwise separable | 8 neighbours, 4x to 8x fewer multiply-adds | only 30 % faster in practice, worse per multiply-add than line blocks | rejected |
| transformer (attention over squares) | any square | right for a large GPU network (Leela Chess Zero); at this size and batch 1 it is 2x to 6x slower per evaluation than a line network with the same parameters, and it has to learn the line structure that the line kernel has built in | rejected for the browser |
| `attention`: self-attention over the squares with a learned bias per (row, column) offset, then a feed-forward layer | any square, also diagonally and far away | a few of them among line blocks: whole-board shapes such as a ring of attackers in one step; written out by hand so it exports to ONNX opset 12 | two of them cost ~1.6 ms more than two line blocks (11x11, PyTorch) |

The browser costs in this table were measured with line kernels only N long; the 2N-1 kernels make
`line`, `line3` and `thick` dearer (section 4).

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
| `mobility_planes`: squares each side can move to, squares the king can move to | Reach and Arrival, computed exactly by the move generator | 0.17 ms per position at 11x11 |
| `region_plane`: squares connected to the king without crossing an attacker | the encirclement question, answered | 0.15 ms per position at 11x11 |
| `history_length`: piece planes of earlier positions | repetition and tempo | none |

The reach and region planes hand the network its hardest facts. In training they are built in
Python (`reach_planes` with numpy: the nearest piece in each direction of every square;
`king_region_plane` with `scipy.ndimage.label`), checked against the move generator and the
encirclement test in `test_tafl_network.py`. For scale: one network call of `line_10x64_gpool`
takes 6.6 ms and generating the legal moves of a Hnefatafl position 0.9 ms (one core of owl31). A browser build must compute the same planes in the same order; the
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
the C++ wrapper, as the self-play workers run it), on the shipped configs, with the 2N-1 line
kernels, on one core of an idle owl31 (AMD EPYC 7551):

| Board | Baseline config | Line config | Self-play speed-up |
|---|---|---|---|
| 7x7 | 6x32 res + SE: 1.36 ms | `line_4x32`: 0.75 ms | 1.8x |
| 9x9 | 6x48 res: 2.64 ms | `line_6x48_gpool`: 2.07 ms | 1.3x |
| 11x11 | 8x64 res: 8.06 ms | `line_10x64_gpool`: 6.62 ms | 1.2x |

The first version had line kernels only N long (a corner square then saw only half of its edge
line in one block). On owl31 that version of `line_10x64_gpool` took 4.3 ms, so the full-line
kernels cost about 1.5x. Half of their work multiplies the zero padding; a matrix-product form
without padding was only 11 % faster in a test, not worth the extra code. The browser gains more
than self-play, as before, but the browser numbers above need measuring again.

## 5. The configs (`configs/<game>/<name>.json`)

Each board has a baseline and experiments that change only the `network` section, so a comparison
measures the network alone (`test_tafl_network.py` checks this). The Brandubh and Tablut baselines
reproduce the removed `train_<game>.py` defaults, except an evaluation temperature of 1.0 instead of
0.65 (0.65 repeated the same few openings). The Hnefatafl baseline is new: the old script's
settings had never been tuned (see the note below the table). Parameter counts are of the shipped
configs.

| Game | Config | Network | Parameters | Hypothesis |
|---|---|---|---|---|
| brandubh | `baseline` | 6x32 res + SE, dense heads, 4 planes | 243,911 | the current network |
| brandubh | `line_4x32` | 4 line blocks x 32, conv policy, pooled value, 7 planes | 39,995 | equal or better per node, 1.8x the nodes |
| brandubh | `line_4x32_mobility` | as above + reach planes, 10 planes | 40,859 | faster learning; if large, 3 blocks are enough |
| brandubh | `line3_4x32` | 4 line3 blocks x 32, 7 planes | 49,339 | tells whether diagonal shapes matter at 7x7 |
| tablut | `baseline` | 6x48 res, dense heads, 4 planes | 685,079 | the current network |
| tablut | `line_6x48_gpool` | 6 line blocks x 48, global pool, 7 planes | 149,003 | 1.3x faster at equal depth |
| tablut | `line3_6x48_gpool` | 6 line3 blocks x 48, global pool, 7 planes | 180,395 | my guess for the eventual winner at 9x9 |
| tablut | `line_6x48_gpool_mobility` | line_6x48_gpool + reach planes, 10 planes | 150,299 | the engine-computed planes |
| hnefatafl | `baseline` | 8x64 res, dense heads, 4 planes | 1,786,367 | the reference: an old-style network with the same settings |
| hnefatafl | `line_10x64_gpool` | 10 line blocks x 64, global pool, 7 planes | 509,323 | the main run; depth for encirclement, 1.2x faster than the baseline |
| hnefatafl | `line_10x64_gpool_mobility_region` | line_10x64_gpool + reach planes + king-region plane, 11 planes | 511,627 | the encirclement question answered by the engine (+0.3 ms per node) |
| hnefatafl | `line_10x64_gpool_attention` | line_10x64_gpool with blocks 4 and 8 as attention blocks, 7 planes | 485,075 | whole-board shapes learned instead of computed; 8.2 ms per call against 6.6 ms |

Not in the configs, but one `--set` away: `network.policy_head=bilinear`,
`network.tie_blocks=true`, `network.features.history_length=4`, `network.blocks=["thick",...]`,
and the cheaper `network.blocks` of 6 line blocks.

Note on the Hnefatafl settings: the old `train_hnefatafl.py` defaults (still in the git history)
were never tuned, and some would have crippled any network: value targets discounted by 0.98 per
ply (0.98^150 = 0.05, so positions far from the end got targets near zero) while the search did not
discount, a value-loss weight of 40, no simulation balancing, and a 40 M-sample replay buffer
(about 100 GB, since every stored position keeps a 4,840-entry policy). The configs now use the
settings that worked for Brandubh and Tablut, scaled to the longer games: discount 0.995 in both
the targets and the search (0.995^150 = 0.47), c_puct 2.0 and FPU 0 in self-play (1.5 and -0.25
in evaluation), Dirichlet alpha 0.1 (about three times as many legal moves as Brandubh), the 20-80
simulation balancer, random openings, a draw after 300 plies, value-loss weight 1.5, weight decay
0.01, a 12 M-sample buffer (1.5 M positions, about 40 iterations), and 256 games per iteration.
With 128 workers, 128 games would mean that about half of the started games are killed at the
cut-off (mostly long ones, which tilts the data towards short games); 256 games bring that down to
about a third.

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
| `configs/` | the run configs (section 5); `run_hnefatafl.sh <name>` starts a Hnefatafl one |
| `test_tafl_network.py` | planes vs move generator, head index order, every block kind through ONNX, checkpoint loading, configs, a short end-to-end run |
| `network.py`, `network_tablut.py`, `network_hnefatafl.py` | the old fixed classes; still loaded for old checkpoints (`arch: legacy`) |
