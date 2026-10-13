# Brandubh AlphaZero: runtime and training reference from the CPU cluster

This document summarises how the Brandubh (7x7 Tafl) AlphaZero training code performed on CPU
cluster nodes (owl31-owl35: 2x AMD EPYC 7551, 64 cores / 128 hardware threads, 256 GB RAM each).
It is meant as a baseline for porting and tuning the code on a desktop with one consumer GPU
(16 GB VRAM), a consumer CPU and 64 GB RAM.

All numbers come from the training logs of 21 Brandubh runs (2,904 iterations, Nov 13 - Dec 20, 2025)
and from the configs stored inside their checkpoints. A few benchmarks were re-run in Oct 2026 on a
**newer** node (owl39, AMD EPYC 9965). Those benchmarks show what the code does on modern cores;
they are not the training hardware. None of the source files (logs, checkpoints, benchmark notes)
are in git, so this document repeats everything that matters.

---

## 0. Conventions

- **Ply**: one move by one side. The logs say "moves" but mean plies.
- **Searched ply**: a ply where MCTS ran. Each searched ply becomes one training position.
  Random opening plies and the final "mate-in-1" plies are not searched.
- **Simulation**: one MCTS playout. It costs one network call unless it ends in a terminal node.
- **Worker**: one single-threaded Python process that plays whole games.
- **Iteration**: 512 self-play games, then training, then (every 4th iteration) a gating match.
- Numbers are **medians over iterations 6 and later**. Brackets show the 10th-90th percentile where given.
- Run names: `SE/...` = `checkpoints_brandubh_SE/` (squeeze-excitation network, the newest runs),
  `new/...` = `checkpoints_brandubh_new/`, `old/...` = `checkpoints_brandubh/`.
  The first word of a run name is a **network-size label**, not a machine name:
  mosquito < gnome < jack < wizard < fatman, later also grus (6x48) and littleboy (6x32).
- "Net 6x48, vh64, 4pl" means 6 residual blocks, 48 channels, value-head hidden size 64,
  4 input planes (13 planes = 4 history steps).

---

## 1. Key numbers

These are for the current-generation runs (`SE/` and `new/`): 1000+1000 simulations per ply in
self-play (split dynamically between attacker and defender), 128 workers, 0.24-0.58 M parameters.

| Quantity | Value on the CPU node |
|---|---|
| Hardware per run | one owl31-owl35 node: 2x AMD EPYC 7551 (2017), 64 cores / 128 hardware threads, 256 GB RAM |
| Worker processes per run | 128 single-threaded workers, one per hardware thread (so 2 per physical core) |
| Self-play work per iteration | 512 games, ~22-42 plies/game, ~11-21 k searched plies, ~11-21 M simulations |
| Self-play wall time per iteration | 620-1,800 s (run medians); grows during a run as games get longer |
| Games per second (whole run) | 0.29-0.82 |
| Searched plies per second (whole run) | 10-17 |
| MCTS simulations per second (whole run) | 10,100-16,600 (79-129 per worker) |
| Wall time per simulation, per worker | 6.3-9.2 ms, of which the network call is 3.8-6.3 ms (59-72 %) |
| Same code on a newer CPU (owl39, EPYC 9965), 1 process | 1.4-1.6 ms per simulation (network call 0.8-1.1 ms) |
| Same code on owl39, 16 processes on 16 cores | 2.0 ms per simulation (network call 1.5 ms) |
| Batch-1 forward pass on owl39, 1 thread | 0.40-0.87 ms with `torch.compile`, 0.71-1.31 ms eager |
| Training volume per iteration | 512,000 samples forward+backward (+128,000 validation forward) |
| Training wall time per iteration | 145-330 s while buffer < 0.5 M samples; 550-1,420 s with full 6 M buffer (batch-64 run: 530 s and 2,600 s) |
| Training speed | 5-7 steps/s at batch 512 with small buffer; 1.0-1.3 steps/s with 6 M buffer |
| Pure fwd+bwd+AdamW speed on owl39 (eager, batch 512) | 1,200-3,600 samples/s on 1 thread; 8,600-18,400 samples/s on 16 threads |
| Gating match (128 games, 800 sims) | 760-3,300 s; about 2x longer when a game hits the 500-ply limit |
| Wall time per iteration | 1,700-3,700 s; 21-42 iterations/day (97/day for the no-augmentation run) |
| Longest run | `SE/jack_history_and_decay`: 365 iterations in 12.8 days, 186,880 games, 5.1 M searched plies, ~5.1 G simulations |
| Replay buffer in RAM | 6 M samples = 35 GB (4 planes) or ~45 GB (13 planes) incl. Python overhead; main process peaks at ~70-90 GB during training |
| End-of-run losses | policy cross-entropy 2.3-2.4; value MSE 0.11-0.15 (runs with 0.98 reward discount) or 0.31-0.35 (no discount) |

The two most important observations:
- **Per-worker MCTS speed on the cluster was low.** It was 3-6x lower than the same code achieves
  on one modern core. Two things explain this: the cluster CPUs are from 2017, and two workers
  shared each physical core.
- **Training time was dominated by costs that grow with replay-buffer size, not by backpropagation.**

Sections 4 and 5 give the details.

---

## 2. Hardware and software

- **Training nodes: owl31-owl35**, one run per node. Runs that overlapped in time (up to 5)
  ran on different nodes, so they did not compete for CPU or RAM.
  Each node: 2x AMD EPYC 7551 (Zen 1 "Naples", 2017; 32 cores per socket, 2.0 GHz base, up to
  3.0 GHz boost), 2 sockets x 32 cores = 64 physical cores, SMT on = 128 hardware threads,
  256 GB RAM (251 GB usable). Checked with `lscpu` on all five nodes.
  The CPU has AVX2 but no AVX-512, and it executes 256-bit AVX2 instructions as two 128-bit halves.
  The logs do not name the node. The node type is confirmed by the user and fits all evidence:
  - The worker count defaults to `mp.cpu_count()`, which is 128 on these nodes.
  - TorchInductor compiled AVX2 code (`-D CPU_CAPABILITY_AVX2` in the logs); on an AVX-512 node
    it picks AVX-512.
  - One run needs ~70-90 GB at peak, which fits in 256 GB (5.5).
- **Workers per run: 128**, one per hardware thread, so two workers share each physical core.
- **Benchmark node (not used for training): owl39**, 2x AMD EPYC 9965 (Zen 5c, 2024; 384 cores,
  SMT off, up to 3.7 GHz, AVX-512), 1.5 TB RAM. Sections 4.6 and 5.4 were measured here, in a
  16-core allocation on a shared node, with the same venv.
- **Device**: everything ran on CPU. `TrainingConfig.device` defaults to `"cpu"`; the
  "auto-detect" comment in `train_brandubh.py` is wrong.
- **Software**: Python 3.10.2, PyTorch 2.9.0 (CUDA 12.8 wheel, but used on CPU), NumPy 2.2.6.
  The venv lived outside this repo (`../RL_tictactoe/.RL_venv`).
- **Temp dir**: `/dev/shm` (RAM disk) for the per-game network file and per-process Inductor
  caches. Earlier runs used `/tmp` and sometimes crashed with `CppCompileError`, because a cached
  Inductor header under `/tmp` had disappeared.
- **Process start method**: default `fork` (Linux, Python 3.10). Workers are forked from the main
  process, which holds the replay buffer.
- `torch.set_flush_denormal(True)` was on in all runs analysed here (see 4.7 for why).

---

## 3. How the CPU pipeline spends its time

This describes the code as it was during these runs (git HEAD `8b4e4fb`, Dec 4 2025; `train.py`,
`mcts.py`, `network.py` are unchanged since). The numbers below only make sense with this structure in mind.

**One iteration** (`train.py: train()`):
1. Recreate a `multiprocessing.Pool(128)` (fork from the main process, which holds the replay buffer).
2. **Self-play** (`generate_self_play_data`): `imap_unordered` over up to 1,024 game tasks.
   Each task is one full game in one worker. When 512 games have finished, the pool is
   **terminated**; games still in progress are thrown away.
3. Main process converts each game into samples: value targets, then **9 copies per position**
   (8 board symmetries, with the identity stored twice), appended to a `deque` replay buffer.
4. **Training** (`train_network`) in the main process (torch threads not set, so PyTorch default).
5. **Gating** every 4 iterations: 128 games new-vs-best, 800 sims. The result only decides which
   network is saved as `best_model.pth`. **Self-play always uses the latest network**, so gating
   does not steer training.
6. Save a checkpoint (every iteration) and the JSON history.

**Inside one self-play game** (`_play_self_play_game_worker`):
- `torch.set_num_threads(1)`; load weights from a temp `.pth`; `torch.compile(mode='default')`
  for every game (first call compiles: ~20 s with a cold cache, ~2-4 s warm).
- Before each search: scan every legal move for an immediate win (clone + make_move each).
  If one exists, store a one-hot policy and skip MCTS.
- MCTS (`mcts.py`): a new tree for every ply (no tree reuse), one **batch-1** network call per
  simulation, PUCT selection, lazy child creation (clone + make_move in selection), Dirichlet
  noise at the root, legal moves generated about 3 times per expanded node
  (mask, legal-prob extraction, expansion).
- Timers: the log's "MCTS timing breakdown" sums over all *finished* games (worker-seconds).
  "Network eval" **contains** "Get legal moves" (nested timer), so the printed total counts that
  time twice. Compile time sits inside "Network eval". All per-simulation numbers in this
  document subtract the nested legal-move time.

**Training step** (`train_network`), per iteration:
- Copy the whole `deque` into new NumPy arrays (Python loop over up to 6 M samples).
- `torch.compile(network, dynamic=True)` again every iteration.
- 10 "epochs" x N batches; each batch index set is drawn with
  `np.random.choice(total, batch, replace=False)` (cost grows with buffer size).
- Per batch: compute an L2 sum and per-parameter gradient norms with `.item()` calls (logging only).
- After each epoch: a "validation" pass over 25 % extra samples drawn from the **same** buffer.

**Gating match**: two `pool.map` calls of 64 games each, one after the other. So only 64 of 128
workers are busy, and each half waits for its slowest game. The game limit here is 500 plies
(self-play uses 100).

---

## 4. Self-play performance

### 4.1 Work per iteration

- 512 games x ~22-42 plies (grows during training) x 1000 simulations per searched ply
  = **about 11-21 million simulations (network calls) per iteration**.
- Samples added per iteration: ~100-190 k (= searched plies x 9).
- The 6 M buffer therefore holds the last ~30-55 iterations of data.

### 4.2 Throughput per run

`Sims/ply` is the mean of the attacker and defender budgets. `Worker utilisation` is MCTS
worker-time divided by (self-play wall time x 128).

| Run | Net (blocks x ch, value head, input planes) | Params | Sims/ply (A+D)/2 | Self-play s/iter | Games/s | Searched plies/s | Simulations/s (run) | Sims/s per worker | ms per simulation (worker) | ms per network call (worker) | Worker utilisation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SE/fatman_decay | 6x64, vh256, 4pl | 582,658 | 1000 | 1289 | 0.40 | 10.4 | 10,445 | 82 | 8.9 | 6.3 | 73% |
| SE/grus_default | 6x48, vh64, 4pl | 375,478 | 1000 | 1783 | 0.29 | 11.0 | 10,999 | 86 | 8.5 | 5.6 | 74% |
| SE/grus_high-CPUCT-zero-rel-FPU | 6x48, vh64, 4pl | 375,478 | 1000 | 1092 | 0.47 | 11.8 | 11,841 | 93 | 8.6 | 5.0 | 81% |
| SE/grus_no-data-aug | 6x48, vh64, 4pl | 375,478 | 1000 | 621 | 0.82 | 16.6 | 16,556 | 129 | 7.2 | 4.3 | 93% |
| SE/jack_highexplore | 4x64, vh64, 4pl | 422,706 | 1000 | 1663 | 0.31 | 10.1 | 10,138 | 79 | 9.2 | 6.1 | 73% |
| SE/jack_history_and_decay | 4x64, vh256, 13pl | 437,682 | 1000 | 1409 | 0.36 | 10.5 | 10,463 | 82 | 8.8 | 5.8 | 72% |
| SE/jack_history_and_decay_smallbatch | 4x64, vh256, 13pl | 437,682 | 1000 | 893 | 0.57 | 13.3 | 13,305 | 104 | 6.9 | 4.5 | 72% |
| SE/littleboy_decay | 6x32, vh256, 4pl | 243,946 | 1000 | 871 | 0.59 | 14.6 | 14,559 | 114 | 6.3 | 3.8 | 72% |
| new/grus_default | 6x48, vh64, 4pl | 372,274 | 1000 | 1030 | 0.50 | 13.8 | 13,766 | 108 | 6.7 | 4.3 | 73% |
| new/grus_kingtemp | 6x48, vh64, 4pl | 372,274 | 1000 | 848 | 0.60 | 13.5 | 13,467 | 105 | 7.0 | 4.5 | 72% |
| old/fatman_nothrone | 6x96, vh64, 4pl | 1,122,514 | 600 | 2503 | 0.20 | 6.1 | 3,667 | 29 | 11.7 | 9.5 | 33% |
| old/fatman_nothrone_lossfixed | 6x96, vh64, 4pl | 1,122,514 | 1100 | 2626 | 0.19 | 7.1 | 7,787 | 61 | 8.2 | 6.4 | 50% |
| old/grus_nothrone | 6x48, vh64, 4pl | 372,274 | 1100 | 1444 | 0.35 | 12.4 | 13,641 | 107 | 5.0 | 3.0 | 53% |
| old/grus_nothrone_highcpuct | 6x48, vh64, 4pl | 372,274 | 1000 | 1313 | 0.39 | 13.3 | 13,253 | 104 | 7.0 | 4.3 | 73% |
| old/grus_nothrone_kingtemp | 6x48, vh64, 4pl | 372,274 | 800 | 684 | 0.75 | 17.3 | 13,849 | 108 | 7.0 | 4.5 | 76% |
| old/grus_nothrone_lessrng | 6x48, vh64, 4pl | 372,274 | 800 | 737 | 0.69 | 17.6 | 14,088 | 110 | 6.8 | 4.4 | 75% |
| old/grus_nothrone_lowvalweight | 6x48, vh64, 4pl | 372,274 | 1000 | 1461 | 0.35 | 13.4 | 13,435 | 105 | 7.0 | 4.3 | 74% |
| old/grus_nothrone_relfpu_dynsim | 6x48, vh64, 4pl | 372,274 | 1200 | 1856 | 0.28 | 11.1 | 13,269 | 104 | 7.0 | 4.3 | 72% |
| old/grus_nothrone_rndplay | 6x48, vh64, 4pl | 372,274 | 1000 | 1783 | 0.29 | 12.8 | 12,751 | 100 | 7.5 | 4.6 | 75% |
| old/jack_nothrone | 4x64, vh64, 4pl | 418,834 | 300 | 1043 | 0.49 | 17.0 | 5,094 | 40 | 6.2 | 3.9 | 25% |
| old/jack_nothrone_lossfixed | 4x64, vh64, 4pl | 418,834 | 550 | 1185 | 0.54 | 23.0 | 12,629 | 99 | 3.9 | 2.4 | 43% |

Notes:
- Simulations are estimated as searched plies x mean budget. Simulations that end in a terminal
  node make no network call, so the true number of network calls is a bit lower.
- At the same training stage (iterations 6-70), the SE version of the 6x48 network costs ~10 %
  more per network call than the plain version (4.6-4.8 vs 4.2-4.4 ms). The run medians in the
  table differ more, because the SE runs lasted longer and slowed down over time (4.5).
  Width matters more than SE blocks: 6x32 is the fastest, 6x64 and 6x96 the slowest.
- History planes (13 instead of 4 input planes) cost almost nothing per call.
- Utilisation was low in the oldest runs (25-53 %); the cause is not visible in the logs.

### 4.3 Where MCTS time goes

Share of worker MCTS time, with the nested legal-move time moved out of "network call":

| Run | Network call | Selection (incl. lazy child make_move) | Expansion | Legal-move list + mask | Backup |
|---|---|---|---|---|---|
| SE/fatman_decay | 72% | 11% | 7% | 10% | 0.2% |
| SE/grus_default | 65% | 14% | 8% | 12% | 0.3% |
| SE/grus_high-CPUCT-zero-rel-FPU | 59% | 24% | 7% | 11% | 0.2% |
| SE/grus_no-data-aug | 60% | 24% | 6% | 10% | 0.2% |
| SE/jack_highexplore | 67% | 13% | 8% | 12% | 0.3% |
| SE/jack_history_and_decay | 66% | 15% | 7% | 11% | 0.3% |
| SE/jack_history_and_decay_smallbatch | 65% | 15% | 7% | 12% | 0.3% |
| SE/littleboy_decay | 61% | 16% | 9% | 14% | 0.3% |
| new/grus_default | 63% | 15% | 8% | 13% | 0.3% |
| new/grus_kingtemp | 64% | 15% | 8% | 12% | 0.3% |

So about one third of MCTS time is pure Python tree and game logic. Game-state copying at the
root is negligible (< 0.01 %).

### 4.4 Worker utilisation (72-81 % in the current runs)

About a quarter of worker time is lost outside MCTS. Known contributors:
- **Killed games**: each worker completes ~4 games per iteration. When game 512 finishes, ~127
  games are in progress and are discarded. Long games are more likely to be cut off. Expected loss: ~13-15 %.
- Per-game setup outside the timers: `torch.load`, building the MCTS objects, the mate-in-1 scan
  before each ply, `gc.collect()`, sending results back.
- Main-process work inside the self-play timer: augmentation costs 1.5 ms per position
  (measured), i.e. ~15-30 s per iteration.

The run without augmentation (`SE/grus_no-data-aug`) reached 93 %. Its buffer was also
small (0.75 M), and the gap is not fully explained by the items above. Utilisation fell
only slightly as the buffer grew (e.g. 0.76 -> 0.72 in `SE/jack_highexplore`).

### 4.5 Slowdown during a run

Cost per simulation rose 20-65 % over each long run, in both network time and tree time:

| Run | ms/simulation, iterations 1-25 | ms/simulation, last 25 iterations |
|---|---|---|
| SE/jack_history_and_decay | 6.9 | 10.1 |
| SE/jack_highexplore | 6.8 | 11.1 |
| SE/littleboy_decay | 5.7 | 6.9 |
| SE/fatman_decay | 8.0 | 10.6 |
| SE/grus_high-CPUCT-zero-rel-FPU | 7.6 | 10.3 |

It happens together with longer games and more draws. Each run had its own node, and per-call time
did not depend on how many runs were active. The cause is unknown. Self-play wall time per iteration
rose even more (e.g. 750 -> 1,700 s), because games also got longer.

### 4.6 Same code on a newer CPU (owl39, Oct 2026)

These tests ran on `owl39.uio.no` (2x EPYC 9965, see Section 2), **not** on the training nodes. They
used a 16-core allocation on a shared node (load ~90 of 384 cores) and the training venv (torch
2.9.0). They show what the code does on modern cores, one worker per core, without the training
pipeline around it. Here Inductor compiled AVX-512 code; on the training nodes it compiled AVX2.

**Batch-1 forward pass, 1 thread, after warm-up:**

| Net | Params | Eager | `torch.compile` | First compiled call (compile) |
|---|---|---|---|---|
| littleboy 6x32, vh256, 4pl | 243,946 | 0.71 ms | 0.40 ms | 21.8 s (cold cache) |
| grus 6x48, vh64, 4pl | 375,478 | 0.96 ms | 0.59 ms | 3.5 s (warm cache) |
| jack 4x64, vh64, 4pl | 422,706 | 0.89 ms | 0.61 ms | 2.7 s |
| jack 4x64, vh256, 13pl | 437,682 | 0.91 ms | 0.62 ms | 2.2 s |
| fatman 6x64, vh256, 4pl | 582,658 | 1.31 ms | 0.87 ms | 1.8 s |

**One real self-play game** (the actual worker function, best checkpoint, 1000+1000 sims):

| Setup | ms per simulation | Network call | Legal moves + mask | Selection | Expansion | Backup |
|---|---|---|---|---|---|---|
| littleboy, 1 process | 1.38 | 0.83 | 0.17 | 0.23 | 0.14 | 0.003 |
| jack 4x64 history-4, 1 process | 1.61 | 1.10 | 0.15 | 0.23 | 0.12 | 0.003 |
| littleboy, 16 processes on 16 cores | 1.98 | 1.50 | - | 0.19 | - | - |
| littleboy in training on owl31-35 (log median) | 6.3 | 3.8 | ~0.9 | ~1.0 | ~0.6 | ~0.02 |

Per-game work outside MCTS (load, compile with warm cache, mate-in-1 scans, cleanup) was ~2.6-3.6 s.

Even in isolation, the network-call step costs about twice the bare forward pass (0.83 vs 0.40 ms
for littleboy). The extra time goes to building the input tensor, converting outputs to NumPy and
the amortised per-game compile.

In training, littleboy took 6.3 ms per simulation: 3.2x slower than 16 parallel processes on owl39,
and 4.6x slower than a single process there. The training hardware explains most of this gap:
- Two workers shared each physical core (128 workers on 64 SMT cores). Running 16 processes in
  parallel on owl39 already cost 1.4x compared with a single process.
- Zen 1 cores from 2017 are much slower per clock than Zen 5c, run at a lower clock (2.0-3.0 vs
  up to 3.7 GHz), and execute AVX2 at half width.

So the per-worker cluster numbers reflect old hardware with two workers per core, not a limit of
the code on modern CPUs.

The Python tree logic alone costs ~0.5 ms per simulation on one owl39 core (~1,800 simulations/s
per core if network calls were free). It cost ~2.5 ms per simulation per worker on the training nodes.

### 4.7 Older benchmark notes in the repo (untracked)

- `BENCHMARK_RESULTS.md` (Nov 12, 2025; plain ResNet before SE blocks; eager; thread count not
  pinned): batch-1 forward 1.6-3.6 ms for 64-128 channels, 4-6 blocks
  (4x64: 1.9 ms, 524 calls/s; 6x128: 3.6 ms, 276 calls/s).
- `DENORMAL_FIX.md`: without `torch.set_flush_denormal(True)`, trained checkpoints got slower on CPU
  as weights decayed toward tiny values: 6x96 best model 4.8 ms vs 2.6 ms with flushing (1.8x).
  A scan of 56 checkpoints showed 2.2 ms early and 4.0-4.9 ms late in training. This is CPU-specific.
- `inference_benchmark_results.csv` for `old/jack_nothrone` (4x64), with flushing on: 1.86-1.93 ms
  for all 160 checkpoints, i.e. no drift.

---

## 5. Training (backpropagation) performance

### 5.1 Volume

- Main runs: `batch x batches_per_epoch x epochs` = 512 x 100 x 10 or 256 x 200 x 10
  = **512,000 samples forward+backward per iteration**, plus 128,000 validation forward passes.
  Exceptions: `SE/grus_no-data-aug` (512 x 12 x 10 = 61,440) and
  `SE/jack_history_and_decay_smallbatch` (64 x 800 x 10).
- New data per iteration: ~11-21 k positions (x 9 copies). With a full 6 M buffer, each stored
  sample is drawn with probability 512 k / 6 M = 0.085 per iteration. Over its ~30-55 iterations in
  the buffer that is ~3-4 draws per sample, or ~30 draws per position counting all 9 copies.
- Optimizer: AdamW (`fused=True`), weight decay 0.01, loss = policy cross-entropy + 1.5 x value MSE
  (current runs).

### 5.2 Measured speed

"Full buffer" means at least 95 % of the maximum buffer size. The fit column is a per-run
least-squares fit of training time against buffer size in millions of samples.

| Run | Batch x steps/iter | Samples/iter (fwd+bwd) | Train s/iter (median) | Steps/s, buffer < 0.5M | Samples/s, buffer < 0.5M | Steps/s, full buffer | Samples/s, full buffer | Fit: s + s per M buffered samples |
|---|---|---|---|---|---|---|---|---|
| SE/fatman_decay | 256 x 2000 | 512,000 | 1422 | 6.11 | 1,565 | 1.34 | 342 | 149 + 227/M |
| SE/grus_default | 512 x 1000 | 512,000 | 792 | 6.46 | 3,309 | 1.23 | 629 | 53 + 130/M |
| SE/grus_high-CPUCT-zero-rel-FPU | 512 x 1000 | 512,000 | 798 | 5.74 | 2,937 | 1.22 | 624 | 39 + 129/M |
| SE/grus_no-data-aug | 512 x 120 | 61,440 | 24 | 6.19 | 3,167 | 4.90 | 2,508 | 18 + 9/M |
| SE/jack_highexplore | 512 x 1000 | 512,000 | 813 | 5.42 | 2,773 | 1.21 | 619 | 68 + 127/M |
| SE/jack_history_and_decay | 512 x 1000 | 512,000 | 941 | 4.81 | 2,462 | 1.05 | 539 | 45 + 150/M |
| SE/jack_history_and_decay_smallbatch | 64 x 8000 | 512,000 | 2592 | 15.08 | 965 | 1.52 | 97 | -441 + 933/M (poor fit) |
| SE/littleboy_decay | 256 x 2000 | 512,000 | 1231 | 9.85 | 2,522 | 1.59 | 408 | 41 + 206/M |
| new/grus_default | 512 x 1000 | 512,000 | 558 | 6.65 | 3,405 | 1.28 | 654 | 59 + 120/M |
| new/grus_kingtemp | 512 x 1000 | 512,000 | 554 | 6.91 | 3,540 | 1.28 | 657 | 62 + 119/M |
| old/fatman_nothrone | 256 x 1000 | 256,000 | 346 | 8.18 | 2,094 | n/a | n/a | 178 + 61/M |
| old/fatman_nothrone_lossfixed | 512 x 1000 | 512,000 | 627 | n/a | n/a | 0.92 | 472 | 240 + 148/M |
| old/grus_nothrone | 512 x 1000 | 512,000 | 615 | n/a | n/a | 1.03 | 527 | 62 + 117/M |
| old/grus_nothrone_highcpuct | 512 x 1000 | 512,000 | 632 | 7.17 | 3,673 | 1.26 | 646 | 53 + 122/M |
| old/grus_nothrone_kingtemp | 512 x 1000 | 512,000 | 813 | 6.73 | 3,448 | 0.80 | 409 | 65 + 118/M |
| old/grus_nothrone_lessrng | 512 x 1000 | 512,000 | 730 | 6.50 | 3,328 | 0.85 | 438 | 77 + 110/M |
| old/grus_nothrone_lowvalweight | 512 x 1000 | 512,000 | 209 | 6.85 | 3,507 | n/a | n/a | 101 + 90/M |
| old/grus_nothrone_relfpu_dynsim | 512 x 1000 | 512,000 | 614 | 5.99 | 3,068 | 1.26 | 645 | 50 + 122/M |
| old/grus_nothrone_rndplay | 512 x 1000 | 512,000 | 719 | 6.55 | 3,353 | 1.19 | 611 | 40 + 129/M |
| old/jack_nothrone | 256 x 1000 | 256,000 | 638 | 8.46 | 2,167 | n/a | n/a | 123 + 104/M |
| old/jack_nothrone_lossfixed | 512 x 1000 | 512,000 | 965 | n/a | n/a | 0.69 | 354 | 110 + 133/M |

The "buffer < 0.5 M" columns include the per-iteration `torch.compile` and buffer copy, so they
are a lower bound on pure step speed. One iteration (`SE/jack_history_and_decay`, iteration 73)
took 68,142 s (19 h) of training time; it is excluded as an outlier.

### 5.3 Why training slowed down as the buffer filled

Training time grew linearly with buffer size, and the slope grows with the number of optimizer
steps: about **0.10-0.13 s extra per step per million buffered samples**. At 6 M samples that is
~0.7 s per step, while the step itself costs ~0.15 s at batch 512. So at full buffer, **~80 % of
training time was overhead that scales with buffer size, not backpropagation**.

Partial cause, measured on owl39: `np.random.choice(N, 512, replace=False)` shuffles all N indices
on each call. It costs ~16 ms per million samples (97 ms at N = 6 M), i.e. ~100 s per 1,000 steps.
On the older training CPUs it is probably 1.5-2x slower, so it explains roughly 15-30 % of the
slope. `np.random.default_rng().choice(...)` takes ~0.02 ms for the same draw. Gathering 512 random
rows costs little in comparison (< 2 ms at 2 M samples on owl39). The rest of the slope is not
explained by the logs. Candidates: per-batch logging loops with `.item()`, and random reads from a
35-45 GB array spread over the 8 NUMA nodes of a two-socket EPYC 7551 node. The main-process torch
thread count was not set in the code (PyTorch default). Each run had its own node, so other runs did
not compete with training.

Fixed costs per iteration (the ~40-70 s intercept): the Python copy of the `deque` into NumPy, and
`torch.compile(dynamic=True)` of the training graph.

### 5.4 Pure compute cost on a newer CPU (owl39, Oct 2026)

Measured on owl39, not on the training nodes; expect lower per-thread speed on owl31-35.
Synthetic data, eager mode, one optimizer step (forward, cross-entropy + 1.5 x MSE, backward, AdamW):

| Net | 1 thread, batch 256 | 1 thread, batch 512 | 16 threads, batch 256 | 16 threads, batch 512 |
|---|---|---|---|---|
| littleboy 6x32, vh256, 4pl | 79 ms (3,254 samples/s) | 144 ms (3,556/s) | 20 ms (12,664/s) | 28 ms (18,404/s) |
| grus 6x48, vh64, 4pl | 139 ms (1,839/s) | 272 ms (1,883/s) | 28 ms (9,289/s) | 42 ms (12,105/s) |
| jack 4x64, vh64, 4pl | 146 ms (1,754/s) | 294 ms (1,739/s) | 22 ms (11,593/s) | 42 ms (12,214/s) |
| jack 4x64, vh256, 13pl | 147 ms (1,738/s) | 296 ms (1,728/s) | 23 ms (10,922/s) | 42 ms (12,096/s) |
| fatman 6x64, vh256, 4pl | 213 ms (1,202/s) | 429 ms (1,194/s) | 35 ms (7,254/s) | 60 ms (8,583/s) |

At 16 threads on owl39, the 512,000 samples of one iteration take ~30-60 s of pure compute. The
training nodes spent 145-330 s with a small buffer (includes compile and buffer copy) and
550-1,420 s with a full buffer.

### 5.5 Memory and disk

- One stored sample = float32 state + float32 policy over 1,176 moves
  = 4 x (planes x 49 + 1,176) bytes = **5,488 B (4 planes)** or **7,252 B (13 planes)**.
  With the Python objects around it (one tuple, two array headers, one float), a sample in the
  `deque` costs **5,809 B** (measured, 4 planes), i.e. ~320 B overhead.
- 6 M-sample buffer: **35 GB** (4 planes) or **~45 GB** (13 planes).
- `train_network` builds a second full copy as NumPy arrays (33 / 44 GB), so the main process
  peaks at **~70-90 GB** during training. Each iteration also holds ~0.6-1.4 GB of new samples
  before merging.
- Self-play workers: one standalone worker peaks at 0.65 GB RSS (torch, compiled network, MCTS;
  measured on owl39). Forked workers share most pages with the main process, so 128 workers add at
  most ~80 GB. The buffer's tuples are tracked by Python's garbage collector, and every worker runs
  `gc.collect()` after each game. That probably forces private copies of the pages holding those
  tuples (up to ~0.5 GB per worker at 6 M samples; not measured). Even this worst case (~190 GB
  during self-play) fits in a 256 GB node. The workers are gone during training (the pool is terminated).
- So **one run fits on one 256 GB node**, which matches the one-run-per-node setup.
  It does **not** fit in 64 GB RAM.
- Checkpoints (weights + AdamW state + history): 3.1 MB (6x32), 5.4 MB (4x64, 13 planes),
  7.1 MB (6x64). One file per iteration: the `SE/` runs use 8.9 GB in total.

---

## 6. Evaluation cost

- **Gating match** (every 4 iterations; 128 games; 800 sims per side in the current runs; c_puct
  1.0-1.5 and FPU -0.5 or -0.25 depending on run; temperature 0.65 for the first 10 plies, then 0;
  500-ply limit; **played with the throne enabled by mistake**, see Appendix B item 3): median
  760-3,300 s per run.
  Evals where at least one game was a draw (500 plies) took about twice as long:

  | Run | Evals without draws | Evals with a draw |
  |---|---|---|
  | SE/jack_highexplore | 1,556 s (n=24) | 3,404 s (n=53) |
  | SE/grus_high-CPUCT-zero-rel-FPU | 1,418 s (n=35) | 3,260 s (n=25) |
  | SE/jack_history_and_decay | 1,635 s (n=89) | 3,453 s (n=2) |
  | SE/littleboy_decay | 1,154 s (n=56) | 2,513 s (n=3) |

  Only 64 of 128 workers are busy (two sequential halves), and each half waits for its longest
  game. Gating used 13-30 % of total wall time (Section 7), though it does not steer self-play.
- **Versus random player** (only early in old runs; 128 games): 50-1,270 s, median ~400 s.
  Networks reached 99-100 % win rate within 1-3 iterations, after which this eval was skipped.

---

## 7. Wall clock per iteration and per run

| Run | Dates (start -> end) | Iterations | Wall days | Iter/day | Median s/iter | Net-eval s (median) | Share of wall time: self-play / train / eval / other | Games | Searched plies | Simulations |
|---|---|---|---|---|---|---|---|---|---|---|
| SE/fatman_decay | 12-09 -> 12-16 | 201 | 7.4 | 27 | 3109 | 1648 | 41 / 41 / 13 / 5% | 102,912 | 2.65M | 2.65G |
| SE/grus_default | 11-24 -> 12-01 | 176 | 6.8 | 26 | 3035 | 3073 | 52 / 21 / 23 / 5% | 90,112 | 3.29M | 3.29G |
| SE/grus_high-CPUCT-zero-rel-FPU | 11-27 -> 12-04 | 241 | 6.9 | 35 | 2225 | 1957 | 43 / 28 / 22 / 6% | 123,392 | 3.07M | 3.07G |
| SE/grus_no-data-aug | 11-27 -> 11-28 | 143 | 1.5 | 97 | 698 | 759 | 71 / 3 / 24 / 3% | 73,216 | 1.49M | 1.49G |
| SE/jack_highexplore | 11-29 -> 12-11 | 308 | 11.8 | 26 | 2971 | 3280 | 50 / 23 / 21 / 6% | 157,696 | 4.99M | 4.99G |
| SE/jack_history_and_decay | 12-04 -> 12-16 | 365 | 12.8 | 28 | 2716 | 1583 | 47 / 31 / 15 / 8% | 186,880 | 5.13M | 5.13G |
| SE/jack_history_and_decay_smallbatch | 12-04 -> 12-07 | 70 | 3.3 | 21 | 3705 | 1028 | 22 / 69 / 7 / 3% | 35,840 | 0.84M | 0.84G |
| SE/littleboy_decay | 12-13 -> 12-20 | 237 | 6.7 | 36 | 2392 | 1159 | 35 / 46 / 13 / 7% | 121,344 | 2.94M | 2.94G |
| new/grus_default | 11-23 -> 11-25 | 74 | 2.0 | 36 | 1962 | 2990 | 46 / 21 / 28 / 4% | 37,888 | 1.09M | 1.09G |
| new/grus_kingtemp | 11-23 -> 11-25 | 84 | 2.0 | 42 | 1702 | 2871 | 40 / 24 / 30 / 5% | 43,008 | 0.96M | 0.96G |
| old/fatman_nothrone | undated | 70 | 2.9 (sum) | n/a | 3487 | 2717 | 72 / 9 / 19 / 0% | 35,840 | 1.06M | 0.64G |
| old/fatman_nothrone_lossfixed | 11-15 -> 11-20 | 86 | 5.1 | 17 | 3792 | 2924 | 66 / 13 / 20 / 0% | 44,032 | 1.60M | 1.76G |
| old/grus_nothrone | 11-17 -> 11-20 | 108 | 3.2 | 34 | 2369 | 858 | 62 / 23 / 13 / 1% | 55,296 | 1.94M | 2.14G |
| old/grus_nothrone_highcpuct | 11-23 -> 11-25 | 68 | 2.0 | 33 | 2256 | 3027 | 48 / 21 / 27 / 4% | 34,816 | 1.14M | 1.14G |
| old/grus_nothrone_kingtemp | 11-17 -> 11-20 | 120 | 2.6 | 45 | 1876 | 975 | 36 / 41 / 14 / 10% | 61,440 | 1.41M | 1.13G |
| old/grus_nothrone_lessrng | 11-18 -> 11-20 | 100 | 2.1 | 47 | 1852 | 999 | 40 / 37 / 15 / 7% | 51,200 | 1.31M | 1.05G |
| old/grus_nothrone_lowvalweight | 11-23 -> 11-24 | 17 | 0.4 | 40 | 1761 | 3118 | 56 / 9 / 34 / 1% | 8,704 | 0.30M | 0.30G |
| old/grus_nothrone_relfpu_dynsim | 11-21 -> 11-23 | 53 | 2.0 | 26 | 2808 | 3817 | 57 / 17 / 23 / 3% | 27,136 | 1.09M | 1.30G |
| old/grus_nothrone_rndplay | 11-23 -> 11-25 | 56 | 2.0 | 28 | 2756 | 3224 | 53 / 18 / 25 / 3% | 28,672 | 1.21M | 1.21G |
| old/jack_nothrone | undated | 140 | 2.6 (sum) | n/a | 1671 | 413 | 58 / 33 / 8 / 0% | 71,680 | 2.29M | 0.69G |
| old/jack_nothrone_lossfixed | 11-17 -> 11-20 | 187 | 3.2 | 25 | 2432 | 1222 | 52 / 34 / 13 / 1% | 136,192 | 5.82M | 3.20G |

"Other" is mainly checkpoint and JSON writing, merging new samples into the `deque`, and
recreating the worker pool. Dates are 2025. Runs with logs split over several files
(`output.txt`, `output2.txt`, ...) were resumed. On resume the replay buffer is **not** restored:
it starts empty and refills in ~2 iterations (`min_buffer_size` 200 k).

Overlap: on Nov 23 21:51-21:52 five runs started within two minutes. Usually 3 runs were active
at once, sometimes 5, each on its own node (owl31-owl35). The timings of one run are therefore not
affected by the others.

---

## 8. Training metrics

### 8.1 What the logged losses mean

- **Policy loss**: cross-entropy between the MCTS visit distribution (the target, with root
  Dirichlet noise) and the network softmax over all 1,176 move slots (illegal moves are not masked
  in the loss). Its floor is the entropy of the targets, not 0. Uniform over 1,176 slots = 7.07.
  The first training pass of a run starts around 5.9, after ~25 iterations it is ~3.5, and the
  long runs end at 2.3-2.4.
- **Value loss**: MSE against the game result z in [-1, 1]. With `reward_discount = 0.98`, the
  target is z x 0.98^(plies until the end), so targets are smaller in size and the MSE is lower
  (0.11-0.15 at the end). Without discount the MSE ends at 0.31-0.35. **Do not compare value
  loss between the two groups.** Mate-in-1 positions get target +1.
- **Validation** columns use samples from the same buffer as training (not held out), so they track
  the training loss almost exactly. They do not show overfitting.
- **"L2" column** = 0.5 x weight_decay x sum(w^2). It is only reported; AdamW applies decoupled
  weight decay. In the 6 M-buffer runs it rises to ~150-215 by iteration 75-100 and then falls.
- Within one iteration, policy loss changes by less than 0.02 from epoch 1 to epoch 10. Nearly all
  progress happens across iterations.

### 8.2 Learning curves of reference runs

Medians per bin of 25 iterations (20 for the short run). "Gate score" is the new network's score
against the current best (draw = 0.5).

**SE/jack_history_and_decay** (4x64, 13 input planes, value head 256, batch 512 x 1000 steps,
reward and search discount 0.98; these are the defaults committed in `train_brandubh.py`):

| Iter | Policy loss (train / val) | Value loss (train / val) | Policy loss epoch 1 -> 10 | 'L2' column | Grad norm pol / val | LR | Avg game length (plies) | Attacker win % | Draw % | Sims att / def | Gate score % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-25 | 3.54 / 3.54 | 0.204 / 0.212 | 3.56 -> 3.54 | 82 | 0.48 / 0.35 | 2e-03 | 23.2 | 56 | 0.0 | 1187 / 813 | 72 |
| 26-50 | 3.08 / 3.08 | 0.196 / 0.200 | 3.08 -> 3.08 | 165 | 0.50 / 0.44 | 2e-03 | 25.4 | 53 | 0.0 | 633 / 1367 | 59 |
| 51-75 | 2.81 / 2.82 | 0.182 / 0.192 | 2.81 -> 2.81 | 200 | 0.56 / 0.51 | 2e-03 | 28.4 | 51 | 1.3 | 446 / 1554 | 61 |
| 76-100 | 2.68 / 2.68 | 0.160 / 0.168 | 2.68 -> 2.67 | 215 | 0.61 / 0.57 | 2e-03 | 29.2 | 50 | 0.8 | 422 / 1578 | 59 |
| 101-125 | 2.58 / 2.58 | 0.138 / 0.138 | 2.58 -> 2.58 | 197 | 0.68 / 0.62 | 1e-03 | 30.0 | 52 | 1.6 | 290 / 1710 | 65 |
| 126-150 | 2.53 / 2.53 | 0.130 / 0.132 | 2.53 -> 2.53 | 167 | 0.76 / 0.69 | 1e-03 | 30.3 | 51 | 1.4 | 254 / 1746 | 55 |
| 151-175 | 2.49 / 2.50 | 0.131 / 0.134 | 2.49 -> 2.49 | 150 | 0.82 / 0.74 | 1e-03 | 31.5 | 49 | 2.0 | 231 / 1769 | 48 |
| 176-200 | 2.48 / 2.48 | 0.128 / 0.130 | 2.48 -> 2.48 | 139 | 0.85 / 0.76 | 1e-03 | 30.4 | 49 | 0.6 | 285 / 1715 | 49 |
| 201-225 | 2.40 / 2.40 | 0.118 / 0.118 | 2.40 -> 2.40 | 130 | 0.90 / 0.78 | 2e-04 | 32.6 | 50 | 2.3 | 173 / 1827 | 57 |
| 226-250 | 2.33 / 2.33 | 0.121 / 0.120 | 2.33 -> 2.33 | 121 | 0.99 / 0.86 | 2e-04 | 32.9 | 48 | 2.9 | 158 / 1842 | 53 |
| 251-275 | 2.32 / 2.32 | 0.116 / 0.116 | 2.32 -> 2.32 | 112 | 1.07 / 0.93 | 2e-04 | 32.3 | 49 | 2.1 | 181 / 1819 | 54 |
| 276-300 | 2.32 / 2.31 | 0.109 / 0.109 | 2.32 -> 2.32 | 104 | 1.14 / 1.00 | 2e-04 | 31.3 | 49 | 1.4 | 194 / 1806 | 54 |
| 301-325 | 2.31 / 2.30 | 0.108 / 0.106 | 2.31 -> 2.31 | 97 | 1.21 / 1.06 | 2e-04 | 32.6 | 49 | 2.3 | 184 / 1816 | 52 |
| 326-350 | 2.30 / 2.30 | 0.108 / 0.108 | 2.30 -> 2.30 | 90 | 1.27 / 1.12 | 2e-04 | 31.8 | 48 | 2.1 | 190 / 1810 | 50 |
| 351-375 | 2.30 / 2.30 | 0.108 / 0.109 | 2.30 -> 2.30 | 86 | 1.32 / 1.16 | 2e-04 | 32.8 | 49 | 2.9 | 175 / 1825 | 52 |

**SE/littleboy_decay** (6x32, value head 256, batch 256 x 2000 steps, LR steps at 100/200/300,
discounts 0.98; these match the last working-copy defaults, which were never committed):

| Iter | Policy loss (train / val) | Value loss (train / val) | Policy loss epoch 1 -> 10 | 'L2' column | Grad norm pol / val | LR | Avg game length (plies) | Attacker win % | Draw % | Sims att / def | Gate score % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-25 | 3.66 / 3.67 | 0.250 / 0.260 | 3.67 -> 3.65 | 58 | 0.92 / 0.83 | 1e-03 | 22.3 | 53 | 0.0 | 1140 / 860 | 71 |
| 26-50 | 3.23 / 3.23 | 0.231 / 0.245 | 3.24 -> 3.23 | 119 | 0.97 / 0.94 | 1e-03 | 24.3 | 53 | 0.0 | 754 / 1246 | 67 |
| 51-75 | 3.03 / 3.05 | 0.214 / 0.216 | 3.03 -> 3.03 | 150 | 1.12 / 1.12 | 1e-03 | 26.8 | 52 | 0.4 | 653 / 1347 | 58 |
| 76-100 | 2.82 / 2.82 | 0.190 / 0.201 | 2.82 -> 2.82 | 156 | 1.33 / 1.32 | 1e-03 | 27.8 | 53 | 0.4 | 558 / 1442 | 57 |
| 101-125 | 2.72 / 2.72 | 0.175 / 0.180 | 2.72 -> 2.72 | 146 | 1.55 / 1.52 | 5e-04 | 28.9 | 52 | 0.6 | 477 / 1523 | 54 |
| 126-150 | 2.68 / 2.68 | 0.168 / 0.171 | 2.67 -> 2.67 | 128 | 1.79 / 1.74 | 5e-04 | 29.8 | 51 | 1.2 | 434 / 1566 | 59 |
| 151-175 | 2.66 / 2.66 | 0.166 / 0.175 | 2.66 -> 2.65 | 117 | 1.97 / 1.92 | 5e-04 | 26.3 | 53 | 0.0 | 539 / 1461 | 56 |
| 176-200 | 2.65 / 2.66 | 0.163 / 0.169 | 2.65 -> 2.65 | 111 | 2.10 / 2.06 | 5e-04 | 27.2 | 53 | 0.2 | 468 / 1532 | 48 |
| 201-225 | 2.59 / 2.59 | 0.152 / 0.153 | 2.59 -> 2.59 | 107 | 2.26 / 2.20 | 1e-04 | 29.2 | 51 | 1.0 | 289 / 1711 | 55 |
| 226-250 | 2.55 / 2.54 | 0.152 / 0.152 | 2.55 -> 2.55 | 103 | 2.39 / 2.32 | 1e-04 | 30.2 | 49 | 1.6 | 258 / 1742 | 59 |

**SE/jack_highexplore** (4x64, value head 64, no reward discount, so value loss is on the
undiscounted scale):

| Iter | Policy loss (train / val) | Value loss (train / val) | Policy loss epoch 1 -> 10 | 'L2' column | Grad norm pol / val | LR | Avg game length (plies) | Attacker win % | Draw % | Sims att / def | Gate score % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-25 | 3.61 / 3.61 | 0.407 / 0.412 | 3.61 -> 3.61 | 84 | 0.58 / 0.53 | 2e-03 | 23.1 | 56 | 0.0 | 1142 / 858 | 75 |
| 26-50 | 3.21 / 3.20 | 0.419 / 0.426 | 3.21 -> 3.21 | 168 | 0.61 / 0.58 | 2e-03 | 23.7 | 54 | 0.0 | 573 / 1427 | 64 |
| 51-75 | 2.98 / 3.01 | 0.426 / 0.451 | 2.99 -> 2.98 | 203 | 0.69 / 0.68 | 2e-03 | 25.4 | 50 | 0.4 | 494 / 1506 | 64 |
| 76-100 | 2.79 / 2.80 | 0.412 / 0.433 | 2.79 -> 2.79 | 217 | 0.77 / 0.77 | 2e-03 | 30.0 | 49 | 4.1 | 408 / 1592 | 52 |
| 101-125 | 2.71 / 2.71 | 0.378 / 0.380 | 2.71 -> 2.71 | 199 | 0.84 / 0.80 | 1e-03 | 34.2 | 47 | 6.2 | 327 / 1673 | 59 |
| 126-150 | 2.65 / 2.65 | 0.352 / 0.356 | 2.65 -> 2.65 | 169 | 0.88 / 0.84 | 1e-03 | 33.8 | 49 | 5.5 | 309 / 1691 | 58 |
| 151-175 | 2.60 / 2.60 | 0.340 / 0.347 | 2.59 -> 2.59 | 151 | 0.95 / 0.89 | 1e-03 | 35.3 | 47 | 6.6 | 302 / 1698 | 59 |
| 176-200 | 2.55 / 2.55 | 0.333 / 0.336 | 2.55 -> 2.55 | 140 | 0.96 / 0.90 | 1e-03 | 38.9 | 46 | 8.8 | 288 / 1712 | 59 |
| 201-225 | 2.43 / 2.42 | 0.316 / 0.315 | 2.43 -> 2.43 | 132 | 0.98 / 0.88 | 2e-04 | 41.2 | 45 | 10.9 | 221 / 1779 | 64 |
| 226-250 | 2.38 / 2.39 | 0.312 / 0.312 | 2.39 -> 2.39 | 122 | 1.05 / 0.92 | 2e-04 | 41.9 | 44 | 11.5 | 215 / 1785 | 61 |
| 251-275 | 2.37 / 2.37 | 0.309 / 0.310 | 2.38 -> 2.38 | 113 | 1.12 / 0.98 | 2e-04 | 42.8 | 45 | 12.1 | 225 / 1775 | 56 |
| 276-300 | 2.38 / 2.38 | 0.307 / 0.307 | 2.38 -> 2.38 | 105 | 1.19 / 1.05 | 2e-04 | 41.7 | 45 | 9.8 | 224 / 1776 | 57 |
| 301-325 | 2.38 / 2.38 | 0.313 / 0.311 | 2.38 -> 2.38 | 101 | 1.25 / 1.10 | 2e-04 | 42.0 | 42 | 13.0 | 216 / 1784 | 59 |

**SE/grus_no-data-aug** (6x48, no symmetry augmentation, 0.75 M buffer, 512 x 120 steps). Its
policy loss is lower than in the augmented runs. Part of the reason may be the augmentation defect
in Appendix B, which gives half of the augmented copies wrong policy targets.

| Iter | Policy loss (train / val) | Value loss (train / val) | Policy loss epoch 1 -> 10 | 'L2' column | Grad norm pol / val | LR | Avg game length (plies) | Attacker win % | Draw % | Sims att / def | Gate score % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-20 | 3.14 / 3.17 | 0.435 / 0.450 | 3.17 -> 3.13 | 20 | 1.07 / 1.09 | 1e-03 | 22.8 | 53 | 0.0 | 1422 / 578 | 73 |
| 21-40 | 2.80 / 2.81 | 0.429 / 0.459 | 2.81 -> 2.80 | 25 | 1.15 / 1.12 | 1e-03 | 21.9 | 53 | 0.0 | 1317 / 683 | 70 |
| 41-60 | 2.63 / 2.65 | 0.413 / 0.430 | 2.64 -> 2.64 | 29 | 1.20 / 1.14 | 9e-04 | 21.9 | 53 | 0.0 | 894 / 1106 | 66 |
| 61-80 | 2.53 / 2.53 | 0.404 / 0.414 | 2.55 -> 2.54 | 31 | 1.25 / 1.19 | 7e-04 | 22.5 | 53 | 0.0 | 640 / 1360 | 59 |
| 81-100 | 2.31 / 2.31 | 0.398 / 0.411 | 2.31 -> 2.31 | 32 | 1.32 / 1.24 | 6e-04 | 23.1 | 50 | 0.0 | 524 / 1476 | 56 |
| 101-120 | 2.19 / 2.20 | 0.386 / 0.416 | 2.20 -> 2.19 | 33 | 1.40 / 1.33 | 5e-04 | 23.4 | 50 | 0.0 | 456 / 1544 | 62 |
| 121-140 | 2.14 / 2.14 | 0.382 / 0.388 | 2.14 -> 2.14 | 34 | 1.45 / 1.39 | 4e-04 | 23.9 | 54 | 0.0 | 388 / 1612 | 58 |
| 141-160 | 2.12 / 2.10 | 0.386 / 0.397 | 2.12 -> 2.11 | 34 | 1.51 / 1.45 | 3e-04 | 24.5 | 49 | 0.2 | 301 / 1699 | n/a |

### 8.3 End-of-run metrics, all runs

Median of the last 20 iterations. "Chained Elo" is the sum of Elo gains from accepted gating
matches. It is biased upward and not comparable between runs.

| Run | Policy loss | Value loss | Reward discount | Avg plies | Attacker win % | Draw % | Sims att / def | Gate accepts / evals | Chained Elo at end |
|---|---|---|---|---|---|---|---|---|---|
| SE/fatman_decay | 2.36 | 0.114 | 0.98 | 31.4 | 50 | 2.0 | 182 / 1818 | 6 / 50 | 1484 |
| SE/grus_default | 2.37 | 0.306 | 1.0 | 50.4 | 40 | 19.6 | 703 / 1297 | 8 / 43 | 2260 |
| SE/grus_high-CPUCT-zero-rel-FPU | 2.68 | 0.345 | 1.0 | 35.4 | 46 | 6.2 | 244 / 1756 | 6 / 60 | 1526 |
| SE/grus_no-data-aug | 2.14 | 0.381 | 1.0 | 24.3 | 52 | 0.0 | 377 / 1623 | 7 / 35 | 1576 |
| SE/jack_highexplore | 2.38 | 0.310 | 1.0 | 41.8 | 45 | 10.7 | 226 / 1774 | 6 / 77 | 1464 |
| SE/jack_history_and_decay | 2.30 | 0.108 | 0.98 | 32.6 | 49 | 2.7 | 176 / 1824 | 6 / 91 | 1591 |
| SE/jack_history_and_decay_smallbatch | 2.88 | 0.185 | 0.98 | 28.8 | 53 | 1.0 | 512 / 1488 | 6 / 17 | 1620 |
| SE/littleboy_decay | 2.56 | 0.152 | 0.98 | 30.0 | 49 | 1.6 | 268 / 1732 | 6 / 59 | 1448 |
| new/grus_default | 2.80 | 0.437 | 1.0 | 41.1 | 45 | 9.9 | 764 / 1236 | 5 / 18 | 1683 |
| new/grus_kingtemp | 2.83 | 0.487 | 1.0 | 27.1 | 51 | 1.0 | 682 / 1318 | 6 / 20 | 1769 |
| old/fatman_nothrone | 3.64 | 0.394 | 1.0 | 28.9 | 68 | 0.0 | 500 / 700 (fixed) | 12 / 17 | 208 |
| old/fatman_nothrone_lossfixed | 2.79 | 0.373 | 1.0 | 36.5 | 84 | 0.0 | 600 / 1600 (fixed) | 1 / 21 | 156 |
| old/grus_nothrone | 2.96 | 0.539 | 1.0 | 36.0 | 73 | 0.0 | 600 / 1600 (fixed) | 2 / 27 | 1181 |
| old/grus_nothrone_highcpuct | 2.67 | 0.426 | 1.0 | 39.4 | 47 | 6.4 | 501 / 1499 | 0 / 17 | 0 |
| old/grus_nothrone_kingtemp | 2.98 | 0.414 | 1.0 | 22.8 | 87 | 0.0 | 800 / 800 (fixed) | 5 / 29 | 1014 |
| old/grus_nothrone_lessrng | 3.20 | 0.455 | 1.0 | 26.9 | 90 | 0.0 | 800 / 800 (fixed) | 4 / 24 | 832 |
| old/grus_nothrone_lowvalweight | 2.77 | 0.439 | 1.0 | 39.2 | 47 | 7.0 | 725 / 1275 | 0 / 4 | 0 |
| old/grus_nothrone_relfpu_dynsim | 2.97 | 0.628 | 1.0 | 44.1 | 52 | 0.0 | 825 / 1575 | 0 / 13 | 0 |
| old/grus_nothrone_rndplay | 2.56 | 0.382 | 1.0 | 46.5 | 43 | 12.8 | 724 / 1276 | 1 / 13 | 166 |
| old/jack_nothrone | 3.20 | 0.467 | 1.0 | 36.5 | 46 | 0.0 | 250 / 350 (fixed) | 20 / 35 | 281 |
| old/jack_nothrone_lossfixed | 3.04 | 0.478 | 1.0 | 50.8 | 75 | 0.0 | 300 / 800 (fixed) | 1 / 21 | 163 |

The `old/` runs used value-loss weight 0.25 (one used 20), weight decay 1e-4 in some, and
symmetry augmentation that was broken before Nov 23, 2025. Treat their loss values as historical only.

### 8.4 Game dynamics

- **Attackers are strongly favoured** under these rules (throne disabled, king captured by 2
  pieces, king can capture). In head-to-head matches between trained networks with equal
  simulations, attackers won 75-95 % of games, and more often at higher simulation counts
  (100 to 1,600 sims tested).
- **Dynamic simulation balancing** keeps attacker + defender = 2,000 simulations and moves budget
  toward the losing side. The attacker's share falls from 1,000 to ~400-560 by iteration 100
  and to ~160-290 (8-15 %) after iteration 200. This keeps the self-play attacker win rate at
  ~45-52 %. Runs without balancing (`old/`, fixed budgets) had 46-90 % attacker wins, mostly above 70 %.
- **Game length** grows from ~22 plies early to 30-42 plies late. Draws (100-ply limit) go from
  0 % to 2-3 % in the discounted runs, 10-13 % in `SE/jack_highexplore`, and ~20 % in `SE/grus_default`.
- 50 % of self-play games start with 1-4 random moves per side (these plies are not stored).
  Unique board states after the first 6 plies: > 99 % per iteration.
- **Gating**: the threshold is a 70 % score. Each run accepted a new best network 6-8 times, all
  before iteration ~130. After iteration 50 the median gate score was 51-58 %. These matches were
  played with the throne enabled (training: disabled) and, for history models, without history
  planes (Appendix B items 3-4), so gate scores and chained Elo measure play under other conditions.
- **Cross-run strength** (separate head-to-head matches with `play.py`, which used the correct
  rules, but gave history models empty history planes; 256 games each, 100-800 sims):
  `SE/jack_history_and_decay` iteration 108 beat `SE/grus_default` iteration 175 by 15-34 Elo,
  and `SE/jack_history_and_decay_smallbatch` iteration 64 by 53-83 Elo.
  `SE/jack_highexplore` iterations 150-200 beat `SE/grus_default` iteration 175 by 38-87 Elo.

---

## 9. Other board sizes (only a few iterations each)

| Variant | Net | Sims per side | Plies/game | Self-play s/iter (512 games) | Training s/iter |
|---|---|---|---|---|---|
| Tablut 9x9 | 5x64, 801,066 params | 120 | 100-106 | 1,720-1,930 | ~225 |
| Hnefatafl 11x11, no throne | 6x96, 2,186,818 params | 200 | 115-118 | 5,610-6,270 | 770-850 |
| Hnefatafl 11x11, standard rules | 6x96, 2,186,818 params | 200 | ~421 | 17,800-18,400 | 1,120-1,780 |
| Hnefatafl 11x11, standard rules | 6x96, 2,186,818 params | 300 | ~160 | 10,230 | 790 |

---

## 10. What these numbers imply for the GPU desktop

- **Network-call demand**: one iteration at the standard setting needs ~11-21 M network
  evaluations. One owl31-35 node (128 workers on 64 cores from 2017) delivered 10-17 k per second,
  i.e. 10-30 minutes of self-play per iteration.
- **The cluster numbers are a low baseline per core.** Each worker managed only ~80-130
  simulations/s. The same code runs ~600-700 simulations/s on one modern core when it has the core to
  itself (Section 4.6). A recent desktop CPU should therefore beat one cluster worker by several
  times per core.
- **The CPU design never batched network calls.** Every call was batch 1 inside a single-threaded
  process. A GPU only helps if many leaf evaluations are batched, from many games in one process
  and/or several leaves per tree.
- **Tree search is Python on the CPU.** On a modern core it costs ~0.5 ms per simulation outside the
  network (~1,800 simulations/s per core). A desktop CPU with N cores therefore caps self-play at
  roughly N x 1,800 simulations/s, unless the tree code gets faster.
- **Backpropagation is cheap** for these networks: 512 k samples per iteration is ~30-60 s on 16
  modern CPU threads, and should be much less on a GPU. On the cluster, training time was dominated
  by buffer-size-dependent overhead (Section 5.3).
- **RAM**: a 6 M-sample buffer (35-45 GB) plus the full copy made for training (~70-90 GB peak) fit
  in the cluster's 256 GB nodes but does not fit in 64 GB. The states are binary piece planes and
  the policies are sparse, so a compact storage format, a smaller buffer, or no stored augmentation
  copies would be needed.
- **Gating cost 13-30 % of wall time** without affecting self-play.

---

## Appendix A: Run configurations

Read from the `config` dict stored in each run's last checkpoint. "n/a" means the option did not
exist yet; the code default then applied (FPU -0.5, no reward discount, no search discount, 500-ply limit).

| Run | Net | Batch x batches/epoch x epochs | LR schedule | WD | Value-loss weight | Self-play sims A/D (start) | Dyn. sim balance | c_puct / FPU (self-play) | c_puct / FPU (eval) | Dirichlet a / eps | Temp mode (moves) | Buffer max / min | Aug | Random openings | Max plies | Reward / search discount |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SE/fatman_decay | 6x64, vh256, 4pl | 256 x 200 x 10 | steps 0.001, 0.0005, 0.0001, 1e-05 at it 100, 200, 300 | 0.01 | 1.5 | 1000/1000 | yes | 2 / 0.0 | 1.5 / -0.25 | 0.3 / 0.35 | decay 20 | 6M / 200k | yes | yes | 100 | 0.98 / 0.98 |
| SE/grus_default | 6x48, vh64, 4pl | 512 x 100 x 10 | 0.0014 x 0.99/it, floor 3e-05 | 0.01 | 1.5 | 1000/1000 | yes | 1.5 / -0.5 | 1.0 / -0.5 | 0.3 / 0.25 | decay 16 | 6M / 200k | yes | yes | 100 | 1.0 / 1.0 |
| SE/grus_high-CPUCT-zero-rel-FPU | 6x48, vh64, 4pl | 512 x 100 x 10 | 0.0014 x 0.99/it, floor 3e-05 | 0.01 | 1.5 | 1000/1000 | yes | 2.5 / 0.0 | 1.5 / -0.25 | 0.3 / 0.35 | decay 16 | 6M / 200k | yes | yes | 100 | 1.0 / 1.0 |
| SE/grus_no-data-aug | 6x48, vh64, 4pl | 512 x 12 x 10 | 0.0014 x 0.99/it, floor 3e-05 | 0.01 | 1.5 | 1000/1000 | yes | 2.5 / 0.0 | 1.5 / -0.25 | 0.3 / 0.35 | decay 16 | 0.75M / 25k | no | yes | 100 | 1.0 / 1.0 |
| SE/jack_highexplore | 4x64, vh64, 4pl | 512 x 100 x 10 | steps 0.002, 0.001, 0.0002, 2e-05 at it 100, 200, 400 | 0.01 | 1.5 | 1000/1000 | yes | 2 / 0.0 | 1.5 / -0.25 | 0.3 / 0.35 | decay 16 | 6M / 200k | yes | yes | 100 | 1.0 / 1.0 |
| SE/jack_history_and_decay | 4x64, vh256, 13pl | 512 x 100 x 10 | steps 0.002, 0.001, 0.0002, 2e-05 at it 100, 200, 400 | 0.01 | 1.5 | 1000/1000 | yes | 2 / 0.0 | 1.5 / -0.25 | 0.3 / 0.35 | decay 16 | 6M / 200k | yes | yes | 100 | 0.98 / 0.98 |
| SE/jack_history_and_decay_smallbatch | 4x64, vh256, 13pl | 64 x 800 x 10 | steps 0.0005, 0.00025, 0.0001, 1e-05 at it 100, 200, 400 | 0.01 | 1.5 | 1000/1000 | yes | 2 / 0.0 | 1.5 / -0.25 | 0.3 / 0.35 | decay 16 | 6M / 200k | yes | yes | 100 | 0.98 / 0.98 |
| SE/littleboy_decay | 6x32, vh256, 4pl | 256 x 200 x 10 | steps 0.001, 0.0005, 0.0001, 1e-05 at it 100, 200, 300 | 0.01 | 1.5 | 1000/1000 | yes | 2 / 0.0 | 1.5 / -0.25 | 0.3 / 0.35 | decay 20 | 6M / 200k | yes | yes | 100 | 0.98 / 0.98 |
| new/grus_default | 6x48, vh64, 4pl | 512 x 100 x 10 | 0.0014 x 0.99/it, floor 3e-05 | 0.01 | 1.5 | 1000/1000 | yes | 1.5 / -0.5 | 1.0 / -0.5 | 0.3 / 0.25 | decay 16 | 6M / 200k | yes | yes | 100 | 1.0 / 1.0 |
| new/grus_kingtemp | 6x48, vh64, 4pl | 512 x 100 x 10 | 0.0014 x 0.99/it, floor 3e-05 | 0.01 | 1.5 | 1000/1000 | yes | 1.5 / -0.5 | 1.0 / -0.5 | 0.3 / 0.25 | king | 6M / 200k | yes | yes | 100 | 1.0 / 1.0 |
| old/fatman_nothrone | 6x96, vh64, 4pl | 256 x 100 x 10 | 0.001 x 0.995/it | 0.0001 | 0.25 | 500/700 | no | 1.4 / n/a | n/a | 0.3 / 0.25 | decay 20 | 10M / 600k | yes | no | 500 | n/a |
| old/fatman_nothrone_lossfixed | 6x96, vh64, 4pl | 512 x 100 x 10 | 0.0014 x 0.99/it, floor 3e-05 | 0.01 | 0.25 | 600/1600 | no | 1.4 / n/a | n/a | 0.3 / 0.25 | decay 16 | 6M / 600k | yes | no | 500 | n/a |
| old/grus_nothrone | 6x48, vh64, 4pl | 512 x 100 x 10 | 0.0014 x 0.99/it, floor 3e-05 | 0.01 | 0.25 | 600/1600 | no | 1.4 / n/a | n/a | 0.3 / 0.25 | decay 16 | 6M / 600k | yes | no | 500 | n/a |
| old/grus_nothrone_highcpuct | 6x48, vh64, 4pl | 512 x 100 x 10 | 0.00028 x 0.99/it, floor 3e-05 | 0.01 | 1.5 | 1000/1000 | yes | 2.5 / -0.5 | 1.0 / -0.5 | 0.3 / 0.25 | decay 16 | 6M / 200k | yes | yes | 100 | n/a |
| old/grus_nothrone_kingtemp | 6x48, vh64, 4pl | 512 x 100 x 10 | 0.0014 x 0.99/it, floor 3e-05 | 0.01 | 0.25 | 800/800 | no | 1.4 / n/a | n/a | 0.3 / 0.25 | king | 10M / 200k | yes | no | 500 | n/a |
| old/grus_nothrone_lessrng | 6x48, vh64, 4pl | 512 x 100 x 10 | 0.0014 x 0.99/it, floor 3e-05 | 0.01 | 0.25 | 800/800 | no | 1.4 / n/a | n/a | 0.3 / 0.15 | decay 12 (temp 0.8) | 10M / 200k | yes | no | 500 | n/a |
| old/grus_nothrone_lowvalweight | 6x48, vh64, 4pl | 512 x 100 x 10 | 0.00028 x 0.99/it, floor 3e-05 | 0.01 | 0.5 | 1000/1000 | yes | 1.5 / -0.5 | 1.0 / -0.5 | 0.3 / 0.25 | decay 16 | 6M / 200k | yes | yes | 100 | n/a |
| old/grus_nothrone_relfpu_dynsim | 6x48, vh64, 4pl | 512 x 100 x 10 | 0.0014 x 0.99/it, floor 3e-05 | 0.01 | 0.25 | 1200/1200 | yes | 1.4 / -0.5 | n/a | 0.3 / 0.25 | decay 16 | 6M / 200k | yes | no | 500 | n/a |
| old/grus_nothrone_rndplay | 6x48, vh64, 4pl | 512 x 100 x 10 | 0.00028 x 0.99/it, floor 3e-05 | 0.01 | 1.5 | 1000/1000 | yes | 1.5 / -0.5 | 1.0 / -0.5 | 0.3 / 0.25 | decay 16 | 6M / 200k | yes | yes | 100 | n/a |
| old/jack_nothrone | 4x64, vh64, 4pl | 256 x 100 x 10 | 0.001 x 0.995/it | 0.0001 | 0.25 | 250/350 | no | 1.4 / n/a | n/a | 0.3 / 0.25 | decay 20 | 10M / 600k | yes | no | 500 | n/a |
| old/jack_nothrone_lossfixed | 4x64, vh64, 4pl | 512 x 100 x 10 | 0.0014 x 0.99/it, floor 3e-05 | 0.01 | 0.25 | 300/800 | no | 1.4 / n/a | n/a | 0.3 / 0.25 | decay 16 | 10M / 1000k | yes | no | 500 | n/a |

Common to all runs: 512 games per iteration, 128 workers, 10 epochs, gating every 4 iterations
with 128 games and a 70 % threshold, Brandubh with throne disabled, 2-piece king capture, king can
capture. Gating sims: 800 in all `SE/` and `new/` runs; 250-1,200 in `old/` runs. Gating
temperature: 0.65 for the first 10 plies (1.0 in `old/fatman_nothrone` and `old/jack_nothrone`). `SE/` networks have squeeze-excitation blocks (reduction 8).
`new/` and `old/` networks are plain residual blocks with the same layout.

Network layout (all runs): 3x3 conv stem -> N residual blocks (two 3x3 conv + BN, optional SE) ->
policy head (1x1 conv to 2 channels, BN, FC 98 -> 1,176) and value head (1x1 conv to 1 channel,
BN, FC 49 -> hidden -> 1, tanh). Move encoding: from-square (49) x direction (4) x distance (1-6) = 1,176.

---

## Appendix B: Code issues in the CPU-era code

All runs in this document used the code before these fixes. Items 1-10 were fixed in the commit
after `555c1b3` (Oct 2026); `test_training_fixes.py` covers items 1, 3, 4 and 5.

1. Symmetry augmentation: `np.rot90` rotates the board counter-clockwise, while `rotate_policy`
   rotates moves clockwise. So the 90° and 270° versions (and their mirrored versions), i.e. 4 of
   the 8 symmetries, had policy targets that did not match the board. The identity sample was also
   stored twice (9 copies per position). Now: 8 consistent copies.
2. Workers forked after the main process ran `torch.compile` (training) could hang forever:
   the forked copy of Inductor's compile thread pool has no live threads. Seen reliably in the
   vs-random evaluation right after training; timing-dependent elsewhere. Now each worker compiles
   in-process (`compile_threads = 1`).
3. **Gating and vs-random games used the game class's default rules (throne enabled)**, while all
   self-play used `throne_enabled=False`. They also used a 500-ply limit instead of
   `max_game_length` (100). All gate scores, gating draws and chained Elo values in this document
   were therefore measured under different rules from training.
4. History models (13 input planes) got empty history planes in gating, `play.py` and the GUI:
   `MCTS.search` without an explicit history filled only the current position.
   `StateHistory.from_game` now rebuilds the real history from `game.position_history`.
5. Legal moves were generated three times per expanded MCTS node (now once).
6. The "Network eval" timer included the "Get legal moves" timer, so the printed MCTS total
   double-counted it (now separate).
7. `np.random.choice(N, batch, replace=False)` cost O(N) per batch (97 ms at N = 6 M); now
   `np.random.default_rng().choice` (O(batch)). A per-batch L2 sum that was never used was removed,
   and gradient norms now need 2 `.item()` calls per batch instead of one per parameter.
8. Gating and vs-random evaluation ran their two colour halves one after the other; both are now
   submitted at once.
9. `training_history.json` stored 0 for `avg_game_length`; now the real value.
10. The `--device` help claimed auto-detection; the default is `"cpu"`.

Not changed (design decisions): the replay buffer is not saved on resume; "validation" samples come
from the training buffer; the buffer is stored as float32 and fully copied for training; games in
progress are discarded when 512 games have finished; `torch.compile` runs for every game and
every training call.

---

## Appendix C: Data sources and paths outside the repo

- Logs: `checkpoints_brandubh{,_new,_SE}/<run>/output*.txt` (21 runs; resumed runs have several
  files), `training_history.json`, per-iteration checkpoints `checkpoint_iter_<n>.pth` (their
  `config` dict was used for Appendix A), plots `plot_*.png`.
- Benchmark notes: `BENCHMARK_RESULTS.md`, `DENORMAL_FIX.md`,
  `checkpoints_brandubh/fatman_nothrone/INFERENCE_DEGRADATION_REPORT.md`,
  `inference_benchmark_results.csv`. Head-to-head matches: `output_eval*.txt`, `model_eval_output.txt`.
- None of these are tracked by git.
- Outside the repo: the Python venv `../RL_tictactoe/.RL_venv` (Python 3.10.2 from a shared module
  install under another user's home directory, torch 2.9.0, numpy 2.2.6); temp files and Inductor
  caches in `/dev/shm/` (earlier `/tmp/torchinductor_<pid>`).
- Earlier experiments (`checkpoints_old/`, Nov 3-9, 2025) and `mosquito_nothrone` have no
  `output.txt` with timings and are not included.
