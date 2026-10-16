#!/bin/bash
# Rerun of SE/littleboy_decay with the fixed code, to measure what the correctness fixes and the
# 20-80 simulation-balancer band gained. The old run's stored config equals the Brandubh baseline
# config (configs/brandubh/baseline.json, the former train_brandubh.py defaults) with 128 workers
# (checked), except the band (default dynamic_sim_min_win_rate 0.2; the old run pushed the score to
# 50 %). Compare it with the old run afterwards with compare_littleboy_fixed.sh (new against old
# checkpoint at the same iteration).
#
# 64 training threads: PyTorch's default of one per physical core, as in the old run (the
# OMP_NUM_THREADS=16 in ~/.profile came later). Start it inside screen:
#   screen -S sim7 ./run_littleboy_decay_fixed.sh
# Resume after an interruption (replace N by the last saved iteration):
#   screen -S sim7 ./run_littleboy_decay_fixed.sh --resume checkpoints_brandubh_SE/littleboy_decay_fixed/checkpoint_iter_N.pth
#
# Two settings keep a resume identical to the run as started (checked against its checkpoints):
# - network.arch=legacy: its checkpoints hold the old BrandubhNet class, whose layer names differ
#   from TaflNet's, so the baseline's "tafl" network cannot load them
# - evaluation.eval_temperature=0.65: the default when the run started (the configs now use 1.0)
cd /mn/stornext/u3/jonassl/Games/Hnefatafl_RL
run_dir=checkpoints_brandubh_SE/littleboy_decay_fixed
mkdir -p $run_dir
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 exec .venv/bin/python -u run_training.py configs/brandubh/baseline.json \
    --num-workers 128 --set self_play.num_iterations=236 --set network.arch=legacy \
    --set evaluation.eval_temperature=0.65 --checkpoint-dir $run_dir --log $run_dir/output.txt "$@"
