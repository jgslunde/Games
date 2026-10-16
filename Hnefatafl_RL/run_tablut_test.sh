#!/bin/bash
# Test training run on Tablut (9x9) with the Tablut baseline config (configs/tablut/baseline.json,
# the former train_tablut.py defaults): 6x48 network, 400 simulations per side, 512 games per
# iteration, draw after 200 plies, discount 0.99, 20-80 simulation-balancer band, corner escape,
# king captured by two, no throne (as the Brandubh runs).
# Start it inside screen on a free node (64 training threads, 128 self-play workers):
#   screen -S sim9 ./run_tablut_test.sh
# Resume after an interruption (replace N by the last saved iteration):
#   screen -S sim9 ./run_tablut_test.sh --resume checkpoints_tablut/test_6x48/checkpoint_iter_N.pth
#
# Two settings keep a resume identical to the run as started (checked against its checkpoints):
# - network.arch=legacy: its checkpoints hold the old TablutNet class, whose layer names differ
#   from TaflNet's, so the baseline's "tafl" network cannot load them
# - evaluation.eval_temperature=0.65: the default when the run started (the configs now use 1.0)
cd /mn/stornext/u3/jonassl/Games/Hnefatafl_RL
run_dir=checkpoints_tablut/test_6x48
mkdir -p $run_dir
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 exec .venv/bin/python -u run_training.py configs/tablut/baseline.json \
    --num-workers 128 --set self_play.num_iterations=150 --set network.arch=legacy \
    --set evaluation.eval_temperature=0.65 --checkpoint-dir $run_dir --log $run_dir/output.txt "$@"
