#!/bin/bash
# Test training run on Tablut (9x9) with today's train_tablut.py defaults: 6x48 network, 400
# simulations per side, 512 games per iteration, draw after 200 plies, discount 0.99, 20-80
# simulation-balancer band, corner escape, king captured by two, no throne (as the Brandubh runs).
# Start it inside screen on a free node (64 training threads, 128 self-play workers):
#   screen -S tablut_test ./run_tablut_test.sh
# Resume after an interruption (replace N by the last saved iteration):
#   screen -S tablut_test ./run_tablut_test.sh --resume checkpoints_tablut/test_6x48/checkpoint_iter_N.pth
cd /mn/stornext/u3/jonassl/Games/Hnefatafl_RL
run_dir=checkpoints_tablut/test_6x48
mkdir -p $run_dir
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 ../RL_tictactoe/.RL_venv/bin/python -u train_tablut.py \
    --num-workers 128 --iterations 150 --checkpoint-dir $run_dir "$@" 2>&1 | tee -a $run_dir/output.txt
