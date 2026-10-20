#!/bin/bash
# Training run from configs/<rule set>/<run>.json, writing to checkpoints/<rule set>/<run>/.
# The rule sets and their configs are listed in RULESETS.md; within a rule set the configs share
# all settings except the network.
# Start one per free node inside screen:
#   screen -S sim1 ./run.sh copenhagen line_10x64_gpool
#   screen -S sim2 ./run.sh sami_tablut line_6x48_gpool
# Resume after an interruption (replace N by the last saved iteration):
#   screen -S sim1 ./run.sh copenhagen line_10x64_gpool --resume checkpoints/copenhagen/line_10x64_gpool/checkpoint_iter_N.pth
# Further arguments go to run_training.py, e.g. --set self_play.num_iterations=300
#
# 64 self-play workers, one per physical core: 128 (all hardware threads) give 11-18 % more
# simulations per second, but with 256 games per iteration twice as many long games are still
# running at the cut-off and are thrown away, so an iteration takes as long and its games are
# shorter on average. TRAIN_THREADS: threads of the training step; in the real pipeline 64 was
# faster than 32 (hyades10: 0.71 against 1.19 s per batch) and 128 was far slower.
# The processes show up as "run_sims" in top and ps; the output also goes to output.txt.
TRAIN_THREADS=64
ruleset=$1
name=$2
shift 2
cd /mn/stornext/u3/jonassl/Games/Hnefatafl_RL
run_dir=checkpoints/$ruleset/$name
mkdir -p $run_dir
OMP_NUM_THREADS=$TRAIN_THREADS MKL_NUM_THREADS=$TRAIN_THREADS exec .venv/bin/python -u run_training.py \
    configs/$ruleset/$name.json --num-workers 64 --log $run_dir/output.txt "$@"
