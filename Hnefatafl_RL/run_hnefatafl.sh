#!/bin/bash
# Hnefatafl (11x11) training run from configs/hnefatafl/<name>.json. The three configs share all
# settings except the network:
#   line_10x64_gpool                    the main run: 10 line blocks x 64, global pooling
#   line_10x64_gpool_mobility_region    the same + reach planes + king-region plane
#   baseline                            8x64 res blocks with the old dense heads (reference)
# Start one per free node inside screen (64 training threads, all CPUs as self-play workers):
#   screen -S hnef_line ./run_hnefatafl.sh line_10x64_gpool
# Resume after an interruption (replace N by the last saved iteration):
#   screen -S hnef_line ./run_hnefatafl.sh line_10x64_gpool --resume checkpoints_hnefatafl/line_10x64_gpool/checkpoint_iter_N.pth
# Further arguments go to run_training.py, e.g. --set self_play.num_iterations=300
name=$1
shift
cd /mn/stornext/u3/jonassl/Games/Hnefatafl_RL
run_dir=checkpoints_hnefatafl/$name
mkdir -p $run_dir
OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 ../RL_tictactoe/.RL_venv/bin/python -u run_training.py configs/hnefatafl/$name.json \
    "$@" 2>&1 | tee -a $run_dir/output.txt
