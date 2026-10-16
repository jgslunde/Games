#!/bin/bash
# Compare the rerun with the fixed code (run_littleboy_decay_fixed.sh) against the old run at the
# same iterations, with the same search for both networks, so the result measures only what the
# training fixes (augmentation, FPU, ...) and the 20-80 simulation-balancer band gained.
# Each point k averages the checkpoint pairs k-4 ... k (640 games) to remove checkpoint noise;
# expect about +-15-30 Elo (1 sigma) per point. About 10-20 minutes per point on an owl node.
# Reading the result: the new run ahead by 50 Elo or more from iteration ~100 on means the fixes
# matter; within +-30 Elo means they did not change the strength much.
#   screen -S sim8 ./compare_littleboy_fixed.sh        (results in new_vs_old.txt)
cd /mn/stornext/u3/jonassl/Games/Hnefatafl_RL
OLD=checkpoints_brandubh_SE/littleboy_decay
NEW=checkpoints_brandubh_SE/littleboy_decay_fixed
for k in 25 50 100 150 200 236; do
    for i in $(seq $((k - 4)) $k); do
        echo -n "k=$k i=$i: "
        .venv/bin/python play.py $NEW/checkpoint_iter_$i.pth $OLD/checkpoint_iter_$i.pth \
            --simulations 800 --num-games 64 --swap-colors --num-workers 128 --temperature 1.0 \
            --temperature-threshold 10 --c-puct 1.5 --fpu-reduction -0.25 \
            | grep -E "Agent 1 wins|Agent 2 wins|Draws:" | tr -s ' \n' ' '
        echo
    done
done > new_vs_old.txt
