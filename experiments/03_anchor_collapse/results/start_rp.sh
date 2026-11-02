#!/bin/bash
# Faithful Representative Attention pathway on the experiment 03 collapse benchmark, 3 seeds.
source /venv/main/bin/activate
cd /root/exp03
mkdir -p rpattention/logs
run() { CUDA_VISIBLE_DEVICES=$1 RART_SEED=$2 RART_MODELS=rpattention RART_WORK_DIR=/root/exp03/rpattention \
        python -u anchor_collapse.py > rpattention/logs/rpattention_seed$2.log 2>&1; }
run 0 42 & run 1 43 & wait
run 0 44
echo done > rpattention/ALL_DONE
