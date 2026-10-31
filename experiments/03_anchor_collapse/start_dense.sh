#!/bin/bash
source /venv/main/bin/activate >/dev/null
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p /root/exp03/dense && ln -sfn /root/exp03/working/data /root/exp03/dense/data
cd /root/exp03
for s in 42 43 44; do RART_SEED=$s RART_MODELS=dense RART_WORK_DIR=/root/exp03/dense python -u anchor_collapse.py > dense_seed$s.log 2>&1 & done
wait
