#!/bin/bash
# usage: diag.sh TAG LR  -- fixed-16 seed 43, phase B recipe, bf16, spike logging
source /venv/main/bin/activate >/dev/null
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p /root/diag; cd /root/exp06
RART_KIND=fixed RART_ANCHORS=16 RART_SEED=43 RART_EPOCHS=30 RART_GRAD_CLIP=1.0 RART_WARMUP_EPOCHS=2 \
RART_PRECISION=bf16 RART_LR=$2 RART_SPIKE_FACTOR=8 RART_TAG=$1 RART_OUT_DIR=/root/diag RART_DATA_DIR=/root/exp06/data \
  nohup setsid python -u train.py > /root/diag/$1.log 2>&1 < /dev/null &
