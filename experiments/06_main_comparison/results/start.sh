#!/bin/bash
# usage: start.sh PHASES   (starts the runner detached, 6 jobs on the GPU via MPS)
source /venv/main/bin/activate >/dev/null
export RART_PRECISION=bf16
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
cd /root/exp06
RART_PHASES=$1 RART_JOBS_PER_GPU=6 RART_LONGEST_FIRST=1 RART_TIME_BUDGET_HOURS=100 \
  nohup setsid python -u run_suite.py >> suite.log 2>&1 < /dev/null &
