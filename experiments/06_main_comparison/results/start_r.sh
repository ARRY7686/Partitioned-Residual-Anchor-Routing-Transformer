#!/bin/bash
source /venv/main/bin/activate >/dev/null
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log RART_PRECISION=bf16
cd /root/exp06
RART_PHASES=R RART_JOBS_PER_GPU=3 RART_LONGEST_FIRST=1 RART_TIME_BUDGET_HOURS=100 python -u run_suite.py >> suite_R.log 2>&1
