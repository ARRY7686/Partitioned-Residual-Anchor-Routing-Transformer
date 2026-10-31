#!/bin/bash
# Re-run phase R (it failed with OOM) once at least 20 GB of GPU memory is free; 2 jobs at a time.
until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -gt 20000 ]; do sleep 60; done
source /venv/main/bin/activate >/dev/null
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log RART_PRECISION=bf16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/exp06
echo "=== restart $(date)" >> suite_R.log
RART_SKIP_TAGS=B_pvt_seed44 RART_PHASES=R RART_JOBS_PER_GPU=2 RART_LONGEST_FIRST=1 RART_TIME_BUDGET_HOURS=100 python -u run_suite.py >> suite_R.log 2>&1
