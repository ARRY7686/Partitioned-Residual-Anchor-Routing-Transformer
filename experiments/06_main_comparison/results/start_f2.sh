#!/bin/bash
source /venv/main/bin/activate >/dev/null
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log RART_PRECISION=bf16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/exp06
RART_SKIP_TAGS=F2_keep0.25_importance-bounded_clip_seed42,F2_keep0.25_importance-bounded_clip_seed43,F2_keep0.25_importance-bounded_clip_seed44 RART_PHASES=F2 RART_JOBS_PER_GPU=2 RART_TIME_BUDGET_HOURS=100 python -u run_suite.py >> suite_F2.log 2>&1
