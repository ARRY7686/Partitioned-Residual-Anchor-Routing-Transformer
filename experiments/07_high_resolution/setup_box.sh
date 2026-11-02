#!/bin/bash
# One-time setup of a rented GPU box for experiment 07.
# usage (on the box, inside a folder holding train.py, run_suite.py, build_data.py): bash setup_box.sh
set -e
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

python - <<'PY' || pip install -q --index-url https://download.pytorch.org/whl/cu128 torch torchvision
import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, torch.cuda.get_device_name(0))
PY
pip install -q timm

python build_data.py mnist
[ -f data/dfg/index.json ] || python build_data.py dfg     # 7.5 GB; about 10-20 min on a datacenter link

# MPS lets two jobs share the GPU with less overhead than time-slicing
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d

# smoke tests: one tiny focus job per dataset, end to end
RART_SMOKE=1 RART_TAG=smoke_m RART_DATASET=mpmnist python -u train.py 2>&1 | tail -2
RART_SMOKE=1 RART_TAG=smoke_d RART_DATASET=dfg python -u train.py 2>&1 | tail -2
echo "setup done. Next: RART_PHASES=C RART_JOBS_PER_GPU=1 python run_suite.py, then RART_PHASES=M,D"
