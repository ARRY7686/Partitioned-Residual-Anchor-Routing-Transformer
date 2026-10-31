#!/bin/bash
# One-time setup of a fresh rented GPU box for experiment 06.
# usage (on the box, inside /root/exp06 holding train.py, run_suite.py, start.sh, stop.sh,
# build_cifar_from_hf.py):  bash setup_box.sh
set -e
cd /root/exp06
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

python - <<'EOF' || pip install -q --index-url https://download.pytorch.org/whl/cu128 torch torchvision
import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, torch.cuda.get_device_name(0))
EOF
pip install -q timm huggingface_hub pyarrow pillow

[ -f data/cifar-10-batches-py/test_batch ] || python build_cifar_from_hf.py data

# MPS lets several jobs share the GPU at about twice the throughput of time-slicing
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d

# smoke test: one tiny focus job end to end in bf16
RART_SMOKE=1 RART_PRECISION=bf16 RART_TAG=smoke python -u train.py 2>&1 | tail -3
chmod +x start.sh stop.sh
echo "setup done; start with: ./start.sh A,B,D,F   (phase C afterwards, alone)"
