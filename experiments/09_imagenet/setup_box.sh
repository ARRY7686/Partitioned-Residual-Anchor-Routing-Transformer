#!/bin/bash
# One-time setup of a rented GPU box for experiment 09 (ImageNet).
# Copy experiments/08_efficient_routing and experiments/09_imagenet to the box (same layout),
# then, inside 09_imagenet:  bash setup_box.sh
# The Hugging Face login is done by the user on the box first:  hf auth login
set -e
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

python - <<'PY' || pip install -q --index-url https://download.pytorch.org/whl/cu128 torch torchvision
import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, torch.cuda.get_device_name(0))
PY
pip install -q timm pyarrow "huggingface_hub[hf_xet]"
python -c "from huggingface_hub import whoami; print('HF user:', whoami()['name'])" \
  || { echo "run 'hf auth login' first (ImageNet is gated)"; exit 1; }

CPUS=$(python -c "import os; print(len(os.sched_getaffinity(0)))")
echo "usable CPU cores: $CPUS"
df -h . | tail -1

# ImageNet-100: streams all 294 train shards (~140 GB download) but keeps ~10%; needs ~15 GB disk
[ -f data/in100/meta.json ] || python prepare_data.py --classes imagenet100_classes.txt --out data/in100 --workers $CPUS

# Data loader speed alone, then one short end-to-end run per model family
python train.py --data data/in100 --bench-loader 200 --workers $((CPUS - 2))
python train.py --data data/in100 --model anchor_b0_fixed --epochs 1 --max-steps 100 --out runs_smoke --eval-every 1
echo "setup done"
