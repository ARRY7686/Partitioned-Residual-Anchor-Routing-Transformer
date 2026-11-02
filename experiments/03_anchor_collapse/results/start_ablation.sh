#!/bin/bash
# Local-bypass ablation on the collapse benchmark: Representative Attention without its 3x3
# depth-wise conv, and the fixed partition with it. 3 seeds each, two GPUs.
source /venv/main/bin/activate
cd /root/exp03
mkdir -p ablation_local/logs ablation_local/data
ln -sfn /root/exp03/rpattention/data/cifar-100-python ablation_local/data/cifar-100-python
run() { CUDA_VISIBLE_DEVICES=$1 RART_SEED=$2 RART_MODELS=$3 RART_WORK_DIR=/root/exp03/ablation_local \
        python -u anchor_collapse.py > ablation_local/logs/${3}_seed$2.log 2>&1; }
# Different seeds write different result files; the two models of one seed run one after the other.
for spec in "0 42" "1 43" "0 44"; do set -- $spec; gpu=$1; seed=$2
  (run $gpu $seed rpattention_nolocal; run $gpu $seed partition_local) &
  [ $seed = 43 ] && wait
done
wait
echo done > ablation_local/ALL_DONE
