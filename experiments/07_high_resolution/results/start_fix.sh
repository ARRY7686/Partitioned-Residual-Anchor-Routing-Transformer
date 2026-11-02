#!/bin/bash
# Does a lower learning rate or fewer anchors prevent the reproducible seed-42 focus
# instability on Megapixel MNIST 512^2 at 30 epochs? Same settings as phase N otherwise.
source /venv/main/bin/activate
cd /root/exp07
common="RART_DATASET=mpmnist RART_CANVAS=512 RART_NOISE=12 RART_TRAIN_SIZE=10000 RART_FRESH=1 RART_EPOCHS=30 RART_SEED=42 RART_KIND=focus RART_SPIKE_FACTOR=10 RART_OUT_DIR=/root/exp07/exp07_results RART_DATA_DIR=/root/exp07/data"
env $common CUDA_VISIBLE_DEVICES=1 RART_TAG=N_focus_A64_lr1e-4_seed42 RART_ANCHORS=64 RART_LR=1e-4 \
  python -u train.py > exp07_results/logs/N_focus_A64_lr1e-4_seed42.log 2>&1 &
env $common CUDA_VISIBLE_DEVICES=0 RART_TAG=N_focus_A16_seed42 RART_ANCHORS=16 RART_LR=2e-4 \
  python -u train.py > exp07_results/logs/N_focus_A16_seed42.log 2>&1 &
wait
echo done > fix_done
