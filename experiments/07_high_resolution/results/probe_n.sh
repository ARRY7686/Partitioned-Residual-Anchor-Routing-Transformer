#!/bin/bash
# Stop the queue runner and the stuck Megapixel MNIST job; leave the DFG probes running.
source /venv/main/bin/activate
cd /root/exp07
for pid in $(pgrep -f "python -u run_suite.py"); do kill $pid; done
for pid in $(pgrep -f "python train.py"); do
  tag=$(tr "\0" "\n" < /proc/$pid/environ 2>/dev/null | grep "^RART_TAG=" | cut -d= -f2)
  case "$tag" in M_*) kill $pid; echo "killed $tag";; esac
done
sleep 3
mkdir -p probe
for spec in "0 focus 0 64" "1 vit 128 64"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$1 RART_TAG=probe_N_$2$3 RART_KIND=$2 RART_VIT_INPUT=$3 RART_ANCHORS=$4 RART_DATASET=mpmnist \
    RART_CANVAS=512 RART_NOISE=12 RART_TRAIN_SIZE=10000 RART_FRESH=1 RART_EPOCHS=3 RART_OUT_DIR=/root/exp07/probe \
    RART_DATA_DIR=/root/exp07/data nohup python -u train.py > probe/probe_N_$2$3.log 2>&1 < /dev/null &
done
echo launched
