#!/bin/bash
# Restart phase N without focus seeds 43/44 (and without the jobs still running), then run
# three 15-epoch focus diagnostics at 512²: spike logging, 16 anchors, lower learning rate.
source /venv/main/bin/activate
cd /root/exp07
running=""
for pid in $(pgrep -f "python train.py"); do
  tag=$(tr "\0" "\n" < /proc/$pid/environ 2>/dev/null | grep "^RART_TAG=" | cut -d= -f2)
  [ -n "$tag" ] && running="$running$tag,"
done
for pid in $(pgrep -f "python -u run_suite.py"); do kill $pid; done
sleep 2
echo "left running: $running"
cat > start_n2.sh <<EOS
source /venv/main/bin/activate
cd /root/exp07
RART_SKIP_TAGS=${running}N_focus_A64_seed43,N_focus_A64_seed44 RART_PHASES=N RART_JOBS_PER_GPU=2 RART_TIME_BUDGET_HOURS=17 python -u run_suite.py > suite_N2.log 2>&1
EOS
(nohup bash start_n2.sh > /dev/null 2>&1 < /dev/null &)
mkdir -p diag
common="RART_DATASET=mpmnist RART_CANVAS=512 RART_NOISE=12 RART_TRAIN_SIZE=10000 RART_FRESH=1 RART_EPOCHS=15 RART_SEED=42 RART_KIND=focus RART_SPIKE_FACTOR=10 RART_OUT_DIR=/root/exp07/diag RART_DATA_DIR=/root/exp07/data"
for spec in "0 diag_A64_lr2e-4 64 2e-4" "1 diag_A16_lr2e-4 16 2e-4" "0 diag_A64_lr1e-4 64 1e-4"; do
  set -- $spec
  env $common CUDA_VISIBLE_DEVICES=$1 RART_TAG=$2 RART_ANCHORS=$3 RART_LR=$4 nohup python -u train.py > diag/$2.log 2>&1 < /dev/null &
done
echo launched
