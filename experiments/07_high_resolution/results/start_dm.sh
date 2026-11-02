#!/bin/bash
# Waits until the focus diagnostics and the DFG sanity run have finished, then runs the DFG
# max-pool comparison and the phase N focus reruns, one job per GPU.
source /venv/main/bin/activate
cd /root/exp07
busy() {
  for pid in $(pgrep -f "python"); do
    tag=$(tr "\0" "\n" < /proc/$pid/environ 2>/dev/null | grep "^RART_TAG=" | cut -d= -f2)
    case "$tag" in diag_*|sanity_*) return 0;; esac
  done
  return 1
}
while busy; do sleep 60; done
echo "$(date -u +%H:%M) diagnostics finished, starting DM,NF" > dm_launch.log
RART_PHASES=NF,DM RART_JOBS_PER_GPU=1 RART_TIME_BUDGET_HOURS=16 python -u run_suite.py > suite_DM.log 2>&1
