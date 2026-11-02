#!/bin/bash
# After the DM queue ends, run the DM phase again: finished runs are skipped, so only the
# failed DM_pvt_seed42 (timm PVTv2 had no max pooling; fixed in train.py) runs.
source /venv/main/bin/activate
cd /root/exp07
while pgrep -f "start_dm.sh" > /dev/null; do sleep 120; done
RART_PHASES=DM RART_JOBS_PER_GPU=1 RART_TIME_BUDGET_HOURS=6 python -u run_suite.py > suite_DM_retry.log 2>&1
