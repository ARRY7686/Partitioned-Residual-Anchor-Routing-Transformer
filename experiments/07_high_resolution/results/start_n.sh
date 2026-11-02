source /venv/main/bin/activate
cd /root/exp07
RART_PHASES=N RART_JOBS_PER_GPU=2 RART_TIME_BUDGET_HOURS=18 python -u run_suite.py > suite_N.log 2>&1
