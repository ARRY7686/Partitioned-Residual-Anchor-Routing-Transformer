source /venv/main/bin/activate
cd /root/exp06
RART_PRECISION=bf16 RART_PHASES=W RART_JOBS_PER_GPU=1 RART_TIME_BUDGET_HOURS=14 python -u run_suite.py > suite_W.log 2>&1
