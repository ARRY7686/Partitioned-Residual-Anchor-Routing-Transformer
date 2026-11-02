source /venv/main/bin/activate
cd /root/exp07
RART_PHASES=M,D RART_JOBS_PER_GPU=1 RART_TIME_BUDGET_HOURS=23 python -u run_suite.py > suite_MD.log 2>&1
