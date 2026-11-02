source /venv/main/bin/activate
cd /root/exp06
RART_PRECISION=bf16 RART_PHASES=V,VT RART_JOBS_PER_GPU=1 RART_TIME_BUDGET_HOURS=100 python -u run_suite.py > suite_VT.log 2>&1
