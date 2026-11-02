source /venv/main/bin/activate
cd /root/exp07
RART_PHASES=C RART_JOBS_PER_GPU=1 python -u run_suite.py > suite_C.log 2>&1
