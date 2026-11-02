source /venv/main/bin/activate
cd /root/exp07
RART_SKIP_TAGS=N_focus_A64_seed43,N_vit128_seed43,N_pvt_seed43,N_resnet18_lr1e-3_seed43,N_focus_A64_seed43,N_focus_A64_seed44 RART_PHASES=N RART_JOBS_PER_GPU=2 RART_TIME_BUDGET_HOURS=17 python -u run_suite.py > suite_N2.log 2>&1
