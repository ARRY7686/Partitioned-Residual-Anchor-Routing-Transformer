"""Experiment 07 job runner (same design as experiment 06's). Run next to train.py.

    python build_data.py mnist && python build_data.py dfg
    RART_PHASES=C RART_JOBS_PER_GPU=1 python run_suite.py          # cost sweep first, alone
    RART_PHASES=M,D RART_JOBS_PER_GPU=2 python run_suite.py        # training runs

Phases:
  M  Megapixel MNIST 1024², 3 seeds: focus/fixed (64 anchors), densewin, dense ViT at native
     resolution and on 512² / 256² inputs (digits at 14 / 7 px), PVTv2-B0, ResNet-18
  D  DFG full HD (1920×1152 padded), 3 seeds: the same models; ViTs on 960- and 480-wide inputs
  C  cost of untrained models from 512² to 2048² (and full HD): fused (flash) vs plain (math)
     attention for the dense ViT, against the anchor models; out-of-memory is recorded
"""
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import torch

PHASES = os.environ.get("RART_PHASES", "M,D").split(",")
TIME_BUDGET_HOURS = float(os.environ.get("RART_TIME_BUDGET_HOURS", "100"))
JOBS_PER_GPU = int(os.environ.get("RART_JOBS_PER_GPU", "1"))
SKIP_TAGS = set(filter(None, os.environ.get("RART_SKIP_TAGS", "").split(",")))
SEEDS = [int(s) for s in os.environ.get("RART_SEEDS", "42,43,44").split(",")]
EPOCHS = {"mpmnist": int(os.environ.get("RART_EPOCHS_M", "30")), "dfg": int(os.environ.get("RART_EPOCHS_D", "20"))}
OUT_DIR = os.path.abspath(os.environ.get("RART_SUITE_OUT", "exp07_results"))
DATA_DIR = os.path.abspath(os.environ.get("RART_DATA_DIR", "data"))
LOG_DIR = os.path.join(OUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Rough A100 minutes per epoch, only for the time budget; rescaled once jobs finish.
MINUTES_PER_EPOCH = {"mpmnist": 2.0, "dfg": 6.0}

MODELS = {
    "mpmnist": {"focus_A64": dict(kind="focus", anchors=64), "fixed_A64": dict(kind="fixed", anchors=64),
                "densewin": dict(kind="densewin"), "vit_native": dict(kind="vit", vit_input=0),
                "vit512": dict(kind="vit", vit_input=512), "vit256": dict(kind="vit", vit_input=256),
                "pvt": dict(kind="pvt"), "resnet18_lr1e-3": dict(kind="resnet", lr="1e-3")},
    "dfg": {"focus_A64": dict(kind="focus", anchors=64), "fixed_A64": dict(kind="fixed", anchors=64),
            "densewin": dict(kind="densewin"), "vit_native": dict(kind="vit", vit_input=0),
            "vit960": dict(kind="vit", vit_input=960), "vit480": dict(kind="vit", vit_input=480),
            "pvt": dict(kind="pvt"), "resnet18_lr1e-3": dict(kind="resnet", lr="1e-3")},
}


def job(tag, dataset, epochs, **settings):
    env = {"RART_TAG": tag, "RART_DATASET": dataset, "RART_EPOCHS": str(epochs),
           **{f"RART_{k.upper()}": str(v) for k, v in settings.items()}}
    return {"tag": tag, "env": env, "estimate": max(1, epochs) * MINUTES_PER_EPOCH[dataset]}


def phase_jobs(phase):
    if phase in ("M", "D"):
        dataset = "mpmnist" if phase == "M" else "dfg"
        return [job(f"{phase}_{name}_seed{seed}", dataset, EPOCHS[dataset], seed=seed, **settings)
                for seed in SEEDS for name, settings in MODELS[dataset].items()]
    if phase == "C":
        models = {"vit_flash": dict(kind="vit", vit_input=0, sdpa="flash"),
                  "vit_math": dict(kind="vit", vit_input=0, sdpa="math"),
                  "densewin_flash": dict(kind="densewin", sdpa="flash"),
                  "fixed_A64": dict(kind="fixed", anchors=64), "focus_A64": dict(kind="focus", anchors=64)}
        jobs = [job(f"C_{name}_{canvas}_b{batch}", "mpmnist", 0, canvas=canvas, measure_only=1, measure_batch=batch,
                    **settings)
                for canvas in (512, 1024, 1536, 2048) for batch in (1, 8) for name, settings in models.items()]
        jobs += [job(f"C_dfg_{name}_b{batch}", "dfg", 0, measure_only=1, measure_batch=batch, **settings)
                 for batch in (1, 8) for name, settings in models.items()]
        return jobs
    raise ValueError(f"unknown phase {phase}")


jobs = [j for phase in PHASES for j in phase_jobs(phase)]
pending = [j for j in jobs if not os.path.exists(os.path.join(OUT_DIR, j["tag"] + ".json")) and j["tag"] not in SKIP_TAGS]
if "C" in PHASES and JOBS_PER_GPU > 1:
    raise ValueError("Run phase C on its own with RART_JOBS_PER_GPU=1, so timings are not shared")
pending.sort(key=lambda j: -j["estimate"])   # long jobs first
gpus = list(range(torch.cuda.device_count())) or [None]
print(f"{len(jobs)} jobs in phases {PHASES}: {len(jobs) - len(pending)} done, {len(pending)} to run on GPUs {gpus}, "
      f"{JOBS_PER_GPU} per GPU; estimated {sum(j['estimate'] for j in pending) / 60:.1f} GPU-hours", flush=True)

start = time.time()
work = queue.Queue()
for j in pending:
    work.put(j)
lock = threading.Lock()
speed = {"measured": 0.0, "estimated": 0.0}
finished, failed, skipped = [], [], []


def elapsed_minutes():
    return (time.time() - start) / 60


def worker(gpu):
    while True:
        try:
            j = work.get_nowait()
        except queue.Empty:
            return
        with lock:
            scale = speed["measured"] / speed["estimated"] if speed["estimated"] else 1.0
        if elapsed_minutes() + j["estimate"] * scale > TIME_BUDGET_HOURS * 60:
            with lock:
                skipped.append(j["tag"])
            continue
        env = {**os.environ, **j["env"], "RART_OUT_DIR": OUT_DIR, "RART_DATA_DIR": DATA_DIR}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        job_start = time.time()
        with open(os.path.join(LOG_DIR, j["tag"] + ".log"), "w") as log:
            code = subprocess.run([sys.executable, "train.py"], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        minutes = (time.time() - job_start) / 60
        result_path = os.path.join(OUT_DIR, j["tag"] + ".json")
        with lock:
            if code == 0 and os.path.exists(result_path):
                with open(result_path) as f:
                    score = json.load(f)["final_test_acc"]
                speed["measured"] += minutes
                speed["estimated"] += j["estimate"]
                finished.append(j["tag"])
                shown = "measured" if score is None else f"{score:.2f}"
                print(f"[{elapsed_minutes() / 60:5.2f} h] GPU {gpu} done {j['tag']}: {shown} in {minutes:.0f} min "
                      f"({len(finished)}/{len(pending)})", flush=True)
            else:
                failed.append(j["tag"])
                print(f"[{elapsed_minutes() / 60:5.2f} h] GPU {gpu} FAILED {j['tag']} (exit {code}); "
                      f"see logs/{j['tag']}.log", flush=True)


threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus for _ in range(JOBS_PER_GPU)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"\nFinished {len(finished)}, failed {len(failed)}, not started (time budget) {len(skipped)} "
      f"in {elapsed_minutes() / 60:.2f} h")
if failed:
    print("Failed:", failed)

# Summary: mean ± std over seeds of the final validation and test scores.
groups = defaultdict(list)
measured = []
for path in sorted(glob.glob(os.path.join(OUT_DIR, "*.json"))):
    with open(path) as f:
        r = json.load(f)
    if r.get("measure_only"):
        measured.append(r)
    elif "_seed" in r["tag"]:
        groups[r["tag"].rsplit("_seed", 1)[0]].append(r)
if groups:
    print(f"\n{'model':28s} {'seeds':>5s} {'val':>14s} {'test':>14s} {'GFLOPs':>8s} {'min':>6s}")
    for name, runs in sorted(groups.items()):
        val = np.array([r["final_val_score"] for r in runs])
        test = np.array([r["final_test_score"] for r in runs])
        print(f"{name:28s} {len(runs):5d} {val.mean():7.2f} ± {val.std(ddof=1) if len(runs) > 1 else 0:4.2f} "
              f"{test.mean():7.2f} ± {test.std(ddof=1) if len(runs) > 1 else 0:4.2f} "
              f"{runs[0]['gflops_total']:8.1f} {np.mean([r['total_minutes'] for r in runs]):6.0f}")
if measured:
    print(f"\n{'measured':32s} {'GFLOPs':>8s} {'ms/step':>8s} {'img/s':>8s} {'train MB':>9s} {'infer MB':>9s}")
    for r in measured:
        step = "OOM" if r.get("train_oom") else f"{r.get('train_step_ms', float('nan')):.1f}"
        print(f"{r['tag']:32s} {r.get('gflops_total') or float('nan'):8.1f} {step:>8s} "
              f"{r.get('throughput_img_s') or float('nan'):8.1f} {r.get('peak_train_memory_mb') or float('nan'):9.0f} "
              f"{r.get('peak_inference_memory_mb') or float('nan'):9.0f}")
