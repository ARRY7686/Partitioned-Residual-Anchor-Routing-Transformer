"""Experiment 06 job runner: the same code as the Kaggle notebook's runner cells, as a script.

Run next to train.py. Settings: see the RART_* variables at the top.
"""
import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import torch

# Defaults suit Kaggle (T4 x2, 12 h sessions). On a rented GPU, e.g. one H100:
#   RART_PHASES=A,B,D,F RART_JOBS_PER_GPU=6 RART_TIME_BUDGET_HOURS=100 RART_LONGEST_FIRST=1 python run_suite.py
PHASES = os.environ.get("RART_PHASES", "A,B,D,F").split(",")
TIME_BUDGET_HOURS = float(os.environ.get("RART_TIME_BUDGET_HOURS", "11"))
JOBS_PER_GPU = int(os.environ.get("RART_JOBS_PER_GPU", "1"))      # concurrent jobs sharing one GPU
LONGEST_FIRST = os.environ.get("RART_LONGEST_FIRST", "0") == "1"  # start long jobs first (no time limit)
SKIP_TAGS = set(filter(None, os.environ.get("RART_SKIP_TAGS", "").split(",")))  # e.g. jobs running elsewhere
SEEDS = [42, 43, 44]
OUT_DIR = "/kaggle/working/exp06_results" if os.path.isdir("/kaggle/working") else os.path.abspath("exp06_results")
LOG_DIR = os.path.join(OUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Rough minutes per epoch on a T4, used only to decide whether a job still fits in the
# time budget; rescaled by the measured speed once jobs finish.
MINUTES_PER_EPOCH = {"router": 2.5, "vit256": 1.8, "pvt": 1.8, "vit_small": 0.6}


def job(tag, epochs, rate, save=False, **settings):
    env = {"RART_TAG": tag, "RART_EPOCHS": str(epochs), **{f"RART_{k.upper()}": str(v) for k, v in settings.items()}}
    if save:
        env["RART_SAVE_CHECKPOINT"] = "1"
    return {"tag": tag, "env": env, "estimate": epochs * MINUTES_PER_EPOCH[rate]}


def phase_jobs(phase):
    jobs = []
    if phase == "C":
        # Efficiency pass: one untrained model per distinct compute configuration. Run with
        # RART_JOBS_PER_GPU=1 so no other job shares the GPU while it is timed.
        configs = {
            "vit256": dict(kind="vit", vit_input=256, vit_patch=16),
            "vit128": dict(kind="vit", vit_input=128, vit_patch=16),
            "vit64p8": dict(kind="vit", vit_input=64, vit_patch=8),
            "pvt": dict(kind="pvt"),
            "focus_A16": dict(kind="focus", anchors=16),
            "focus_A4": dict(kind="focus", anchors=4),
            "fixed_A16": dict(kind="fixed", anchors=16),
            "fixed_A4": dict(kind="fixed", anchors=4),
            "focus_A16_keep0.25": dict(kind="focus", anchors=16, keep_ratio=0.25, skip_score="importance", gate="bounded"),
            "focus_A16_keep0.5": dict(kind="focus", anchors=16, keep_ratio=0.5, skip_score="importance", gate="bounded"),
        }
        return [job(f"C_{name}", 0, "vit_small", measure_only=1, **settings) for name, settings in configs.items()]
    for seed in SEEDS:
        if phase == "A":
            jobs += [
                job(f"A_vit256_seed{seed}", 10, "vit256", kind="vit", vit_input=256, vit_patch=16, seed=seed),
                job(f"A_vit128_seed{seed}", 10, "vit_small", kind="vit", vit_input=128, vit_patch=16, seed=seed),
                job(f"A_vit64p8_seed{seed}", 10, "vit_small", kind="vit", vit_input=64, vit_patch=8, seed=seed),
                job(f"A_pvt_seed{seed}", 10, "pvt", kind="pvt", seed=seed),
            ]
        elif phase == "B":
            save = seed == 42
            # Same recipe for every phase B model: clip 1.0, 2-epoch warmup and the DeiT-scaled
            # learning rate 5e-4 x 128/512 = 1.25e-4. At 3e-4, about 1 in 6 30-epoch runs diverged.
            recipe = dict(seed=seed, grad_clip=1.0, warmup_epochs=2, lr=1.25e-4)
            jobs += [
                job(f"B_focus_A16_seed{seed}", 30, "router", save, kind="focus", anchors=16, **recipe),
                job(f"B_focus_A4_seed{seed}", 30, "router", save, kind="focus", anchors=4, **recipe),
                job(f"B_fixed_A16_seed{seed}", 30, "router", save, kind="fixed", anchors=16, **recipe),
                job(f"B_vit256_seed{seed}", 30, "vit256", kind="vit", vit_input=256, vit_patch=16, **recipe),
                job(f"B_vit128_seed{seed}", 30, "vit_small", kind="vit", vit_input=128, vit_patch=16, **recipe),
            ]
        elif phase == "D":
            base = dict(kind="focus", anchors=16, seed=seed)
            jobs += [
                job(f"D_nocoverage_seed{seed}", 10, "router", coverage_side=0, **base),
                job(f"D_coverage3x3_seed{seed}", 10, "router", coverage_side=3, **base),
                job(f"D_nosuppression_seed{seed}", 10, "router", suppression=0, **base),
                job(f"D_nohome_seed{seed}", 10, "router", home_width="inf", **base),
                job(f"D_softassign_seed{seed}", 10, "router", assign="soft", **base),
            ]
    if phase == "F":
        base = dict(kind="focus", anchors=16)
        jobs += [job(f"F_keep0.25_importance-bounded_seed{s}", 10, "router", keep_ratio=0.25,
                     skip_score="importance", gate="bounded", seed=s, **base) for s in SEEDS]
        jobs += [job("F_keep0.5_importance-bounded_seed42", 10, "router", keep_ratio=0.5,
                     skip_score="importance", gate="bounded", seed=42, **base)]
        jobs += [job(f"F_keep0.25_{score}_seed{s}", 10, "router", keep_ratio=0.25, skip_score=score, seed=s, **base)
                 for score in ("random", "norm") for s in (43, 44)]
    return jobs


# Results from an earlier session (attached as notebook output) count as done.
for path in glob.glob("/kaggle/input/**/exp06_results/*.json", recursive=True):
    target = os.path.join(OUT_DIR, os.path.basename(path))
    if not os.path.exists(target):
        shutil.copy(path, target)
        print("Restored", os.path.basename(path))

jobs = [j for phase in PHASES for j in phase_jobs(phase)]
pending = [j for j in jobs if not os.path.exists(os.path.join(OUT_DIR, j["tag"] + ".json")) and j["tag"] not in SKIP_TAGS]
if "C" in PHASES and JOBS_PER_GPU > 1:
    raise ValueError("Run phase C on its own with RART_JOBS_PER_GPU=1, so timings are not shared")
if LONGEST_FIRST:
    pending.sort(key=lambda j: -j["estimate"])
gpus = list(range(torch.cuda.device_count())) or [None]
print(f"{len(jobs)} jobs in phases {PHASES}: {len(jobs) - len(pending)} done, {len(pending)} to run on GPUs {gpus}, {JOBS_PER_GPU} per GPU")
print(f"Estimated GPU time left: {sum(j['estimate'] for j in pending) / 60:.1f} h "
      f"(T4 speed; ~{sum(j['estimate'] for j in pending) / 60 / len(gpus):.1f} h wall clock on T4 x{len(gpus)})")


start = time.time()
work = queue.Queue()
for j in pending:
    work.put(j)
lock = threading.Lock()
speed = {"measured": 0.0, "estimated": 0.0}   # minutes, for rescaling the estimates
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
        env = {**os.environ, **j["env"], "RART_OUT_DIR": OUT_DIR,
               "RART_DATA_DIR": DATA_DIR}
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
                    acc = json.load(f)["final_test_acc"]
                speed["measured"] += minutes
                speed["estimated"] += j["estimate"]
                finished.append(j["tag"])
                print(f"[{elapsed_minutes() / 60:5.2f} h] GPU {gpu} done {j['tag']}: {"measured" if acc is None else f"{acc:.2f}%"} in {minutes:.0f} min "
                      f"({len(finished)}/{len(pending)})", flush=True)
            else:
                failed.append(j["tag"])
                print(f"[{elapsed_minutes() / 60:5.2f} h] GPU {gpu} FAILED {j['tag']} (exit {code}); see logs/{j['tag']}.log",
                      flush=True)


# Download CIFAR-10 once before the workers start, so two processes do not race.
import torchvision
# (skipped when the batch files are already there, e.g. a copy rebuilt from a mirror).
DATA_DIR = "/kaggle/working/data" if os.path.isdir("/kaggle/working") else os.path.abspath("data")
if not os.path.isdir(os.path.join(DATA_DIR, "cifar-10-batches-py")):
    torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
    torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)

threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus for _ in range(JOBS_PER_GPU)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"\nFinished {len(finished)}, failed {len(failed)}, not started (time budget) {len(skipped)} "
      f"in {elapsed_minutes() / 60:.2f} h")
if failed:
    print("Failed:", failed)
if skipped:
    print("Run the notebook again with this version's output attached to continue:", skipped)


import re
from collections import defaultdict

import numpy as np

groups = defaultdict(list)
measured = []
for path in sorted(glob.glob(os.path.join(OUT_DIR, "*.json"))):
    with open(path) as f:
        r = json.load(f)
    if r.get("measure_only"):
        measured.append(r)
    else:
        groups[re.sub(r"_seed\d+$", "", r["tag"])].append(r)

lines = ["| Config | Seeds | Epochs | Params | Test acc (mean ± std) | GFLOPs (all) | GFLOPs (linear) | "
         "Throughput (img/s)* | Peak train memory (MB) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for name, runs in groups.items():
    accs = np.array([r["final_test_acc"] for r in runs])
    std = f" ± {accs.std(ddof=1):.2f}" if len(accs) > 1 else ""
    first = runs[0]
    lines.append(f"| {name} | {len(runs)} | {first['epochs']} | {first['params'] / 1e6:.2f}M | {accs.mean():.2f}{std} | "
                 f"{first['gflops_total']:.2f} | {first['gflops_linear']:.2f} | "
                 f"{np.mean([r['throughput_img_s'] or 0 for r in runs]):.0f} | "
                 f"{np.mean([r['peak_train_memory_mb'] or 0 for r in runs]):.0f} |")
table = "\n".join(lines)
print(table)
with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
    f.write("# Experiment 06 summary\n\nFinal-epoch test accuracy. *Throughput is inference at batch 128, fp16, "
            "measured right after training, while other jobs may share the GPU; use the phase C table for "
            f"efficiency. GPU: {runs[0]['gpu'] if groups else '?'}.\n\n" + table + "\n")

if measured:
    eff = ["| Config | Params | GFLOPs (all) | Train step, batch 128 (ms) | Peak train memory (MB) | "
           "Inference (img/s) | Peak inference memory (MB) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in measured:
        eff.append(f"| {r['tag'][2:]} | {r['params'] / 1e6:.2f}M | {r['gflops_total']:.2f} | {r['train_step_ms']:.1f} | "
                   f"{r['peak_train_memory_mb'] or 0:.0f} | {r['throughput_img_s'] or 0:.0f} | {r['peak_inference_memory_mb'] or 0:.0f} |")
    print("\n" + "\n".join(eff))
    with open(os.path.join(OUT_DIR, "summary.md"), "a", encoding="utf-8") as f:
        f.write(f"\n## Efficiency (phase C, one job per GPU, {measured[0]['gpu']})\n\n" + "\n".join(eff) + "\n")
