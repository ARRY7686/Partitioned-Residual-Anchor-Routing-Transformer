# 06 — Main comparison: baselines, long training, ablations, skipping

**Question.** Does focus routing beat the obvious alternatives, above all simply downsampling
the image, at equal or lower compute? Does its advantage over fixed anchors survive longer
training, and which parts of the focus design matter?

**Notebook:** [`main-comparison-kaggle.ipynb`](main-comparison-kaggle.ipynb) runs on Kaggle
(GPU T4 ×2). Its first cell writes [`train.py`](train.py), and each job is one run of that
script. `train.py` keeps experiment 04's anchor models unchanged: with default settings,
`focus` and `fixed` are the same code. It adds the baselines, the ablation switches, the
bounded skipping gate, and efficiency measurements.

## Phases

All phases use Cluttered CIFAR-10 at 256² with 3 seeds (42, 43, 44) unless noted.

| Phase | Configs | Epochs | Answers |
|---|---|---:|---|
| A | dense ViT at 256² (patch 16, 256 tokens); ViT on 128² input (patch 16, 64 tokens); ViT on 64² input (patch 8, 64 tokens); PVTv2-B0 | 10 | Does downsampling lose accuracy? How do standard models compare with focus-16 from experiment 04? |
| B | focus-16, focus-4, fixed-16, dense ViT 256², ViT 128² | 30 | Does the focus advantage hold once models are trained longer? This is the headline table. All phase B models use gradient-norm clipping at 1.0 and a 2-epoch linear warmup. With experiment 04's recipe (no clipping, no warmup), 2 of the first 6 runs were knocked off course by gradient spikes: fixed-16 seed 43 fell to chance at epoch 8. Those logs are kept in `logs_unstable_30ep_norecipe/`. |
| C | One untrained model per distinct compute configuration, timed alone on the GPU: GFLOPs, training step time, throughput, peak memory | — | The efficiency table. Compute cost does not depend on the weights, so untrained models give the same numbers as trained ones. |
| D | focus-16 with: no coverage grid (16 focus anchors), 3×3 coverage grid (7 focus), no suppression, no home prior, soft assignment | 10 | Which parts of the focus design matter? |
| F | focus-16 skipping: bounded importance gate at keep 25% (3 seeds) and 50% (seed 42); random and norm at keep 25% (seeds 43, 44) | 10 | Continues experiment 05 with more seeds and a gate whose gradient is bounded |
| T | Tiny ImageNet canvas: focus-16, fixed-16, ViT 256², ViT 128², ViT 64² patch 8 | 15 | Does the result hold when the object has real detail at full resolution? 64 px Tiny ImageNet images are pasted without resizing among native 32 px crops (`RART_DATASET=tinyimagenet`, 200 classes, the 10k validation images as test set). 15 epochs over 100k images is the same number of steps as phase B. Phase B recipe. |
| R | ResNet-18 at lr 1.25e-4 and 1e-3 (30 epochs, and Tiny ImageNet 15 epochs); PVTv2-B0 at 30 epochs | 30 / 15 | Baselines from outside the ViT family (`timm`, no pretraining) |
| D2 | soft assignment, no suppression, and the full model, each with clip 1.0 and 2-epoch warmup | 10 | Phase D's soft-assignment and no-suppression runs diverged: are they unstable or just worse? The full model is the matching reference. |
| F2 | random skipping at keep 25% with clip 1.0 and 2-epoch warmup | 10 | Skipping cost under the stable recipe; the reference is D2's full model. The bounded-gate runs of this phase were stopped at epoch 5 (see results). |
| V | focus-16 with soft assignment balanced by 3 Sinkhorn iterations (`RART_ASSIGN=balanced`, 3 seeds), and a one-at-a-time sweep of β (`RART_SUPPRESSION`), σ_s (`RART_SUPPRESSION_WIDTH`, patches) and σ_h (`RART_HOME_WIDTH`) at seed 42; all with clip 1.0 and 2-epoch warmup. Phases V/VT also run `densewin` (the anchor model with the anchor pathway replaced by global attention among all patches, local windows kept) on Cluttered CIFAR-10 (30 epochs) and the Tiny ImageNet canvas (15 epochs), 3 seeds, on 2× A100. | 10 / 30 / 15 | Review follow-ups: does balanced soft routing train, how sensitive is focus placement, and is the anchors' lead over the dense ViT locality? Reference: D2's full model. |
| W | focus-16 with patches reading from their 2 or 3 nearest anchors (`RART_SCATTER_K`; gathering stays exclusive), and soft Gumbel-softmax assignment with temperature annealed from 1.0 to 0.1 (`RART_ASSIGN=gumbel`); clip 1.0, 2-epoch warmup, 3 seeds | 10 | Second review round. Reference: D2's full model. |

Every run records:
* final test accuracy
* GFLOPs counted two ways: all matmuls including attention (`gflops_total`), and experiment 05's Linear/Conv-only count (`gflops_linear`)
* inference throughput at batch 128 in fp16
* peak training memory

## Precision

All experiment 06 runs on the A100 use **bf16** autocast (`RART_PRECISION=bf16`, no
GradScaler). The first attempts used fp16, as in experiments 03–05, and 3 runs blew up with
non-finite gradients:
* fixed-16 seed 43 collapsed at epoch 6 in fp16, even with clipping and warmup; it had also collapsed at epoch 8 without them
* focus-4 seed 42 fell from 61% to 42% at epoch 21
* the soft-assignment ablation (seed 42) blew up at epoch 5

The same fixed-16 seed 43 configuration in bf16 trained cleanly past epoch 10, with no
overflowed steps and a largest gradient norm of 7.4. The aborted logs are kept in
`results/exp06_results/logs_unstable_30ep_norecipe/`, `logs_unstable_30ep_fp16_clip_warmup/`
and `logs_aborted_fp16/`. The code still defaults to fp16 for the Kaggle T4, which has no
fast bf16.

## Phase B learning rate

In bf16 at experiment 04's learning rate of 3e-4, fixed-16 seed 43 still diverged at epoch 12
with no overflow: one gradient spike sent it to chance. A repeat of the same run survived,
so the divergence is stochastic, roughly 1 in 6 30-epoch runs. At the DeiT-scaled learning
rate 5e-4 × 128/512 = 1.25e-4, the same configuration trained smoothly: largest gradient norm
6.1 over 15 epochs, and 56.5% at epoch 15 against 54.7% at 3e-4. Phase B therefore uses
1.25e-4 for every model. Logs are in `logs_aborted_bf16_lr3e-4/`.

## Data

CIFAR-10 was rebuilt from the Hugging Face mirror `uoft-cs/cifar10` into the original batch
format, because the original server was too slow. The test set is in the original order, so
the fixed test canvases are identical to experiments 04–05. The training set holds the same
50,000 images in a different order. That changes which images share a batch, much as a
different seed would. Every experiment 06 run uses this one copy.

## Caveat on the downsampling baselines

The object is a 32 px CIFAR image upscaled to 64 px. Downsampling the canvas to 128² therefore
returns the object to its native 32 px and loses almost no pixel information; the 64² input
(object at 16 px) is the only lossy setting. In the results the 128² ViT still lost 11.8
points to focus-16, so on this task the loss comes from coarser tokens (the object spans 2×2
patches), not from lost pixels. Phase T was added so that downsampling also loses real detail.

The dense ViT baselines use global attention in every block and no local windows, while the
anchor models also have windowed local attention. `densewin` closes this gap: same blocks and
windows, global attention instead of anchors. On Cluttered CIFAR-10 it explains the fixed
partition's lead over the dense ViT (fixed − densewin +0.19 [−2.71, +3.08]), and focus anchors
stay ahead of it in every seed without a significant gap (+1.72 [−0.80, +4.25]). On the Tiny
ImageNet canvas densewin is the best transformer (20.50, against 17.68 for focus).

## Estimated cost

These times are guesses from local timings, scaled to a T4; the notebook rescales them once
the first jobs finish.

| Phases | GPU time | Wall clock on T4 ×2 |
|---|---:|---:|
| A + B | ~17 h | ~8.5 h (fits in one 12 h session) |
| D + F | ~10 h | ~5 h |

## Running on a rented GPU

[`run_suite.py`](run_suite.py) is the notebook's runner as a script. Run it next to
`train.py`. On one H100 (80 GB; each job needs at most about 5 GB), run several jobs at once:

```bash
RART_JOBS_PER_GPU=6 RART_LONGEST_FIRST=1 RART_TIME_BUDGET_HOURS=100 python run_suite.py
```

Finished jobs are skipped on a re-run. While jobs share a GPU they slow each other down, so
the efficiency table comes from a separate phase C pass, run alone afterwards on the same GPU:

```bash
RART_PHASES=C RART_JOBS_PER_GPU=1 python run_suite.py
```

## Results

All 102 runs of phases A–F2 finished on one vast.ai A100-80GB (2026-10-30/31, bf16); phase V
ran on an RTX 5060 laptop GPU (2026-10-31, bf16, log in `results/exp06_results/suite_V.log`);
the densewin runs (phases V/VT) and phase W ran on 2× A100-80GB (2026-11-02, bf16; logs
`results/suite_VT.log`, `results/suite_W.log`). Mean ± std over seeds
42 / 43 / 44; paired gaps with 95% t-intervals (n = 3). The full per-run table is printed at
the end of each `results/suite_*.log`.

### Main comparison

| Model | Params | GFLOPs | Cluttered, 10 ep | Cluttered, 30 ep | Tiny IN, 15 ep |
|---|---:|---:|---:|---:|---:|
| focus-16 | 11.4M | 5.49 | 49.84 ± 0.26 (exp 04) | **66.82 ± 0.66** | 17.68 ± 0.33 |
| focus-4 | 11.4M | 5.45 | 47.52 ± 1.11 (exp 04) | 64.67 ± 0.26 | — |
| fixed-16 | 11.4M | 5.49 | 46.71 ± 1.72 (exp 04) | 65.29 ± 0.94 | 16.73 ± 0.46 |
| dense with windows | 12.8M | 7.25 | — | 65.10 ± 0.45 | **20.50 ± 0.29** |
| ViT 256² | 9.3M | 5.28 | 46.74 ± 0.42 | 61.99 ± 0.54 | 18.25 ± 0.33 |
| ViT 128² | 9.2M | 1.21 | 43.20 ± 0.22 | 55.05 ± 0.26 | 12.49 ± 0.28 |
| ViT 64², patch 8 | 9.0M | 1.18 | 42.39 ± 0.41 | — | 13.01 ± 0.39 |
| PVTv2-B0 | 3.4M | 1.51 | 71.83 ± 0.30 | 75.07 ± 0.25 | — |
| ResNet-18, lr 1e-3 | 11.2M | 4.74 | — | 86.11 ± 0.26 | 37.07 ± 0.28 |
| ResNet-18, lr 1.25e-4 | 11.2M | 4.74 | — | 74.29 ± 0.79 | 20.16 ± 0.03 |

| Paired gap | Cluttered, 30 ep | Tiny IN, 15 ep |
|---|---:|---:|
| focus-16 − fixed-16 | +1.54 [+0.53, +2.55] | +0.95 [−0.93, +2.83] |
| focus-16 − ViT 256² | +4.83 [+3.86, +5.80] | −0.58 [−2.11, +0.95] |
| fixed-16 − ViT 256² | +3.29 [+1.32, +5.27] | −1.53 [−1.89, −1.16] |
| focus-16 − ViT 128² | +11.77 [+9.73, +13.82] | +5.19 [+4.43, +5.95] |
| focus-16 − dense with windows | +1.72 [−0.80, +4.25] | −2.82 [−3.99, −1.66] |
| fixed-16 − dense with windows | +0.19 [−2.71, +3.08] | −3.77 [−4.65, −2.90] |

Focus beats fixed in every paired seed on both datasets (significantly at 30 epochs on
Cluttered CIFAR-10) and beats the downsampled ViTs by 4.7–11.8 points. Dense attention with the
same local windows is level with the anchors on Cluttered CIFAR-10 and ahead of them on the
Tiny ImageNet canvas, at 7.25 against 5.49 GFLOPs. PVTv2-B0 and ResNet-18 are far ahead in this
from-scratch regime.

### Ablations (focus-16, 10 epochs)

| Variant | Accuracy | Note |
|---|---:|---|
| full (exp 04 recipe) | 49.84 ± 0.26 | |
| no coverage grid | 49.63 ± 0.28 | |
| 3×3 coverage grid | 49.40 ± 0.78 | |
| no home prior | 47.23 ± 1.17 | −2.6 points |
| no suppression | 49.34 / 13.25 / 48.89 | seed 43 diverged |
| soft assignment | 14.34 / 36.17 / 10.00 | diverged (42, 44) or stalled (43) |
| full, clip + warmup (D2) | 53.78 ± 0.09 | |
| no suppression, clip + warmup | 53.57 ± 0.43 | −0.21 [−1.51, +1.08] |
| soft assignment, clip + warmup | 10.00 × 3 | diverged in every seed (grad norm up to 1e14) |

Balancing does not rescue soft assignment. With 3 Sinkhorn iterations (every anchor receives
the same total mass), clip and warmup, the runs peaked at 37.5 / 26.5 / 39.0% (epochs 4 / 2 / 5),
then diverged with gradient norms of 1e11–1e15 and ended at 9.98 / 10.00 / 12.44% (phase V).

Gumbel-softmax assignment with an annealed temperature (phase W) is the only soft variant that
trains: 49.53 and 51.30 for seeds 43 and 44 (−4.3 and −2.5 points against the hard reference),
while seed 42 learned normally to 44.7% at epoch 8 and then diverged as the temperature
approached 0.1 (gradient norms above 1000, final 22.08%).

Hard exclusive assignment is the only variant that is both stable and most accurate;
suppression is a stability device, not an accuracy one.

### Reading from several anchors (phase W, clip + warmup)

Patches read from their k nearest anchors, weighted by the distance softmax; gathering stays
exclusive, so collapse cannot happen.

| Variant | Accuracy | vs D2 full (paired, 95% CI) |
|---|---:|---:|
| own anchor (D2 full) | 53.78 ± 0.09 | |
| 2 nearest anchors | 53.76 ± 0.19 | −0.03 [−0.35, +0.30] |
| 3 nearest anchors | 54.61 ± 0.55 | +0.83 [−0.55, +2.20] |

### Placement sensitivity (phase V, seed 42, clip + warmup)

| Setting | Accuracy | vs D2 full seed 42 (53.68) |
|---|---:|---:|
| β = 3 | 53.98 | +0.30 |
| β = 30 | 49.33 | −4.35 |
| σ_s = 0.75 patches | 53.59 | −0.09 |
| σ_s = 3 patches | 21.88 | −31.80 (stalled near 20% for all 10 epochs) |
| σ_h = 0.15 | 52.90 | −0.78 |
| σ_h = 1.0 | 52.26 | −1.42 |

Defaults are β = 10, σ_s = 1.5 patches, σ_h = 0.35. Placement is robust around them and
breaks when suppression is much stronger or wider.

### FFN skipping (focus-16, keep 25%, 10 epochs)

| Selection | Seeds | Accuracy |
|---|---|---:|
| random | 43 / 44 | 50.15 / 46.76 |
| token norm | 43 / 44 | 52.16 / 50.82 |
| learned importance, bounded gate | 42 / 43 / 44 | 15.67 / 28.50 / 14.27 (diverged) |
| learned importance, bounded gate, keep 50% | 42 | 15.09 (diverged) |
| random, clip + warmup (F2) | 3 | 53.40 ± 0.35 (−0.38 [−1.48, +0.71] vs D2 full) |

The bounded gate `1 + imp − imp.detach()` diverged in every run, also with clip + warmup
(seeds 42/43, stopped at epoch 5 near 20%; partial logs kept). Random skipping costs about
0.4 points for 25% fewer linear-layer GFLOPs.

### Cost at 256² (phase C, A100, bf16, untrained, alone on the GPU)

| Config | GFLOPs | Train step, batch 128 | Peak train memory | Inference | Peak inference memory |
|---|---:|---:|---:|---:|---:|
| focus-16 | 5.49 | 99.3 ms | 4528 MB | 5259 img/s | 767 MB |
| focus-16, keep 25% | 4.13 | 93.0 ms | 4050 MB | 5248 img/s | 767 MB |
| focus-4 | 5.45 | 70.8 ms | 4244 MB | 6207 img/s | 765 MB |
| fixed-16 | 5.49 | 60.7 ms | 4314 MB | 6408 img/s | 766 MB |
| ViT 256² | 5.28 | 32.4 ms | 3111 MB | 11176 img/s | 728 MB |
| ViT 128² | 1.21 | 20.6 ms | 1174 MB | 39339 img/s | 536 MB |
| PVTv2-B0 | 1.51 | 52.4 ms | 4937 MB | 6310 img/s | 1174 MB |

At 256 patches the anchor models are slower and use more memory than dense attention, and FFN
skipping lowers FLOPs but not time. The memory advantage of exclusive routing only appears at
high resolution (see the scaling table in the main README).

## Files

| Path | Contents |
|---|---|
| `train.py`, `run_suite.py` | One training run; the parallel runner with all phases |
| `setup_box.sh`, `build_cifar_from_hf.py`, `build_extra_data.py` | Fresh-box setup: CIFAR-10/100 and Tiny ImageNet from Hugging Face |
| `plot_curves.py` | Training-curve figure for the paper (`paper/figures/curves.pdf`) |
| `results/exp06_results/` | One JSON per run (accuracy history, cost, collapse metrics) and `logs/`; earlier aborted attempts in the `logs_*` folders |
| `results/suite_*.log` | Runner log per phase (`suite_v2.log` = A, B, D, F), each ending with a summary table |
| `results/start*.sh`, `wait_then_r.sh` | The exact launch commands used on the box |
