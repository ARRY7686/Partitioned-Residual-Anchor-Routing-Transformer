# Experiments

Each experiment has its own folder with a README (question, setup, results, findings), its
notebook, and a `results/` folder split into `tables/` (JSON and Markdown summaries),
`figures/`, `logs/` and, where relevant, `checkpoints/` (not committed).

| # | Experiment | Question | Status | Key result |
|---|---|---|---|---|
| 01 | [Partition RART](01_partition_rart/) | Can anchors replace dense global attention? | Done, 1 seed | Partition RART 44.62% vs ViT 45.44% on CIFAR-100; unrestricted RART-32 anchors collapse (similarity ≈ 0.99) |
| 02 | [Resolution scaling](02_resolution_scaling/) | Does RART scale better than a dense ViT? | Done | Faster than ViT from 1536²; 27× less memory at 2048²; slower than PVTv2 / EfficientViT |
| 03 | [Anchor collapse](03_anchor_collapse/) | Do other anchor designs collapse too? | Done, 3 seeds, incl. faithful AnchorFormer and Representative Attention | RART-style and AnchorFormer anchors collapse (0.99), AnchorFormer despite its mass normalisation; Representative Attention stays partly diverse (0.68). With the same 3×3 conv bypass, the exclusive partition beats Representative Attention by 2.4 points (49.93 vs 47.54) |
| 04 | [Moving anchors](04_moving_anchors/) | Can anchors move to important regions without collapsing? | Done, 3 seeds (`fixed` / `focus`) | `focus` beats `fixed` in 6/6 paired runs: +1.48 (4 anchors) and +3.13 (16 anchors) points on average |
| 05 | [Compute skipping](05_compute_skipping/) | Can learned importance save compute? | First round done, 1 seed | Keeping 25% of patches in the FFN of blocks 3–6 cuts GFLOPs by 25% at no accuracy cost, but random and token-norm selection do as well as learned importance |
| 06 | [Main comparison](06_main_comparison/) | Does focus beat downsampling and standard models at equal compute? Does it hold at 30 epochs and on Tiny ImageNet? Which parts matter? | Done, 3 seeds (A100) | 30 ep: focus 66.82 vs fixed 65.29 (+1.54, CI [0.53, 2.55]) vs dense-with-windows 65.10 vs ViT 61.99; densewin best transformer on Tiny ImageNet; PVTv2 and ResNet-18 far ahead; only hard assignment is both stable and best (softmax, Sinkhorn, Gumbel tested) |
| 07 | [High resolution](07_high_resolution/) | Do anchors help when downsampling destroys the signal, and do they save time or memory against FlashAttention? | Done, 3 seeds (2× A100) | DFG full HD: downsampling halves mAP (15.8 vs ~30); anchors match dense attention at 1/5 of the FLOPs; focus ≈ fixed. Anchors train 1.5–2.3× faster than FlashAttention at full HD / 2048² but never use less memory; ResNet-18 best |

Experiment 06 has its own scripts and settings; see [its README](06_main_comparison/). Every number is collected in [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) by [`summarize_results.py`](summarize_results.py). The paper draft built from 03–07 is in [`../paper/`](../paper/).

Result files inside 03–05 keep the notebooks' internal run names (`step0_…`, `step1_…`).

## Running experiments 03–05

The notebooks for 03 and 04 (which 05 also uses) run on Kaggle (T4) or locally. They download CIFAR, or find it under
`/kaggle/input`, and write to `/kaggle/working` or `./working`. Results are saved to a JSON
file after every model and finished models are skipped on re-run, so a run can be split
across sessions.

| Variable | Meaning | Default |
|---|---|---|
| `RART_MODELS` | Comma-separated models to run | all |
| `RART_SEED` | Random seed | 42 |
| `RART_EPOCHS` | Training epochs | 10 |
| `RART_SMOKE=1` | One epoch on a small subset, to check the pipeline | off |
| `RART_ANCHORS` | Anchor count (03) | 16 |
| `RART_ANCHOR_COUNTS` | Anchor counts, square numbers (04/05) | 4,16 |
| `RART_DATASET` | `cifar10` or `cifar100` (04/05) | cifar10 |
| `RART_GRAD_CLIP` | Gradient-norm clipping threshold, 0 = off (04/05) | 0 |
| `RART_KEEP_RATIO` | Share of patches kept in the FFN from block `RART_SKIP_FROM` on; 1 = no skipping (04/05) | 1.0 |
| `RART_SKIP_FROM` | First block (0-based) where FFN skipping applies (04/05) | 2 |
| `RART_SKIP_SCORE` | How kept patches are chosen: `importance`, `norm` or `random` (04/05) | importance |
| `RART_WORK_DIR` | Output and data folder | see above |

Measured on an RTX 5060 Laptop GPU: about 1.1 min/epoch (03) and 1.4–1.7 min/epoch
(04/05). Model checkpoints (`*.pth`) are not committed.
