# Paper v2 evaluation protocol

Written 2026-11-25, **before any v2 training run**, and committed before the first one. Its
commit hash is recorded with every test-set evaluation (`experiments/09_imagenet/TEST_ACCESS_LOG.jsonl`).
Anything done differently from this file goes in the deviations log at the bottom, with the date
and reason, and the original plan's result (if one exists) stays in the paper.

Why this exists: v1 chose a learning rate by comparing test accuracy, tuned models unequally,
and made recipe and dataset choices after looking at results (mistakes M1-M7 in `docs/v2_plan.md`).

## 1. Rules for every experiment

1. **Three splits.** Every dataset has train / validation / test. All decisions (learning rate,
   design choices, recipe fixes, which variant goes in the paper) use validation only.
2. **Test is opened once per final run**, after every choice that depends on validation has been
   made and written down. For ImageNet the only code that reads test is
   `experiments/09_imagenet/evaluate_test.py`, which refuses tuning, smoke and unfinished runs
   and logs every access.
3. **Equal tuning.** Every model gets the same learning-rate grid (per family, below), the same
   number of tuning runs, the same recipe and the same epochs. No model gets an extra knob.
4. **Tuning seed 0 is separate from the reported seeds** (1, 2, 3, and 4, 5 for headline gaps).
   Seed 0 runs are never reported as results.
5. **Final checkpoint, raw weights.** The result of a run is its last checkpoint (no best-epoch
   picking, no EMA).
6. **Divergence policy.** A run whose training loss becomes non-finite is "diverged". It is not
   rerun and not repaired; it counts as a result. Per model we report the number of diverged
   runs. Accuracy means are over finished runs, with the count shown; a model with more than one
   diverged seed is flagged in the table. In the grid, a diverged learning rate is not eligible.
   A run finishing at or below twice chance accuracy is reported as "failed to train" the same way.
7. **Statistics.** Paired by seed (the same seed gives every model the same data order and
   augmentation seeds). We report means, and paired differences with 95% t-intervals over
   seeds. No claim rests on one seed.
8. **Pre-resolved ties.** In the grid, the highest validation top-1 wins; if two learning rates
   are within 0.2 points, take the smaller.
9. **Exploratory runs are labelled.** Pilot and smoke runs may look at validation freely. They
   are never reported as results, and they never touch test.

## 2. ImageNet-100 / ImageNet-1k (experiments E1, E2)

**Data** (`experiments/09_imagenet/prepare_data.py`, source: Hugging Face `ILSVRC/imagenet-1k`).
- ImageNet-100 = the standard CMC 100-class subset (`imagenet100_classes.txt`, Tian et al. 2019).
- train = ILSVRC-2012 train minus validation; **validation = 50 images per class held out from
  ILSVRC-2012 train** (10 per class for ImageNet-1k), chosen by the fixed seed `SPLIT_SEED =
  20261125` from the file names. The held-out file list (`val.files.txt`) is committed.
- **test = the official ILSVRC-2012 validation set** (50 per class).
- Images with a short side above 288 px are resized to 288 (Lanczos, JPEG quality 95), and
  smaller ones keep their original bytes. The same files are used for every model.

**Recipe** (DeiT / PVTv2, `experiments/09_imagenet/train.py` defaults): 224 px; AdamW with
weight decay 0.05 (none on biases, norms or learned tokens), batch 512; 5 warmup epochs then
cosine to 1e-5; gradient clipping at 1.0; bf16 autocast; label smoothing 0.1; Mixup 0.8 and
CutMix 1.0; RandAugment `rand-m9-mstd0.5-inc1`; random erasing 0.25; drop-path 0.1. Evaluation
is resize 256, centre crop 224. Differences from the PVTv2 paper (batch 512 vs 1024, clipping,
100 vs 300 epochs) apply equally to every model.

**Learning-rate grid** (peak LR at batch 512), fixed now:
- transformers and anchor models: {2.5e-4, 5e-4, 1e-3}
- ResNet: {5e-4, 1e-3, 2e-3}

The grid is run with seed 0 for 30 epochs; the winner by validation (rule 8) is used for 100-epoch
runs on seeds 1-3 (1-5 for the headline comparison). The 30-epoch grid is the same for all models.

**Models.** All are at their smallest official size, with parameters and GFLOPs reported.
- Headline (5 seeds): `pvt_v2_b0`, `anchor_b0_fixed`, `anchor_b0_local`.
- Baselines (3 seeds): `timm:deit_tiny_patch16_224`, `timm:resnet18`, `pvt_v2_b1`,
  `anchor_b1_fixed`, `anchor_b1_local`, Swin-T (`timm:swin_tiny_patch4_window7_224`), BiFormer-T
  and RegionViT-Ti from official code, and v1's focus / fixed / densewin at 224 px.

Models whose code is not ready yet may be added, but the list is not shortened after any test
evaluation. `anchor_b0_focus` (global placement) and the read-all variant are ablations
(3 seeds, chosen LR of their parent).

**Metrics:** test top-1 (primary), top-5, validation top-1, training throughput (images/s on
the stated GPU), peak memory, params, GFLOPs at 224, plus GFLOPs and step time at 1024² and
1920×1152 from `experiments/08_efficient_routing/compare_v2.py`.

## 3. v1 experiments re-run under this protocol (E3, E6)

Splits, fixed before any run:
- CIFAR-10/100: train 45k / val 5k (fixed index file); the canvas datasets build their
  validation canvases from validation images only.
- Tiny ImageNet: train 90/10 by a fixed seed; the official val set is test.
- Megapixel MNIST and DFG keep v1's validation splits (block-held-out for DFG).

The transformer grid is {1e-4, 2e-4, 4e-4} and the ResNet grid {5e-4, 1e-3, 2e-3}, with weight
decay 0.05, chosen on seed 0 at the full schedule and reported on seeds 1-3 (1-5 for the focus /
fixed / densewin headline). There is one recipe per dataset family (AdamW, bf16, clip 1.0,
warmup, cosine), checked for convergence (validation curve flat over the last 20% of training).
The full per-experiment settings are added here before those runs start (an addition, not a
deviation).

## 4. Deviations log

| Date | What changed | Why | Runs affected |
|---|---|---|---|
| 2026-11-25 | (before any run) ImageNet stored as pre-resized JPEG (short side 288) instead of the fixed-size uint8 arrays in `v2_plan.md` | Keeps the exact DeiT/PVT augmentation (RandomResizedCrop over the whole image); the same format scales to ImageNet-1k (~250 GB as raw arrays) | none |
