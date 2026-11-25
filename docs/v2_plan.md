# Paper v2 plan (draft, 2026-11-24)

Goal: a version the TMLR editor would call "up to standard". Two halves: **fix how v1 was run**
(protocol), then **add what v1 is missing** (scale, strong baselines, pretraining).
Target: TMLR resubmission around March 2027; arXiv v2 the same day. v1 still goes to IRIS
(2-3 Jan) and the ICLR 2027 non-archival workshops.

---

## Part A: mistakes in v1 and how v2 fixes them

| # | v1 mistake | Where | v2 fix |
|---|---|---|---|
| M1 | **Test-set peeking**: learning rate 1.25e-4 chosen by comparing test accuracy at epoch 15 | Secs 3-5 (exp 03, 04, 06): no validation split at all | Validation split for every dataset; test touched once, at the end |
| M2 | **Unequal tuning**: ResNet-18 got the better of two learning rates, other models one; recipe fixes (clip, warmup, bf16) found while watching our own runs fail | exp 06, 07 | Same tuning budget for every model: a learning-rate grid fixed in advance, picked on validation |
| M3 | **Post-hoc fixes after seeing results**: focus learning rate 1e-4 tested on seed 42 only; Megapixel MNIST 512^2 picked after a 3-epoch probe; DFG max pooling decided after looking at test mAP | exp 07 | Protocol written and committed *before* runs; any change goes in a deviations log with its reason, and the old result stays in the paper |
| M4 | **Few seeds**: 3 seeds for headline gaps, 1 seed for the sensitivity sweep and several ablations | everywhere | 5 seeds for the headline comparison (focus / fixed / densewin), 3 for everything else, no single-seed claims |
| M5 | **Short, mixed recipes**: 10 vs 30 epochs, fp16 vs bf16, learning rates 3e-4 / 1.25e-4 / 2e-4 across sections; 10-epoch numbers undertrained | Appendix recipes table | One recipe per dataset family (AdamW, bf16, clip 1.0, warmup, cosine), long enough to converge (check: validation curve flat for the last 20%) |
| M6 | **Divergences handled ad hoc** (reruns, recipe changes mid-suite) | exp 06, 07 | Divergence policy fixed in advance: a diverged run counts as a result; report divergence rate per model |
| M7 | **Best-epoch / checkpoint choice unclear** | | Always the final checkpoint (or best-on-validation, decided in advance, same for all) |

### Protocol (to commit as `PROTOCOL.md` before any v2 run)
- Splits: CIFAR-10/100 train 45k / val 5k (fixed indices file); canvas datasets build val
  canvases from the val images; Tiny ImageNet train split 90/10; Megapixel MNIST and DFG keep
  their v1 validation splits (block-held-out for DFG).
- Tuning seed (seed 0) is separate from the test seeds (1-5).
- Learning-rate grid per family, fixed now: transformers {1e-4, 2e-4, 4e-4}; ResNet
  {5e-4, 1e-3, 2e-3}; weight decay 0.05 for all. Pick on validation with seed 0 at the full
  schedule.
- Then train test seeds with the chosen setting and evaluate on test once.
- Statistics: paired differences, 95% t-interval over seeds (as v1), plus divergence counts.

---

## Part B: what v2 adds

| ID | Experiment | Why (who asked) | Priority |
|---|---|---|---|
| E1 | **ImageNet-100 from scratch, 224^2**: focus, fixed, densewin, dense ViT, Swin-T-class, PVTv2, ResNet; matched params/FLOPs; 3 seeds | TMLR editor; all three AI reviews ("ImageNet") | Must |
| E2 | **Published baselines from official code** at matched compute: Swin (timm), PVTv2 (timm), BiFormer (official), RegionViT (official) | reviews: "reimplemented baselines", "BiFormer/VECA" | Must |
| E3 | **Re-run v1 cores under the protocol**: exp 03 collapse bench, exp 06 Cluttered + Tiny canvas, exp 07 Megapixel MNIST + DFG | fixes M1-M7 | Must |
| E4 | **Parallel focus placement** (top-k + NMS instead of the serial loop); check validation accuracy equals the serial version | reviews; also cuts focus cost ~3x | Must (before E1, saves money) |
| E5 | **Pretrained track**: insert the anchor pathway into an ImageNet-pretrained ViT (timm DeiT-S), fine-tune on DFG and Tiny canvas against pretrained Swin/ResNet | "ResNet wins from scratch" limitation | Should |
| E6 | **Soft routing with a diversity regulariser** (orthogonality / spread loss) and **AnchorFormer with query read-back** | "soft routing fails" claim too strong; collapse cause not isolated | Should (cheap, CIFAR-100 bench) |
| E7 | Detection/MIL head on DFG | review 3 | Could (skip unless time) |

---

## Part C: compute and GPU choice

### Live vast.ai prices (on-demand, reliability >= 0.97, queried 2026-11-24 via the offers API)

| GPU | VRAM | cheapest | typical | vast DL-perf / GPU | perf per $ (typical) |
|---|---|---|---|---|---|
| RTX 3090 | 24 GB | $0.12 | $0.20 | 44 | 220 |
| **RTX 5090** | 32 GB | $0.26 | $0.53 (multi-GPU $0.39-0.43) | 199 | **~375** |
| RTX 4090 | 24 GB | $0.36 | $0.47 | 70 | 150 |
| **L40S** | 45 GB | $0.47 (4x/8x boxes, 128 cores) | $0.53 | 93 | 175 |
| A100 40GB PCIe / SXM | 40 GB | $0.38 / $0.41 | $0.62 / $0.80 | 95 | 120-150 |
| RTX PRO 6000 | 96 GB | $1.00 | $1.46 | 283 | 194 |
| H100 NVL / PCIe / SXM | 80-94 GB | $2.34 / $2.47 / $4.05 | | 230-290 | ~100 |
| H200 | 140 GB | $4.61 | $5.72 | 468 | ~80 |

Our models are small (6M parameters, bf16), so speed is set by memory bandwidth, clocks and data
loading, not peak FLOPs; H100/H200 cost 4-8x more per hour for at most 2-3x speed. **RTX 5090 is
the best value** (same Blackwell sm_120 as the laptop RTX 5060, so the code already runs on it).
Its 32 GB is too small for full-HD DFG at batch 32, so DFG goes on **L40S 45 GB** (also cheap, and
sold as 4x/8x boxes with 16 CPU cores per GPU).

### GPU-hours (RTX 5090 / L40S class; refine after the pilot)

| Block | Runs | GPU-h |
|---|---|---|
| E3 collapse bench (9 designs, CIFAR-100 224^2, 30 ep): LR grid 27 + seeds 27 | 54 | ~14 |
| E3 Cluttered CIFAR-10 (9 models, 50 ep): grid 27 + seeds 33 | 60 | ~18 |
| E3 Tiny canvas (6 models, 30 ep): grid 18 + seeds 24 | 42 | ~17 |
| E3 ablations + sensitivity (14 variants x 3 seeds, chosen LR) | 42 | ~10 |
| E6 soft-routing regulariser / AnchorFormer read-back (x 3 seeds) | 18 | ~5 |
| E3 Megapixel MNIST 512^2 (8 models): grid 24 + seeds 30 | 54 | ~27 |
| E3 DFG full HD (7 models): grid 21 + seeds 27 | 48 | ~34 |
| E1+E2 ImageNet-100 (9 models, 100 ep; grid at 30 ep on val; 3-5 seeds) | 60 | ~82 |
| E5 pretrained DeiT-S + anchors vs pretrained baselines (DFG, Tiny canvas) | ~40 | ~35 |
| **Subtotal** | | **~240** |
| Contingency (reruns, debugging) 20% | | ~50 |
| **Total** | | **~290 GPU-h** |

### Rental breakdown (no idle boxes; copy results off before destroying, no volumes)

| Phase | When | Box | Work | Wall time | Cost |
|---|---|---|---|---|---|
| 0 Pilot | Dec | 1x RTX 5090 (~$0.45/h) | throughput of every model on ImageNet-100 and full-HD memory; parallel-placement speed | ~3 h | ~$2 |
| 1 Small tasks | Jan wk 1 | 4x RTX 5090 (~$1.7/h) | collapse bench, Cluttered, Tiny, ablations, E6, Megapixel MNIST (~110 GPU-h with contingency) | ~28 h | ~$48 |
| 2 Full HD | Jan wk 2 | 4x L40S (~$1.9/h) | DFG from scratch + DFG pretrained track (~80 GPU-h) | ~20 h | ~$38 |
| 3 ImageNet-100 | Jan wk 2-3 | 4x RTX 5090, >= 48 CPU cores (~$1.7/h) | ImageNet-100 grid + seeds, pretrained Tiny canvas (~100 GPU-h) | ~25 h | ~$43 |
| **Total** | | | | **~76 h** | **~$130** (range $110-160) |

Money savers: pre-resize ImageNet-100 to fixed-size uint8 arrays (no JPEG decoding, so CPUs stop
being the bottleneck); E4 parallel placement first (focus is ~3x slower than dense with the serial
loop); MPS packing of small jobs as in v1. Interruptible instances are ~50% cheaper but need
mid-run checkpoint resume (v1's runner only resumes whole jobs); worth adding for phases 1 and 3.
Local RTX 5060 (8 GB): smoke tests, E4 correctness, part of E6.

---

## Part D: timeline

| When | What |
|---|---|
| 25 Nov - 20 Dec | Write PROTOCOL.md; build val splits; E4 parallel placement; ImageNet-100 data pipeline (pre-resized shards); add official baselines; local smoke tests; 1-hour A100 pilot for throughput |
| 20 Dec - 3 Jan | IRIS talk prep and presentation (no runs) |
| 4 - 31 Jan | Rent A100s: LR grids on seed 0, then test seeds (E3, E1, E2, E5, E6) |
| Feb | Analysis and rewrite (results section mostly new; keep the collapse story) |
| early Mar | TMLR submission + arXiv v2 |

Adjust when the TMLR editor answers what the main gaps are.

## Decisions (2026-11-24)
1. Scale: ImageNet-100 from scratch **and** the pretrained track (E5).
2. Model size for E1: tiny (~6M, same as v1). Published baselines at their smallest official
   size (DeiT-Ti 5.7M, PVTv2-B0/B1, RegionViT-Ti, BiFormer-T, Swin-T 28M), params and FLOPs reported.
3. GPUs: RTX 5090 boxes for most work, L40S for full HD; ~290 GPU-h, ~$130.

## To do outside the code
- ImageNet-100 data: ImageNet's licence must be accepted with the user's own Hugging Face/ImageNet
  account (I can't do this); class list = the standard CMC ImageNet-100 split.
