# Follow-up experiments

These experiments test whether anchor collapse is a general property of anchor bottlenecks,
and whether anchors can move to important regions without collapsing. Each folder holds a
self-contained notebook and its results.

| Folder | Question | Status |
|---|---|---|
| [`step0_anchor_collapse/`](step0_anchor_collapse/) | Do other anchor designs collapse like RART-32? | Done, 2 seeds |
| [`step1_moving_anchors/`](step1_moving_anchors/) | Can anchors move to important regions without collapsing? | Done, 1 seed |

## Running

Both notebooks run on Kaggle (T4) or locally. They download CIFAR, or find it under
`/kaggle/input`, and write to `/kaggle/working` or `./working`. Results are saved to a JSON
file after every model and finished models are skipped on re-run, so a run can be split
across sessions.

| Variable | Meaning | Default |
|---|---|---|
| `RART_MODELS` | Comma-separated models to run | all |
| `RART_SEED` | Random seed | 42 |
| `RART_EPOCHS` | Training epochs | 10 |
| `RART_SMOKE=1` | One epoch on a small subset, to check the pipeline | off |
| `RART_ANCHORS` | Anchor count (step 0) | 16 |
| `RART_ANCHOR_COUNTS` | Anchor counts, square numbers (step 1) | 4,16 |
| `RART_DATASET` | `cifar10` or `cifar100` (step 1) | cifar10 |
| `RART_GRAD_CLIP` | Gradient-norm clipping threshold, 0 = off (step 1) | 0 |
| `RART_WORK_DIR` | Output and data folder | see above |

Measured on an RTX 5060 Laptop GPU: about 1.1 min/epoch (step 0) and 1.4–1.6 min/epoch
(step 1). Model checkpoints (`*.pth`) are not committed.

## Step 0 — anchor collapse across designs

All variants share one backbone (6 × [7×7 local window attention → anchor pathway → FFN],
384-d, CIFAR-100 at 224², 16 anchors, 10 epochs). Only the anchor pathway differs:

| Variant | How anchors pool patches |
|---|---|
| `perceiver` | RART-16/32 routing: learned anchor queries, softmax over patches |
| `competitive` | Same, but softmax over anchors (slot-attention competition) |
| `anchorformer` | AnchorFormer-style: static anchors, softmax over anchors, Markov round trip |
| `paca` | PaCa-ViT-style: MLP cluster assignment, patches attend to clusters |
| `partition` | Fixed exclusive 4×4 regions (Partition RART) |

The AnchorFormer- and PaCa-style variants are simplified reimplementations, not the
published models.

| Model | Test acc, seed 42 / 43 | Map similarity* | Anchor spread** |
|---|---:|---:|---:|
| `perceiver` | 41.25% / 40.66% | 0.998 / 0.999 | ≈ 0.00 |
| `competitive` | 42.71% / 41.41% | 0.815 / 0.818 | 0.21 |
| `anchorformer` | 42.34% / 43.28% | 0.990 / 0.990 | 0.05 |
| `paca` | 42.08% / 41.94% | 0.606 / 0.592 | 0.17 |
| `partition` | **45.68% / 46.48%** | **0.000 / 0.000** | **0.59** |

\* Mean pairwise cosine similarity of the anchors' pooling maps (1 = every anchor pools the same patches).
\*\* Share of the pooled content that differs between anchors (0 = identical anchors); seed 42.

**Finding.** Anchors that are the same for every image (`perceiver`, `anchorformer`)
collapse completely. Anchors derived from the image (`competitive`, `paca`) are partly
diverse but grow more similar with depth. Fixed exclusive regions do not collapse and gave
the best accuracy in both seeds.

## Step 1–2 — moving anchors on Cluttered CIFAR-10

A 64×64 CIFAR-10 object is placed at a random position on a 256×256 canvas over 8
distractor fragments; it covers 6.25% of the image. Positional models use an exclusive
Voronoi partition: every patch belongs to its nearest anchor.

| Model | How anchors are placed |
|---|---|
| `free` | No positions; RART-style learned queries over all patches |
| `fixed` | Fixed grid (Partition RART) |
| `moving` | Each block, one importance-weighted Lloyd (k-means) step inside the anchor's own region |
| `focus` | A quarter of the anchors on a coarse fixed grid for coverage; the rest placed each block at the peaks of a learned importance map, anywhere in the image, with each chosen peak suppressed so the next anchor lands elsewhere |

Results (seed 42, 10 epochs):

| Model | Test acc, 4 anchors | Test acc, 16 anchors | Map similarity* | Anchors touching object, last block (4 / 16) |
|---|---:|---:|---:|---:|
| `free` | 47.00% | 47.82% | 0.998 / 0.997 | — |
| `fixed` | 47.08% | 48.10% | 0.000 | 0.39 / 0.19 |
| `moving` | 47.61% | 48.64% | 0.000 | 0.33 / 0.20 |
| **`focus`** | **48.79%** | **49.63%** | **0.000** | **0.57** / 0.24 |

\* Mean pairwise cosine similarity of the anchors' pooling maps, 4 / 16 anchors.

**Findings.**

* `focus` is the best model at both anchor counts (+1.7 and +1.5 points over `fixed`),
  and its anchors do not collapse. With 4 anchors, the share of anchors whose region
  contains the object rises through the network from 0.44 to 0.57; `fixed` stays at 0.39
  and `moving` drifts down to 0.33. This is a single seed, and differences of about 1 point
  are within seed-to-seed variation seen in step 0, so more seeds are needed before the
  gap can be claimed.
* The collapsed `free` anchors all collapse onto the object, which does not hurt on a
  single-object task; `free` stays roughly level with `fixed`.
* `moving` behaves as a coverage rule. [`diag_moving.py`](step1_moving_anchors/diag_moving.py)
  shows its importance map does find the object (4–9× more importance per patch than
  background), but its anchors move only about 1 patch, because each Lloyd step keeps an
  anchor inside its own region.
* In the anchor plots (`results/*_anchors.png`), `focus` anchors gather on all image
  content — the object and the distractor fragments — and leave the empty background to a
  few large coverage regions. The importance map learned "where there is content" rather
  than "where the object is", which is expected when the distractors are real image
  fragments.

**Bug fixed during this experiment.** The first `focus` runs failed (the 4-anchor run
diverged at epoch 8; the 16-anchor run never learned). Focus anchors can land on top of each
other and leave another anchor's region empty. The pooling step divided an empty region's
sum by a clamp of `1e-9`, giving a 10⁹ straight-through gradient that overflowed in fp16, so
the mixed-precision scaler skipped those steps. The normaliser is now clamped at 1, which
leaves every non-empty region unchanged (its top patch always contributes exactly 1). The
fix does not affect `free`, `fixed` or `moving`, which never produce empty regions, so their
results were kept. The `focus` results above are from the fixed code.

Logs: `results/step1_run1_free-fixed-moving_A4.log` (ends with an interrupted `free` 16-anchor
run), `results/step1_run2_A16_plus_buggy_focus.log` (the 16-anchor `free`/`fixed`/`moving`
results plus the two failed `focus` runs), and `results/step1_run3_focus_A4-A16.log` (the
reported `focus` runs, with per-epoch gradient-norm statistics).
