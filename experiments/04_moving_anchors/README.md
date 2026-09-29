# 04 — Moving anchors on Cluttered CIFAR-10

**Question.** Can anchors move to the important parts of an image while keeping exclusive regions, and does that beat a fixed partition?

**Notebook:** [`moving-anchors.ipynb`](moving-anchors.ipynb)  
**Diagnostic:** [`diag_moving.py`](diag_moving.py) — importance-map quality and anchor displacement for a trained checkpoint

## Setup

A 64×64 CIFAR-10 object is placed at a random position on a 256×256 canvas over 8
distractor fragments; it covers 6.25% of the image. Positional models use an exclusive
Voronoi partition: every patch belongs to its nearest anchor.

| Model | How anchors are placed |
|---|---|
| `free` | No positions; RART-style learned queries over all patches |
| `fixed` | Fixed grid (Partition RART) |
| `moving` | Each block, one importance-weighted Lloyd (k-means) step inside the anchor's own region |
| `focus` | A quarter of the anchors on a coarse fixed grid for coverage; the rest placed each block at the peaks of a learned importance map, anywhere in the image, with each chosen peak suppressed so the next anchor lands elsewhere |

Results (10 epochs). `fixed` and `focus` were run with 3 seeds; `free` and `moving` with seed 42 only.

| Model | Test acc, 4 anchors | Test acc, 16 anchors | Map similarity* | Anchors touching object, last block (4 / 16)** |
|---|---:|---:|---:|---:|
| `free` | 47.00% | 47.82% | 0.998 / 0.997 | — |
| `moving` | 47.61% | 48.64% | 0.000 | 0.33 / 0.20 |
| `fixed` | 46.04 ± 0.94% (47.08 / 45.77 / 45.26) | 46.71 ± 1.72% (48.10 / 47.23 / 44.79) | 0.000 | 0.39 / 0.19 |
| **`focus`** | **47.52 ± 1.11%** (48.79 / 47.00 / 46.77) | **49.84 ± 0.26%** (49.63 / 50.13 / 49.75) | **0.000** | **0.57** / 0.24 |

\* Mean pairwise cosine similarity of the anchors' pooling maps, 4 / 16 anchors.
\*\* Seed 42.

Accuracies for `fixed` and `focus` are mean ± standard deviation over seeds 42 / 43 / 44, with the individual seeds in brackets.

Paired gap, `focus` − `fixed` (same seed):

| Anchors | Seed 42 | Seed 43 | Seed 44 | Mean |
|---:|---:|---:|---:|---:|
| 4 | +1.71 | +1.23 | +1.51 | +1.48 |
| 16 | +1.53 | +2.90 | +4.96 | +3.13 |

## Findings



* `focus` beats `fixed` in all 6 paired runs, by 1.2–1.7 points with 4 anchors and 1.5–5.0
  points with 16 anchors, and its anchors do not collapse. With 16 anchors it is also much
  more stable across seeds (49.6–50.1%) than `fixed` (44.8–48.1%). With 4 anchors, the share
  of anchors whose region contains the object rises through the network from 0.44 to 0.57;
  `fixed` stays at 0.39 and `moving` drifts down to 0.33.
* The collapsed `free` anchors all collapse onto the object, which does not hurt on a
  single-object task; `free` stays roughly level with `fixed`.
* `moving` behaves as a coverage rule. [`diag_moving.py`](diag_moving.py)
  shows its importance map does find the object (4–9× more importance per patch than
  background), but its anchors move only about 1 patch, because each Lloyd step keeps an
  anchor inside its own region.
* In the anchor plots (`results/figures/*_anchors.png`), `focus` anchors gather on all image
  content — the object and the distractor fragments — and leave the empty background to a
  few large coverage regions. The importance map learned "where there is content" rather
  than "where the object is", which is expected when the distractors are real image
  fragments.

## Bug fixed during this experiment

 The first `focus` runs failed (the 4-anchor run
diverged at epoch 8; the 16-anchor run never learned). Focus anchors can land on top of each
other and leave another anchor's region empty. The pooling step divided an empty region's
sum by a clamp of `1e-9`, giving a 10⁹ straight-through gradient that overflowed in fp16, so
the mixed-precision scaler skipped those steps. The normaliser is now clamped at 1, which
leaves every non-empty region unchanged (its top patch always contributes exactly 1). The
fix does not affect `free`, `fixed` or `moving`, which never produce empty regions, so their
results were kept. The `focus` results above are from the fixed code.

## Files

Logs: `results/logs/step1_run1_free-fixed-moving_A4.log` (ends with an interrupted `free` 16-anchor
run), `results/logs/step1_run2_A16_plus_buggy_focus.log` (the 16-anchor `free`/`fixed`/`moving`
results plus the two failed `focus` runs), `results/logs/step1_run3_focus_A4-A16.log` (the
reported seed-42 `focus` runs, with per-epoch gradient-norm statistics), and
`results/logs/step1_run4_seed{43,44}_fixed-focus.log` (the extra seeds).

| Folder | Contents |
|---|---|
| `results/tables/` | Per-seed results (`.json` with accuracy history, per-block anchor metrics and, for later runs, gradient norms; `.md` summary) |
| `results/figures/` | Test canvases with each model's final-block regions and anchor positions per block |
| `results/logs/` | Training logs, described above |
| `results/checkpoints/` | Model weights (not committed; regenerate with the notebook) |

Result files keep the notebook's internal run name (`step1_…`).
