# Follow-up experiments

These experiments test whether anchor collapse is a general property of anchor bottlenecks,
and whether anchors can move to important regions without collapsing. Each folder holds a
self-contained notebook and its results.

| Folder | Question | Status |
|---|---|---|
| [`step0_anchor_collapse/`](step0_anchor_collapse/) | Do other anchor designs collapse like RART-32? | Done, 2 seeds |
| [`step1_moving_anchors/`](step1_moving_anchors/) | Can anchors move to a small object without collapsing? | 4 anchors done; 16 anchors and `focus` in progress |

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
| `focus` | A quarter of the anchors on a coarse fixed grid; the rest at importance peaks anywhere in the image, with peak suppression |

Results so far (seed 42, 4 anchors):

| Model | Test acc | Map similarity | Anchor spread |
|---|---:|---:|---:|
| `free` | 47.00% | 0.998 | 0.004 |
| `fixed` | 47.08% | 0.000 | 0.530 |
| `moving` | 47.61% | 0.000 | 0.549 |

The three are within seed noise (about 1 point). The collapsed `free` anchors all collapse
onto the object, which does not hurt when there is a single object.
[`diag_moving.py`](step1_moving_anchors/diag_moving.py) shows that the `moving` model's
importance map does find the object (4–9× more importance per patch than background), but
its anchors move only about 1 patch: a Lloyd step keeps each anchor inside its own region,
so it preserves coverage but cannot gather anchors on a small object. The `focus` variant
separates coverage from focus to address this.
