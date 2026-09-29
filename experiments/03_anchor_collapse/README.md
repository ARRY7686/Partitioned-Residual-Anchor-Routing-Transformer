# 03 — Anchor collapse across anchor designs

**Question.** Is the anchor collapse seen in RART-32 (01) a general property of anchor bottlenecks, or specific to RART's routing?

**Notebook:** [`anchor-collapse.ipynb`](anchor-collapse.ipynb)

## Setup

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

## Finding

 Anchors that are the same for every image (`perceiver`, `anchorformer`)
collapse completely. Anchors derived from the image (`competitive`, `paca`) are partly
diverse but grow more similar with depth. Fixed exclusive regions do not collapse and gave
the best accuracy in both seeds.

## Files

| Folder | Contents |
|---|---|
| `results/tables/` | Per-seed results (`.json`: accuracy history and per-block collapse metrics; `.md`: summary table) |
| `results/figures/` | Per-block collapse metrics, trained (solid) and at initialisation (dashed) |
| `results/logs/` | Training logs per seed |

Result files keep the notebook's internal run name (`step0_…`).
