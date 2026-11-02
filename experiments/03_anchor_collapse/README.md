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
| `anchorformer` | AnchorFormer (Shan et al., 2025): per-head learned anchors, softmax over anchors, mass-normalised round trip H = A Δ⁻¹ Aᵀ V |
| `rpattention` | Representative Attention (Li et al., 2026), implemented from the paper: slot-mass-normalised gather, latent self-attention, query distribution, 3×3 depth-wise conv bypass |
| `paca` | PaCa-ViT-style: MLP cluster assignment, patches attend to clusters |
| `partition` | Fixed exclusive 4×4 regions (Partition RART) |

Neither AnchorFormer nor Representative Attention has released code (the Representative
Attention repository holds only a README and figures), so both are implemented from the papers.
`anchorformer` follows the paper's equations 11–13 exactly, including its only stabiliser, the
mass normalisation Δ⁻¹. `rpattention` follows equations 2–10 and appendix equation 14, including
the slot-mass normalisation (ε = 1e-6) and the 3×3 depth-wise convolution bypass. In both cases
the module is placed as the global pathway of our shared block, next to the local windows, like
every other variant; the published models use it as the only attention. Widths and head counts
match the other pathways. The PaCa-style variant is simplified (assignment MLP of width 384
instead of 4×).

| Model | Test acc, mean ± std | Seeds 42 / 43 / 44 | Map similarity* | Anchor spread** |
|---|---:|---:|---:|---:|
| `perceiver` | 40.92 ± 0.30 | 41.25 / 40.66 / 40.85 | 0.998 | ≈ 0.00 |
| `competitive` | 42.14 ± 0.67 | 42.71 / 41.41 / 42.31 | 0.792 | 0.22 |
| `anchorformer` | 42.52 ± 0.68 | 42.34 / 43.28 / 41.95 | 0.990 | 0.06 |
| `paca` | 42.02 ± 0.07 | 42.08 / 41.94 / 42.04 | 0.592 | 0.18 |
| `dense` (no anchors) | 40.93 ± 0.95 | 42.02 / 40.33 / 40.43 | — | — |
| `rpattention` | 47.54 ± 0.73 | 47.09 / 47.16 / 48.38 | 0.683 | 0.27 |
| `partition` | 46.31 ± 0.56 | 45.68 / 46.48 / 46.77 | **0.000** | **0.60** |

**Local-convolution ablation.** Representative Attention includes a 3×3 depth-wise convolution
on the values, which the other pathways lack. `rpattention_nolocal` removes it;
`partition_local` adds the same convolution to the partition (on the patch values, before the
output projection):

| Model | Test acc, mean ± std | Seeds 42 / 43 / 44 | Map similarity | Anchor spread |
|---|---:|---:|---:|---:|
| `partition_local` | **49.93 ± 0.71** | **50.18 / 49.13 / 50.48** | 0.000 | 0.59 |
| `rpattention` | 47.54 ± 0.73 | 47.09 / 47.16 / 48.38 | 0.683 | 0.27 |
| `partition` | 46.31 ± 0.56 | 45.68 / 46.48 / 46.77 | 0.000 | 0.60 |
| `rpattention_nolocal` | 42.68 ± 0.86 | 43.50 / 42.76 / 41.79 | 0.687 | 0.28 |

Paired gaps (95% CI): partition + conv − Representative Attention +2.39 [+0.86, +3.91];
partition − Representative Attention without conv +3.63 [+0.14, +7.11]; the convolution adds
+3.62 [+1.31, +5.93] to the partition and +4.86 [+1.00, +8.72] to Representative Attention.

\* Mean pairwise cosine similarity of the anchors' pooling maps (1 = every anchor pools the same patches), averaged over blocks and seeds.
\*\* Share of the pooled content that differs between anchors (0 = identical anchors), averaged over blocks and seeds.

`dense` replaces the anchor pathway with global attention among all patches on the same
backbone (local windows kept), as a no-anchor reference.

## Finding

Anchors that are the same for every image (`perceiver`, `anchorformer`) collapse completely,
AnchorFormer despite its mass normalisation. Anchors derived from the image (`competitive`,
`paca`, `rpattention`) are partly diverse but grow more similar with depth: Representative
Attention's stabilisers prevent full collapse (map similarity 0.68), but its summaries are far
less distinct than the partition's (spread 0.27 against 0.60). Fixed exclusive regions cannot
collapse (zero overlap holds by construction). Without a convolution bypass the partition beats
every other anchor pathway in every seed, by 3.0–5.9 points, and the dense pathway by 5.4.
Representative Attention's lead over the plain partition (+1.23) comes from its convolution:
with the same convolution on both, the partition is 2.4 points ahead.

## Running

[`anchor_collapse.py`](anchor_collapse.py) is the notebook as a script; seed 44 and the dense
reference were run with it on a vast.ai A100 (2026-10-30/31):

```bash
RART_SEED=44 python anchor_collapse.py                        # all five anchor pathways
RART_SEED=42 RART_MODELS=dense RART_WORK_DIR=dense python anchor_collapse.py
```

Representative Attention and the convolution ablation (2× A100, 2026-11-02):

```bash
RART_SEED=42 RART_MODELS=rpattention RART_WORK_DIR=rpattention python anchor_collapse.py
RART_SEED=42 RART_MODELS=rpattention_nolocal,partition_local RART_WORK_DIR=ablation_local python anchor_collapse.py
```

[`start_dense.sh`](start_dense.sh) is the exact launch used for the dense runs;
`results/start_rp.sh` and `results/start_ablation.sh` for the later ones. The box had no
matplotlib, so the run ends with an import error after the results are saved; the seed-44
figure was plotted locally.

## Files

| Folder | Contents |
|---|---|
| `results/tables/` | Per-seed results (`.json`: accuracy history and per-block collapse metrics; `.md`: summary table) |
| `results/figures/` | Per-block collapse metrics, trained (solid) and at initialisation (dashed) |
| `results/logs/` | Training logs per seed |
| `results/dense/` | The dense reference: per-seed tables and `logs/` |
| `results/rpattention/` | Representative Attention, 3 seeds (2× A100, 2026-11-02), with `logs/` |
| `results/ablation_local/` | `rpattention_nolocal` and `partition_local`, 3 seeds, with `logs/` |

Result files keep the notebook's internal run name (`step0_…`).
