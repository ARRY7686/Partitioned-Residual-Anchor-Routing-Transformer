# 02 — Resolution scaling and efficient-transformer baselines

**Question.** Does Partition RART scale better than a dense ViT as image resolution grows, and
how does it compare with established efficient transformers?

**Notebook:** [`resolution-scaling.ipynb`](resolution-scaling.ipynb) (self-contained; untrained
models, inference only)

## Setup

Resolution-agnostic Partition RART and ViT (384-d, depth 6, patch 16) are timed from 224² to
2048². The number of anchors grows with resolution (one anchor per 4×4 block of patches: 16
anchors at 224², 1,024 at 2048²). Batch size 2, or 1 at 2048². PVTv2-B1 and EfficientViT-B2
from `timm` are added at 2048². All measurements are on a Kaggle Tesla T4.

## Results (from the notebook's saved outputs)

| Resolution | RART latency | ViT latency | RART peak memory | ViT peak memory |
|---:|---:|---:|---:|---:|
| 224² | 35.4 ms | 5.5 ms | 62 MB | 61 MB |
| 512² | 119.4 ms | 28.5 ms | 103 MB | 180 MB |
| 1024² | 454.9 ms | 261.2 ms | 252 MB | 1,704 MB |
| 1280² | 717.6 ms | 632.8 ms | 365 MB | 3,984 MB |
| 1536² | 1,022.9 ms | 1,251.9 ms | 506 MB | 8,091 MB |
| 2048² (batch 1) | 1,641.8 ms | 2,125.2 ms | 463 MB | 12,582 MB |

At 2048², batch 1:

| Model | Parameters | Latency | Peak memory |
|---|---:|---:|---:|
| Partition RART | 17.62M | 1,641.8 ms | 463 MB |
| ViT | 17.27M | 2,125.2 ms | 12,582 MB |
| PVTv2-B1 | 13.55M | 804.8 ms | 1,901 MB |
| EfficientViT-B2 | 22.03M | 207.2 ms | 1,007 MB |

**Findings.** RART becomes faster than the dense ViT between 1280² and 1536², and uses far less
memory at high resolution (27× less at 2048²). PVTv2-B1 and EfficientViT-B2 are faster than
this RART implementation at 2048², so the result is a memory advantage over dense attention,
not a speed advantage over efficient transformers.

## Notes

* The main README's scaling tables were recorded from an earlier run of this notebook and
  differ from the saved outputs above (for example 1,755 ms / 2,429 MB for RART at 2048²).
  The conclusions are the same; the table above is what this notebook currently produces.
* The interaction-count cell uses `(resolution // 64)²` anchors, which gives 9 anchors at
  224² instead of the 16 the model uses, so its 224² row does not match the model.
