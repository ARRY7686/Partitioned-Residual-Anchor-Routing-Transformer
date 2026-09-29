# Step 0 results (16 anchors, seed 42, 10 epochs)

| Model | Params | Final test acc | When | Map sim (mean / last block) | Effective patches (of 196) | Summary sim | Anchor spread |
|---|---:|---:|---|---:|---:|---:|---:|
| perceiver | 11.27M |  | init | 1.000 / 1.000 | 195.9 | 1.000 | 0.000 |
|  |  | 41.25% | trained | 0.998 / 0.999 | 86.3 | 0.996 | 0.005 |
| competitive | 11.27M |  | init | 1.000 / 1.000 | 196.0 | 1.000 | 0.000 |
|  |  | 42.71% | trained | 0.815 / 0.778 | 142.6 | 0.801 | 0.211 |
| anchorformer | 10.09M |  | init | 1.000 / 1.000 | 196.0 | 1.000 | 0.000 |
|  |  | 42.34% | trained | 0.990 / 1.000 | 194.8 | 0.944 | 0.053 |
| paca | 11.40M |  | init | 0.988 / 0.993 | 194.9 | 0.993 | 0.006 |
|  |  | 42.08% | trained | 0.606 / 0.721 | 61.8 | 0.818 | 0.171 |
| partition | 11.27M |  | init | 0.000 / 0.000 | 12.2 | 0.644 | 0.338 |
|  |  | 45.68% | trained | 0.000 / 0.000 | 9.9 | 0.380 | 0.590 |
