# Step 0 results (16 anchors, seed 43, 10 epochs)

| Model | Params | Final test acc | When | Map sim (mean / last block) | Effective patches (of 196) | Summary sim | Anchor spread |
|---|---:|---:|---|---:|---:|---:|---:|
| perceiver | 11.27M |  | init | 1.000 / 1.000 | 195.9 | 1.000 | 0.000 |
|  |  | 40.66% | trained | 0.999 / 1.000 | 93.3 | 0.999 | 0.002 |
| competitive | 11.27M |  | init | 1.000 / 1.000 | 196.0 | 1.000 | 0.000 |
|  |  | 41.41% | trained | 0.818 / 0.809 | 151.2 | 0.820 | 0.187 |
| anchorformer | 10.09M |  | init | 1.000 / 1.000 | 196.0 | 1.000 | 0.000 |
|  |  | 43.28% | trained | 0.990 / 1.000 | 194.8 | 0.922 | 0.074 |
| paca | 11.40M |  | init | 0.987 / 0.991 | 194.7 | 0.990 | 0.010 |
|  |  | 41.94% | trained | 0.592 / 0.676 | 72.6 | 0.807 | 0.181 |
| partition | 11.27M |  | init | 0.000 / 0.000 | 12.2 | 0.620 | 0.361 |
|  |  | 46.48% | trained | 0.000 / 0.000 | 9.8 | 0.379 | 0.589 |
