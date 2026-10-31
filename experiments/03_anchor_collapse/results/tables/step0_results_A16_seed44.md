# Experiment 03 results (16 anchors, seed 44, 10 epochs)

| Model | Params | Final test acc | When | Map sim (mean / last block) | Effective patches (of 196) | Summary sim | Anchor spread |
|---|---:|---:|---|---:|---:|---:|---:|
| perceiver | 11.27M |  | init | 1.000 / 1.000 | 195.9 | 1.000 | 0.000 |
|  |  | 40.85% | trained | 0.998 / 0.999 | 87.3 | 0.997 | 0.004 |
| competitive | 11.27M |  | init | 1.000 / 1.000 | 196.0 | 1.000 | 0.000 |
|  |  | 42.31% | trained | 0.743 / 0.703 | 146.7 | 0.739 | 0.261 |
| anchorformer | 10.09M |  | init | 1.000 / 1.000 | 196.0 | 1.000 | 0.000 |
|  |  | 41.95% | trained | 0.990 / 1.000 | 194.8 | 0.944 | 0.053 |
| paca | 11.40M |  | init | 0.989 / 0.992 | 194.9 | 0.991 | 0.008 |
|  |  | 42.04% | trained | 0.577 / 0.718 | 62.5 | 0.797 | 0.191 |
| partition | 11.27M |  | init | 0.000 / 0.000 | 12.2 | 0.629 | 0.352 |
|  |  | 46.77% | trained | 0.000 / 0.000 | 9.6 | 0.347 | 0.621 |
