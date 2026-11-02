# Experiment 03 results (16 anchors, seed 43, 10 epochs)

| Model | Params | Final test acc | When | Map sim (mean / last block) | Effective patches (of 196) | Summary sim | Anchor spread |
|---|---:|---:|---|---:|---:|---:|---:|
| rpattention_nolocal | 10.51M |  | init | 0.981 / 0.991 | 194.1 | 0.954 | 0.043 |
|  |  | 42.76% | trained | 0.687 / 0.774 | 137.2 | 0.699 | 0.282 |
| partition_local | 11.28M |  | init | 0.000 / 0.000 | 12.2 | 0.641 | 0.340 |
|  |  | 49.13% | trained | 0.000 / 0.000 | 10.0 | 0.376 | 0.591 |
