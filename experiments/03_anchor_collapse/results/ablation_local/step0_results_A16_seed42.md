# Experiment 03 results (16 anchors, seed 42, 10 epochs)

| Model | Params | Final test acc | When | Map sim (mean / last block) | Effective patches (of 196) | Summary sim | Anchor spread |
|---|---:|---:|---|---:|---:|---:|---:|
| rpattention_nolocal | 10.51M |  | init | 0.982 / 0.992 | 194.3 | 0.959 | 0.039 |
|  |  | 43.50% | trained | 0.675 / 0.768 | 130.3 | 0.697 | 0.284 |
| partition_local | 11.28M |  | init | 0.000 / 0.000 | 12.2 | 0.641 | 0.340 |
|  |  | 50.18% | trained | 0.000 / 0.000 | 9.9 | 0.373 | 0.593 |
