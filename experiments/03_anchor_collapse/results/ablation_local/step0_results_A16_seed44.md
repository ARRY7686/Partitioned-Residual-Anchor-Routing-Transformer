# Experiment 03 results (16 anchors, seed 44, 10 epochs)

| Model | Params | Final test acc | When | Map sim (mean / last block) | Effective patches (of 196) | Summary sim | Anchor spread |
|---|---:|---:|---|---:|---:|---:|---:|
| rpattention_nolocal | 10.51M |  | init | 0.984 / 0.993 | 194.5 | 0.964 | 0.033 |
|  |  | 41.79% | trained | 0.697 / 0.749 | 140.6 | 0.708 | 0.273 |
| partition_local | 11.28M |  | init | 0.000 / 0.000 | 12.2 | 0.666 | 0.316 |
|  |  | 50.48% | trained | 0.000 / 0.000 | 9.6 | 0.374 | 0.591 |
