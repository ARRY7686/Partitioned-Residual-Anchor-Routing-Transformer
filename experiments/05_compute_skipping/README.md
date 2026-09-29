# 05 — Compute skipping with learned importance

**Question.** Can the importance map learned by `focus` anchors (04) cut compute by letting unimportant patches skip work, while those patches still receive global context from their anchor?

**Notebook:** [`../04_moving_anchors/moving-anchors.ipynb`](../04_moving_anchors/moving-anchors.ipynb) (the same notebook as 04, run with `RART_KEEP_RATIO`, `RART_SKIP_FROM` and `RART_SKIP_SCORE`)

## Method

Anchors are well under 1% of the computation — `free`, `fixed` and `focus` with 16 anchors all
cost about 5.31 GFLOPs per image — so savings have to come from per-patch work. From block 3
on, only the top share of patches (by the chosen score) go through the FFN, the most
expensive part of each block; the other patches skip it but have already received their
anchor's global update. GFLOPs are counted from the Linear and Conv2d layers that actually
run (attention matmuls and the Voronoi assignment, a few percent of the total, are not
counted). With `importance` scoring, kept patches' FFN output is multiplied by a
straight-through gate (value 1) so the importance map also learns from the skip decision.

## Results so far

`focus`, 16 anchors, seed 42:

| Patches kept in the FFN (blocks 3–6) | Chosen by | Test acc | GFLOPs / image |
|---:|---|---:|---:|
| 100% | — | 49.63% | 5.31 |
| 25% | learned importance | 49.25% | 3.96 (−25%) |
| 50% | learned importance | 38.25%† | 4.41 (−17%) |

† Training was disrupted: gradient-norm spikes to about 17 at epochs 3–4 (normally 2–4), after
which the run never caught up. The suspected cause is the importance gate, whose gradient is
proportional to 1/importance and so becomes very large for kept patches with low importance;
keeping 50% selects many more of those than keeping 25%. This is not yet confirmed and the
run will be repeated with a bounded gate.

## Still running

 25% kept with `random` and `norm` (token L2 norm, as in SparseViT) selection
for `focus`, and 25% kept with `norm` selection for `fixed`. These baselines show whether the
learned importance choice matters. All compute-skipping results are single-seed. Results are in `results/tables/`, logs in `results/logs/`.
