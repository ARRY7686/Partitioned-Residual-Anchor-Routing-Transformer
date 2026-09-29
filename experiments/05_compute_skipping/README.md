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

## Results

All runs: 16 anchors, seed 42, 10 epochs. Patches not kept skip the FFN in blocks 3–6 but still
receive their anchor's global update.

| Model | Patches kept in the FFN (blocks 3–6) | Chosen by | Test acc | GFLOPs / image |
|---|---:|---|---:|---:|
| `focus` | 100% | — | 49.63% | 5.31 |
| `focus` | 25% | learned importance | 49.25% | 3.96 (−25%) |
| `focus` | 25% | random | 50.49% | 3.96 (−25%) |
| `focus` | 25% | token L2 norm (as in SparseViT) | **50.87%** | 3.96 (−25%) |
| `focus` | 50% | learned importance | 38.25%† | 4.41 (−17%) |
| `fixed` | 100% | — | 48.10% | 5.31 |
| `fixed` | 25% | token L2 norm | 48.44% | 3.95 (−25%) |

† Training was disrupted: gradient-norm spikes to about 17 at epochs 3–4 (normally 2–4), after
which the run never caught up. The suspected cause is the importance gate, whose gradient is
proportional to 1/importance and so becomes very large for kept patches with low importance;
keeping 50% selects many more of those than keeping 25%. This is not yet confirmed; a re-run
with a bounded gate is planned.

## Findings

* **Skipping is almost free on this task.** Letting 75% of patches skip the FFN in blocks 3–6
  cuts compute by 25% with no accuracy loss, for both `focus` (49.3–50.9% against 49.63%) and
  `fixed` (48.44% against 48.10%). All anchors stay diverse (map similarity 0.000). A plausible
  reason is that skipped patches still receive their anchor's global update, but this has not
  been isolated.
* **The learned importance map does not choose better than simple rules.** Random and
  token-norm selection matched or beat learned importance (50.49% and 50.87% against 49.25%);
  with a single seed, differences of about 1 point are within noise. So this experiment does
  not support using the importance map to decide which patches to compute.
* A possible reason is that 25% of patches roughly covers all image content on this canvas, so
  every rule ends up keeping the object. A smaller budget (for example 10%) or a
  high-resolution task with a smaller object may separate the rules.

All compute-skipping results are single-seed. Tables are in `results/tables/`, anchor plots in
`results/figures/`, logs in `results/logs/`.
