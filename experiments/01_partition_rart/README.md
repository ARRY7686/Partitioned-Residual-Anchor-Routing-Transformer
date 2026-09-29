# 01 — Building Partition RART

**Question.** Can global communication in a ViT go through a small set of anchor tokens instead
of dense patch-to-patch attention, and what goes wrong when it does?

**Notebook:** [`partition-rart-building.ipynb`](partition-rart-building.ipynb)
**Checkpoint:** `partition_rart_epoch10_checkpoint.pth` — Partition RART after 10 epochs
(11,272,176 parameters, 44.62% test accuracy)

## What the notebook builds

In order: a ViT baseline, compressed anchor RART with 16 and 32 anchors, a mean-pooled ViT,
diagnostics of anchor collapse, the Local → Global → Local model with 5×5 local routing, and
Partition RART, where every patch belongs to exactly one of 16 fixed spatial regions. All are
trained on CIFAR-100 at 224×224 (patch 16, 196 patches) for 10 epochs.

## Results

| Model | Parameters | Test accuracy |
|---|---:|---:|
| ViT (CLS token) | 11.06M | 45.44% |
| MeanPool ViT | 11.06M | 44.59% |
| RART-16 (compressed anchors) | 11.27M | 42.21% |
| RART-32 (compressed anchors) | 11.27M | 39.73% |
| Local → Global → Local (5×5 routing) | 11.27M | 45.38% |
| Partition RART | 11.27M | 44.62% |

**Anchor collapse in RART-32.** Mean pairwise cosine similarity between anchors was 0.989–0.995
across blocks, patch → anchor attention was almost uniform (entropy 5.272 against a maximum of
5.278; 194.8 of 196 patches effectively attended), and anchor attention maps had a similarity
of 0.9995. Restricting each anchor to a local region brought anchor similarity down to
0.26–0.42 in the Local → Global → Local model. Experiment [03](../03_anchor_collapse/) tests
whether this collapse generalises to other anchor designs.

## Notes

* The notebook's cells have no saved outputs; the numbers above come from the interactive run
  recorded in [`docs/research_summary.md`](../../docs/research_summary.md). All results are a
  single seed.
* The `partition_rart_epoch10_checkpoint.pth` weights match this notebook's
  `PartitionLocalGlobalLocalRART`. The resolution-scaling notebook (02) concatenates attention
  heads in a different order in `_patch_to_anchor`, so this checkpoint should not be loaded
  into that notebook's model without fixing the head order.
