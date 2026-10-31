# Partition RART: Anchors That Move Without Collapsing

Research code for a Vision Transformer whose global communication runs through a small set of
**anchor tokens with exclusive regions**. Every patch belongs to exactly one anchor, so two
anchors can never summarise the same patches. On top of that partition, **focus anchors** move
to the parts of the image a learned importance map marks as relevant, while the partition stays
exclusive.

The paper draft is in [`paper/`](paper/) ("Anchors That Move Without Collapsing: Exclusive-Region
Anchor Routing for Vision Transformers"). All experiments, logs and result files are in
[`experiments/`](experiments/).

**Status:** research in progress. The results below are from small, from-scratch training runs
on synthetic canvases. They study a mechanism, not state-of-the-art accuracy.

---

## Summary of findings

1. **Anchors with image-independent queries collapse.** On one shared backbone, RART- and
   AnchorFormer-style anchors learn nearly identical pooling maps (map similarity 0.998 and
   0.990), and image-derived anchors are only partly diverse. The exclusive partition cannot collapse by construction, and it was the most
   accurate anchor pathway in all three seeds on CIFAR-100 (46.31 ± 0.56%, against 40.93 ± 0.95%
   for a dense pathway on the same backbone).
2. **Focus anchors beat the fixed partition.** On a cluttered 256² CIFAR-10 canvas trained for
   30 epochs, focus-16 reaches 66.82 ± 0.66% against 65.29 ± 0.94% for fixed-16. The paired gap is
   +1.54 points (95% CI [0.53, 2.55]), and focus wins in 3 of 3 seeds.
3. **Both beat downsampling.** Focus anchors beat ViTs on downsampled inputs by 4.7–11.8 points,
   on Cluttered CIFAR-10 and on a Tiny ImageNet canvas.
4. **Hard exclusive assignment is what makes moving anchors trainable.** Its softmax relaxation
   diverges in every seed, with or without gradient clipping, and also with Sinkhorn balancing.
5. **The limits are real.** PVTv2-B0 (75.1%) and ResNet-18 (86.1%) are far more accurate in this
   from-scratch regime. At 256² the anchor models are slower per training step than a dense ViT
   and use more memory. The memory advantage only appears at much higher resolution.

---

## Method

### One block: local, global through anchors, local

```text
patches -> local window attention
        -> gather: each anchor pools the patches of its own region      (N -> A)
        -> anchor self-attention                                        (A x A)
        -> scatter: each patch reads back from its own anchor           (A -> N)
        -> FFN
```

The dense N × N attention of a ViT is replaced by A × A attention among anchors. At 224² with
16 anchors, that is 196 + 16² = 452 global interactions instead of 196² = 38,416. This is an
interaction count, not a wall-clock speedup.

### Exclusive regions

Each patch is assigned to its nearest anchor, which splits the patch grid into Voronoi regions.
The forward pass uses this hard assignment. Gradients reach the anchor positions through a
straight-through softmax over distances. Anchors pool only inside their region. The pooling
denominator is clamped at 1, which only affects empty regions. An epsilon clamp produced
gradients of order 10⁹ when two moving anchors met.

With anchors fixed on a grid, the block is **Partition RART** (`fixed`).

### Focus anchors

A linear head scores every patch. In each block:

* **Coverage anchors.** A quarter of the anchors stay on a coarse fixed grid (4 of 16), so every
  part of the image keeps a summary.
* **Focus anchors.** The rest are placed one at a time at the soft-argmax of the scores. After
  each placement, the scores around it are suppressed (strength β = 10, width σ_s = 1.5 patches),
  so that the next anchor goes elsewhere. A weak prior towards spread-out home positions
  (σ_h = 0.35) breaks ties early in training.
* **Importance in pooling.** Gather logits are offset by the log-importance. This trains the
  importance head from the classification loss alone.

All anchors share one Voronoi partition. Focus anchors can crowd onto the object, which makes
their regions small, but they never share a patch.

---

## Results

### Main comparison (experiment 06)

Mean ± std over seeds 42/43/44. In **Cluttered CIFAR-10**, a CIFAR-10 image (upscaled to 64 px)
sits at a random position on a 256² canvas among distractor crops. In the **Tiny ImageNet
canvas**, a 64 px Tiny ImageNet image is pasted at native resolution among 32 px crops
(200 classes).

| Model | Params | GFLOPs | Cluttered, 30 ep | Tiny ImageNet, 15 ep |
|---|---:|---:|---:|---:|
| focus-16 | 11.4M | 5.49 | **66.82 ± 0.66** | 17.68 ± 0.33 |
| focus-4 | 11.4M | 5.45 | 64.67 ± 0.26 | – |
| fixed-16 (Partition RART) | 11.4M | 5.49 | 65.29 ± 0.94 | 16.73 ± 0.46 |
| ViT, 256² input | 9.3M | 5.28 | 61.99 ± 0.54 | **18.25 ± 0.33** |
| ViT, 128² input | 9.2M | 1.21 | 55.05 ± 0.26 | 12.49 ± 0.28 |
| PVTv2-B0 | 3.4M | 1.51 | 75.07 ± 0.25 | – |
| ResNet-18 (lr 1e-3) | 11.2M | 4.74 | 86.11 ± 0.26 | 37.07 ± 0.28 |

| Paired gap (95% CI) | Cluttered, 30 ep | Tiny ImageNet, 15 ep |
|---|---:|---:|
| focus-16 − fixed-16 | +1.54 [+0.53, +2.55] | +0.95 [−0.93, +2.83] |
| focus-16 − ViT 256² | +4.83 [+3.86, +5.80] | −0.58 [−2.11, +0.95] |
| focus-16 − ViT 128² | +11.77 [+9.73, +13.82] | +5.19 [+4.43, +5.95] |

![Test accuracy over 30 epochs on Cluttered CIFAR-10](paper/figures/curves.png)

The dense ViT has no local windows, so part of the anchor models' lead over it may come from
locality. The effect of moving the anchors alone is the focus − fixed gap.

### Anchor collapse (experiment 03)

Five anchor designs on one shared backbone (CIFAR-100, 224², 10 epochs, 3 seeds):

* Anchors whose queries do not depend on the image (RART-style, AnchorFormer-style) collapse to
  a single summary: pooling-map similarity 0.998 and 0.990.
* Image-derived anchors (competitive slots, PaCa-style) are only partly diverse (0.79 and 0.59),
  and they grow more similar with depth.
* The fixed partition has zero map overlap by construction, and it is the most accurate:
  46.31 ± 0.56%, against 40.93 ± 0.95% for a dense pathway.

### Ablations and sensitivity (experiment 06, focus-16, 10 epochs)

| Variant | Accuracy (%) | Note |
|---|---:|---|
| Full model, clip + warmup | 53.78 ± 0.09 | reference |
| No suppression, clip + warmup | 53.57 ± 0.43 | −0.21 [−1.51, +1.08]; one seed diverged without clipping |
| No home prior (no clipping) | 47.23 ± 1.17 | −2.6 against 49.84 |
| No coverage grid / 3×3 grid (no clipping) | 49.63 / 49.40 | no measurable effect |
| Soft assignment, clip + warmup | 10.00 × 3 | diverged in every seed |
| Soft + Sinkhorn balancing, clip + warmup | 9.98 / 10.00 / 12.44 | peaked at 26–39%, then diverged |

Placement is robust near its defaults. In one seed per setting, β = 3, σ_s = 0.75 and
σ_h ∈ {0.15, 1.0} all stay within 1.5 points of the reference. It breaks when suppression is
much stronger (β = 30: −4.4 points) or much wider (σ_s = 3 patches: stalls near 20%).

### Compute skipping (experiments 05 and 06)

Letting 75% of patches skip the FFN in blocks 3–6 cuts compute by 25% (5.31 → 3.96 GFLOPs).
Random selection costs only 0.38 points (53.40 ± 0.35%, clip + warmup). Selection by the learned
importance map is not better than random or token-norm selection, and a bounded
straight-through importance gate diverges.

### Cost

At 256², measured on an A100 (bf16, batch 128):

| Model | Train step | Peak train memory | Inference |
|---|---:|---:|---:|
| ViT 256² | 32.4 ms | 3,111 MB | 11,176 img/s |
| fixed-16 | 60.7 ms | 4,314 MB | 6,408 img/s |
| focus-16 | 99.3 ms | 4,528 MB | 5,259 img/s |

The sequential placement of focus anchors adds about 3–4 ms per focus anchor per step.

At much higher resolution the picture changes. In experiment 02 (inference, batch 1, no
FlashAttention), Partition RART overtook a dense ViT in latency between 1280² and 1536². At 2048²
it used 2.4 GB of peak memory against 14.5 GB for the ViT. PVTv2-B1 and EfficientViT-B2 were
still faster at 2048² (779 ms and 205 ms, against 1,756 ms).

### Early experiments (01–02)

On CIFAR-100 at 224² (10 epochs, 1 seed), Partition RART reached 44.62%, against 45.44% for a
ViT. Unrestricted anchors (RART-16/32) reached 42.21% and 39.73%. Their anchors collapsed:
pairwise similarity 0.99, and each anchor attended to 194.8 of 196 patches. These results
motivated experiments 03–06. Details are in [`docs/research_summary.md`](docs/research_summary.md).

---

## Limitations

* The tasks are small and synthetic, and all models are trained from scratch for 10–30 epochs.
  Absolute accuracy is low, and a CNN and PVTv2 are far ahead.
* Three seeds give wide confidence intervals, and some skipping rows and the sensitivity sweep
  use one or two seeds.
* There is no validation split, and the 30-epoch learning rate was chosen on test accuracy.
* The AnchorFormer- and PaCa-style models are simplified reimplementations, not the published
  models.
* The importance map marks content in general rather than the class object.
* The implementation is not optimised. At moderate resolution it is slower than dense attention,
  and FlashAttention may narrow the memory gap at high resolution.
* Real high-resolution data (for example traffic signs in full-HD street scenes) has not been
  tested yet.

---

## Related work

Anchor-based token bottlenecks are not new. The closest prior work for each component:

* **Fixed regional tokens:** [RegionViT](https://arxiv.org/abs/2106.02689) associates each
  regional token with a fixed group of local tokens, close to the fixed partition here.
* **Anchor bottlenecks (gather, interact, distribute):**
  [AnchorFormer](https://arxiv.org/abs/2505.16463), [LeMeViT](https://arxiv.org/abs/2405.09789),
  [PaCa-ViT](https://arxiv.org/abs/2203.11987),
  [Representative Attention](https://arxiv.org/abs/2605.14913).
* **Content-driven regions:** [SViT super tokens](https://arxiv.org/abs/2211.11167),
  [SALG](https://arxiv.org/abs/2211.14705), [TCFormer / FTCFormer](https://arxiv.org/abs/2507.10283).
* **Learned movement towards important regions:** [DAT](https://arxiv.org/abs/2201.00520),
  [PS-ViT](https://arxiv.org/abs/2108.01684).
* **High-resolution recognition under a memory budget:**
  [Differentiable Patch Selection](https://arxiv.org/abs/2104.03059),
  [Iterative Patch Selection](https://arxiv.org/abs/2210.13007).

What this project contributes is the combination: an exclusive partition that prevents collapse
by construction, anchors that move to content without giving it up, and a controlled
comparison of collapse across anchor designs.

---

## Repository layout

```text
README.md                        this overview
docs/research_summary.md         detailed log of experiments 01–02
experiments/
  README.md                      index of all experiments and how to run them
  01_partition_rart/             ViT, RART, Local-Global-Local and Partition RART on CIFAR-100
  02_resolution_scaling/         latency and memory from 224² to 2048², efficient baselines
  03_anchor_collapse/            anchor collapse across five anchor designs
  04_moving_anchors/             fixed, moving and focus anchors on Cluttered CIFAR-10
  05_compute_skipping/           FFN skipping guided by the learned importance map
  06_main_comparison/            baselines, 30-epoch and Tiny ImageNet runs, ablations, cost
paper/                           LNCS paper draft (main.tex, refs.bib, figures/)
```

Each experiment folder has a README covering the question, setup, results and findings, plus its
code or notebook and a `results/` folder. Model checkpoints are not committed.

## Reproducing

* **Experiments 01–02:** Kaggle notebooks (PyTorch 2.10, CUDA 12.8, Tesla T4).
* **Experiments 03–05:** notebooks that run on Kaggle or locally (PyTorch 2.11, RTX 5060 Laptop
  GPU). They download CIFAR if it is not found under `/kaggle/input`.
* **Experiment 06:** `train.py` and `run_suite.py`, configured through `RART_*` environment
  variables. These were run on a rented A100-80GB in bf16, with phase V on the laptop GPU.

Settings and commands for each experiment are in [`experiments/README.md`](experiments/README.md)
and [`experiments/06_main_comparison/README.md`](experiments/06_main_comparison/README.md).

---

## Roadmap

* [x] ViT baseline, unrestricted anchors, Local-Global-Local and Partition RART (01)
* [x] Resolution scaling to 2048² against dense and efficient transformers (02)
* [x] Anchor collapse across five designs, 3 seeds, with a dense reference (03)
* [x] Moving and focus anchors on Cluttered CIFAR-10, 3 seeds (04)
* [x] Compute skipping against random and token-norm selection (05)
* [x] 30-epoch, Tiny ImageNet, CNN and PVTv2 baselines; ablations; cost (06)
* [x] Sinkhorn-balanced soft assignment and placement sensitivity (06, phase V)
* [ ] Dense baseline with local windows
* [ ] Real high-resolution data (DFG traffic signs, Megapixel MNIST) with a fair memory comparison
* [ ] Comparison with the published AnchorFormer
* [ ] Paper submission

---

## Citation

```bibtex
@misc{gupta2026partitionrart,
  title  = {Anchors That Move Without Collapsing: Exclusive-Region Anchor Routing for Vision Transformers},
  author = {Aadi Gupta},
  year   = {2026},
  note   = {Work in progress},
  url    = {https://github.com/ARRY7686/Partitioned-Residual-Anchor-Routing-Transformer}
}
```
