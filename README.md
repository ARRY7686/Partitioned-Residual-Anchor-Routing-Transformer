# Partitioned Local-Global-Local Vision Transformer

A research implementation investigating whether **global communication in Vision Transformers can be performed through spatially specialized regional anchor tokens instead of dense patch-to-patch attention**.

The project explores a Local → Global → Local architecture in which image patches first interact locally, communicate with a small set of spatially assigned anchors, allow those anchors to exchange information globally, and finally propagate the global information back to their corresponding regions.

> **Research status:** Experimental / ongoing research

---

## Overview

Standard Vision Transformers use global self-attention between all image patches. For an image represented by \(N\) patches, this introduces an \(O(N^2)\) attention interaction structure.

This becomes increasingly expensive as image resolution grows.

This project investigates an alternative:

```text
                    Image
                      │
                      ▼
                Patch Embedding
                      │
                      ▼
             Local Patch Attention
                      │
                      ▼
             Spatial Partitioning
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     Regional Anchor 1       Regional Anchor N
          │                       │
          └───────────┬───────────┘
                      ▼
             Global Anchor Attention
                      │
                      ▼
             Regional Anchor → Patch
                      │
                      ▼
                    FFN
                      │
                      ▼
                Next Block
```

The central idea is to separate:

* **Local representation learning** between nearby patches
* **Global communication** between a small number of regional anchors
* **Local reconstruction / information propagation** from anchors back to patches

---

## Research Question

The project investigates three related questions:

1. Can anchor tokens replace dense global patch-to-patch communication?
2. Do unrestricted anchors collapse into redundant representations?
3. Can spatially constrained routing preserve anchor specialization while retaining global communication?

A secondary hypothesis is that the computational benefits of this approach should become more apparent as image resolution increases.

---

# Architecture

## Partitioned Local-Global-Local

The current architecture consists of:

1. Patch embedding
2. Local window attention
3. Patch → regional anchor routing
4. Global anchor → anchor self-attention
5. Regional anchor → patch routing
6. Feed-forward network

Each patch belongs to exactly one spatial region.

For example, at 224×224:

```text
224 × 224 image
      │
      ▼
14 × 14 patch grid
      │
      ▼
196 patches
      │
      ▼
16 spatial regions
      │
      ▼
16 regional anchors
      │
      ▼
16 × 16 global anchor attention
```

The spatial partition is shared across transformer blocks.

---

## Why Regional Anchors?

An initial version allowed every anchor to attend to the entire patch set.

Although this dramatically compressed the number of global tokens, experiments revealed a significant problem: **anchor collapse**.

For the trained 32-anchor model:

| Metric                           |       Result |
| -------------------------------- | -----------: |
| Mean anchor similarity, Block 1  |       0.9887 |
| Mean anchor similarity, Block 6  |       0.9950 |
| Patch → Anchor attention entropy |       5.2720 |
| Maximum entropy for 196 patches  |      ≈ 5.278 |
| Effective patches attended       | 194.82 / 196 |
| Anchor attention-map similarity  |       0.9995 |

The anchors were therefore receiving almost identical global information.

This motivated explicit spatial routing.

---

# Spatial Partitioning

Instead of allowing every anchor to access every patch, each patch is assigned to exactly one regional anchor.

At 224×224:

* 196 patches
* 16 anchors
* 16 spatial regions
* Region sizes: 9–16 patches
* Mean region size: 12.25 patches

This forces anchors to first summarize different spatial regions before communicating globally.

The resulting architecture is:

```text
Patch
  │
  ▼
Local Attention
  │
  ▼
Assigned Regional Anchor
  │
  ▼
Global Anchor Attention
  │
  ▼
Assigned Anchor
  │
  ▼
Patch
```

---

# Experiments

## Dataset

Initial experiments use **CIFAR-100**:

* 50,000 training images
* 10,000 test images
* 100 classes
* Images resized to 224×224
* Patch size: 16×16

The high-resolution experiments are primarily intended to study **computational scaling**, rather than to claim that CIFAR-100 itself is a high-resolution vision benchmark.

---

# Baseline

The baseline is a custom ViT with:

| Configuration    |   Value |
| ---------------- | ------: |
| Embed dimension  |     384 |
| Depth            |       6 |
| Attention heads  |       6 |
| MLP ratio        |       4 |
| Patch size       |   16×16 |
| Input resolution | 224×224 |
| Patches          |     196 |

Baseline:

* Parameters: **11,057,380**
* Test accuracy after 10 epochs: **45.44%**
* Inference: **201.345 ms**
* Peak GPU memory: **2039.25 MB**
* Throughput: **635.73 images/sec**

---

# Initial RART

The first anchor-based implementation used unrestricted compressed Patch → Anchor routing.

### RART-16

* 16 anchors
* Anchor dimension: 128
* Patch dimension: 384
* Depth: 6

Results:

* Parameters: **11.27M**
* Test accuracy: **42.21%**
* Inference: **212.667 ms**
* Peak memory: **2619.68 MB**
* Throughput: **601.88 images/sec**

### RART-32

Results:

* Parameters: **11.27M**
* Test accuracy: **39.73%**
* Inference: **215.096 ms**
* Peak memory: **2802.46 MB**
* Throughput: **595.08 images/sec**

The degradation motivated the anchor-collapse investigation.

---

# Local-Global-Local Experiment

A constrained 5×5 routing version was introduced:

1. Local patch attention
2. Local Patch → Anchor routing
3. Global Anchor → Anchor attention
4. Local Anchor → Patch routing
5. FFN

Results:

| Metric        |            Result |
| ------------- | ----------------: |
| Parameters    |            11.27M |
| Test accuracy |        **45.38%** |
| Inference     |        399.040 ms |
| Peak memory   |        4000.50 MB |
| Throughput    | 320.77 images/sec |

Anchor similarity remained substantially lower than in the unrestricted model:

| Block | Anchor similarity |
| ----- | ----------------: |
| 1     |            0.2619 |
| 2     |            0.2550 |
| 3     |            0.2976 |
| 4     |            0.3295 |
| 5     |            0.3682 |
| 6     |            0.4227 |

This provided evidence that spatially constrained routing can preserve anchor specialization.

---

# Partition RART

The current main architecture uses a fixed non-overlapping spatial partition.

### Configuration

* Image: 224×224
* Patch size: 16×16
* Patches: 196
* Anchors: 16
* Embed dimension: 384
* Anchor dimension: 128
* Depth: 6
* Patch heads: 6
* Anchor heads: 4
* MLP ratio: 3

Results after 10 epochs:

| Model              | Parameters | Test Accuracy |
| ------------------ | ---------: | ------------: |
| ViT                |     11.06M |    **45.44%** |
| MeanPool ViT       |     11.06M |        44.59% |
| RART-16            |     11.27M |        42.21% |
| RART-32            |     11.27M |        39.73% |
| Local-Global-Local |     11.27M |    **45.38%** |
| Partition RART     |     11.27M |    **44.62%** |

Partition RART is within **0.82 percentage points** of the baseline ViT on this experiment.

The result does **not** establish that Partition RART is more accurate than ViT. The primary motivation is its different global communication structure and its scaling behavior.

---

# Global Interaction Reduction

At 224×224:

### Dense ViT

There are 196 patches.

Dense self-attention involves approximately:

$$
196^2 = 38,416
$$

patch-to-patch interactions per attention layer.

### Partition RART

Patch → Anchor communication:

$$
196
$$

Anchor → Anchor communication:

$$
16^2 = 256
$$

Total:

$$
196 + 256 = 452
$$

Thus, the global communication component uses approximately:

$$
\frac{38,416}{452} \approx 85\times
$$

fewer pairwise interactions.

This is an **interaction-count comparison**, not an equivalent 85× wall-clock speedup.

Local attention, projections, FFNs, memory movement, kernel launch overhead, and implementation details still contribute substantially to runtime.

---

# High-Resolution Scaling

The main computational hypothesis is that dense global attention becomes increasingly expensive as the number of patches grows.

The number of anchors is therefore scaled with the spatial resolution.

| Resolution | Patches | Anchors |
| ---------: | ------: | ------: |
|       224² |     196 |      16 |
|       384² |     576 |      36 |
|       512² |   1,024 |      64 |
|       768² |   2,304 |     144 |
|      1024² |   4,096 |     256 |
|      1280² |   6,400 |     400 |
|      1536² |   9,216 |     576 |
|      2048² |  16,384 |   1,024 |

---

## ViT vs Partition RART

Matched high-resolution inference measurements:

| Resolution |   RART Latency |    ViT Latency | RART Throughput | ViT Throughput |
| ---------: | -------------: | -------------: | --------------: | -------------: |
|       224² |       38.80 ms |        6.63 ms |         51.55/s |       301.66/s |
|       384² |       77.92 ms |       12.90 ms |         25.67/s |       155.05/s |
|       512² |      128.92 ms |       28.41 ms |         15.51/s |        70.41/s |
|       768² |      286.45 ms |      101.09 ms |          6.98/s |        19.78/s |
|      1024² |      505.65 ms |      265.06 ms |          3.96/s |         7.55/s |
|      1280² |      790.15 ms |      641.85 ms |          2.53/s |         3.12/s |
|      1536² | **1155.42 ms** | **1416.91 ms** |      **1.73/s** |     **1.41/s** |

The measured crossover occurs between **1280×1280 and 1536×1536** in this implementation.

At 1536×1536:

* RART: **1155.4 ms**
* ViT: **1416.9 ms**
* RART peak memory: **1253 MB**
* ViT peak memory: **8897 MB**

This corresponds to approximately 18.5% lower latency for RART in this particular measurement and substantially lower peak memory.

These measurements should be interpreted as implementation-specific benchmarks rather than universal complexity guarantees.

---

# 2048×2048 Experiment

At 2048×2048:

* 16,384 patches
* 1,024 regional anchors
* 16 patches per anchor

### Partition RART

| Metric      |              Result |
| ----------- | ------------------: |
| Parameters  |              17.62M |
| Latency     |       **1755.5 ms** |
| Peak memory |       **2428.9 MB** |
| Throughput  | **0.57 images/sec** |

### Dense ViT

Batch size 2 exceeded the available GPU memory.

A batch-size-1 measurement was therefore used:

| Metric      |          Result |
| ----------- | --------------: |
| Parameters  |          17.27M |
| Latency     |       2519.3 ms |
| Peak memory |      14494.7 MB |
| Throughput  | 0.40 images/sec |

At batch size 1, RART used approximately 83% less peak memory than the dense ViT in this experiment.

---

# Efficient Transformer Baselines

Additional high-resolution baselines were investigated using `timm`.

At 2048×2048, batch size 1:

| Model           | Parameters |      Latency |     Memory | Throughput |
| --------------- | ---------: | -----------: | ---------: | ---------: |
| Partition RART  |     17.62M |    1755.5 ms |  2428.9 MB |     0.57/s |
| PVTv2-B1        |     13.55M |     778.9 ms |  3835.4 MB |     1.28/s |
| EfficientViT-B2 |     22.03M | **204.8 ms** |  2907.4 MB | **4.88/s** |
| ViT             |     17.27M |    2519.3 ms | 14494.7 MB |     0.40/s |

These results are important because they show that reducing dense attention alone does not make Partition RART the fastest efficient-transformer architecture.

The current evidence instead supports a narrower observation:

> Spatially partitioned anchor routing can substantially reduce memory usage relative to dense ViT attention at very high resolutions while maintaining a relatively small global communication graph.

---

# Complexity Intuition

Let:

* \(N\) = number of patches
* \(A\) = number of anchors

Dense ViT global attention has approximately:

$$
O(N^2)
$$

patch interactions.

Partition RART separates communication into:

### Local attention

Local computation is restricted to spatial windows.

### Patch → Anchor

Each patch communicates with its assigned anchor:

$$
O(N)
$$

### Anchor → Anchor

Global communication occurs among anchors:

$$
O(A^2)
$$

Therefore, the global routing component is approximately:

$$
O(N + A^2)
$$

instead of:

$$
O(N^2)
$$

When the number of anchors grows much more slowly than the number of patches, this creates a substantially smaller global interaction graph.

---

# Key Findings So Far

### 1. Unrestricted anchors can collapse

The initial compressed anchor architecture produced extremely similar anchor representations.

The trained 32-anchor model reached approximately **0.99 average pairwise cosine similarity** between anchors.

---

### 2. Spatial constraints improve specialization

Constraining patches to regional anchors dramatically reduced attention-map similarity and allowed different anchors to represent different spatial regions.

---

### 3. Local → Global → Local recovers classification performance

The constrained Local-Global-Local model achieved **45.38%** test accuracy, close to the **45.44%** ViT baseline in the initial experiment.

---

### 4. Partition RART provides a compact global communication graph

At 224×224:

```text
ViT:
196 × 196 = 38,416 interactions

Partition RART:
196 patch→anchor
+
16 × 16 anchor→anchor
=
452 interactions
```

---

### 5. The computational benefit appears at higher resolution

In the current implementation, RART is slower than the dense ViT at lower resolutions but becomes competitive at higher resolutions.

The measured crossover is between **1280² and 1536²**.

At 2048², dense ViT attention approaches the GPU memory limit, while Partition RART remains executable with substantially lower memory usage.

---

# Limitations

This project is still experimental and has several important limitations.

### CIFAR-100

The initial classification experiments use CIFAR-100 resized to high resolutions. This is useful for controlled architectural experiments but does not demonstrate effectiveness on genuinely high-resolution imagery.

### Implementation efficiency

The current implementation uses explicit routing and indexing operations. The theoretical reduction in attention interactions does not automatically translate into equivalent GPU speedups.

### Baseline coverage

Several efficient-transformer architectures exist, and the current experiments do not constitute an exhaustive comparison.

### Accuracy

Partition RART has not yet demonstrated superior classification accuracy over the baseline ViT.

### High-resolution task validation

The strongest motivation for this architecture is high-resolution vision, but this still needs to be evaluated on genuinely high-resolution tasks such as dense prediction or small-object detection.

---

# Research Direction

The next stage of the project is to evaluate the architecture on tasks where reducing image resolution can remove important information.

Potential targets include:

* High-resolution object detection
* Tiny-object detection
* Aerial imagery
* Remote sensing
* Dense prediction
* Medical imaging

A particularly relevant research question is whether spatially partitioned anchors can provide global context **without aggressively downsampling high-resolution images**.

---

# Relation to Existing Work

Anchor-based token bottlenecks are not themselves new.

Related approaches include:

* Vision Transformers
* Perceiver
* TokenLearner
* Slot Attention
* Focal Transformer
* Anchor-based efficient transformers
* Hierarchical vision transformers

Closest prior work for specific components:

* **Fixed regional tokens:** [RegionViT](https://arxiv.org/abs/2106.02689) already associates each regional token with a fixed spatial group of local tokens and lets the regional tokens attend globally, which is close to the fixed partition used here.
* **Anchor bottlenecks:** [AnchorFormer](https://arxiv.org/abs/2505.16463), [LeMeViT](https://arxiv.org/abs/2405.09789), [PaCa-ViT](https://arxiv.org/abs/2203.11987) and [Representative Attention](https://arxiv.org/abs/2605.14913) (gather–interact–distribute).
* **Content-driven regions:** [SViT super tokens](https://arxiv.org/abs/2211.11167), [SALG](https://arxiv.org/abs/2211.14705), [TCFormer / FTCFormer](https://arxiv.org/abs/2507.10283).
* **Learned movement toward important regions:** [DAT](https://arxiv.org/abs/2201.00520), [PS-ViT](https://arxiv.org/abs/2108.01684).
* **High-resolution recognition under a memory budget:** [Differentiable Patch Selection](https://arxiv.org/abs/2104.03059), [Iterative Patch Selection](https://arxiv.org/abs/2210.13007).

The project therefore does **not** claim that using anchors, reducing quadratic attention, or a fixed spatial partition is novel by itself.

The current research direction focuses on the combination of:

1. Explicit, exclusive spatial partitioning
2. Anchors that move to important regions while keeping that partition
3. Local → Global → Local information flow
4. Analysis of anchor collapse across anchor designs
5. High-resolution scaling behavior

Further direct comparisons with related anchor-based architectures are required before making a formal novelty claim.

---

# Follow-up Experiments

The [`experiments/`](experiments/) folder contains self-contained notebooks for the next stage:

* **Step 0 — anchor collapse across designs.** RART-style, AnchorFormer-style and PaCa-style anchors, a competitive (slot-style) variant and the fixed partition, trained on one shared backbone. Anchors that are the same for every image collapse completely (map similarity ≈ 0.99), image-derived anchors are only partly diverse, and the fixed partition does not collapse and reached the best accuracy in both seeds (45.68% / 46.48%).
* **Step 1–2 — moving anchors.** Positional anchors with an exclusive Voronoi partition on Cluttered CIFAR-10 (a small object on a large canvas with distractors). A *focus* design — a quarter of the anchors on a coarse fixed grid for coverage, the rest placed at the peaks of a learned importance map — was the best model with 4 and 16 anchors (48.79% / 49.63%, against 47.08% / 48.10% for the fixed partition) without collapsing. This is one seed; more are needed before the gap can be claimed.

See [`experiments/README.md`](experiments/README.md) for setup, full tables and how to run them.

---

# Reproducing the Experiments

The experiments were developed in a Kaggle GPU environment.

The main environment used during development included:

```text
Python 3.12
PyTorch 2.10
CUDA 12.8
Tesla T4
```

The notebooks contain the model definitions, experiments, benchmarks, and analysis.

For the CIFAR-100 experiments, the dataset should be available locally in the Kaggle input environment because the notebook is designed to work without relying on network downloads.

---

# Citation

This repository currently represents an ongoing research project.

If you use the implementation or experimental findings, please cite the repository once a formal paper or technical report is released.

```bibtex
@misc{partitioned_rart,
  title  = {Partitioned Local-Global-Local Vision Transformer},
  author = {Aadi Gupta, Priyansh Saxena},
  year   = {2026},
  note   = {Ongoing research project}
}
```

---

# Status

🚧 **Research in progress**

Current focus:

* [x] ViT baseline
* [x] Initial anchor-based architecture
* [x] Anchor-collapse analysis
* [x] Spatial routing investigation
* [x] Local-Global-Local architecture
* [x] Partitioned routing
* [x] High-resolution scaling experiments
* [x] 2048×2048 experiment
* [x] Efficient transformer baseline comparison
* [x] Anchor-collapse comparison across anchor designs (AnchorFormer-style, PaCa-style, competitive)
* [x] Moving anchors on a small-object task, with anchor-trajectory visualization (1 seed)
* [ ] Multi-seed confirmation of the moving-anchor results
* [ ] Direct comparison with the published AnchorFormer
* [ ] High-resolution detection benchmark
* [ ] Tiny-object evaluation
* [ ] Training scalability experiments
* [ ] Formal ablation study
* [ ] Paper / technical report

---

## Core Hypothesis

The central hypothesis of this project is:

> **Global reasoning in high-resolution vision may not require every image patch to communicate directly with every other patch. Spatially specialized regional anchors can provide a compact global communication layer while local attention preserves fine-grained spatial information.**

The experiments so far suggest that **spatial specialization is important**: unrestricted anchor compression can produce severe anchor redundancy, while explicit regional routing encourages distinct anchor representations.

Whether this translates into a meaningful advantage on real high-resolution vision tasks remains an open research question.
