

**# Partitioned Local-Global-Local Vision Transformer**

**## Research Objective**

Investigate whether global communication in Vision Transformers can be performed through a small number of spatially specialized anchor tokens instead of dense patch-to-patch attention.

The initial motivation was that standard ViT global attention has O(N^2) complexity with respect to the number of image patches. The proposed architecture attempts to separate local representation learning from global communication.

The core architecture is:

Patch Embedding

    ↓

Local Patch Attention

    ↓

Patch → Regional Anchor

    ↓

Global Anchor → Anchor Attention

    ↓

Regional Anchor → Patch

    ↓

FFN

    ↓

Next Block

**## Baseline**

Dataset:

\- CIFAR-100

\- 50,000 training images

\- 10,000 test images

\- 100 classes

\- Images resized to 224×224

\- Patch size: 16×16

\- 196 patches

ViT configuration:

\- Embed dimension: 384

\- Depth: 6

\- Heads: 6

\- MLP ratio: 4

Baseline results:

\- Parameters: 11,057,380

\- Inference: 201.345 ms

\- Peak GPU memory: 2039.25 MB

\- Throughput: 635.73 images/sec

\- Test accuracy after 10 epochs: 45.44%

**## Initial Anchor-Based RART**

The first architecture used local patch attention together with a compressed anchor pathway.

Configuration:

\- 16 anchors

\- Anchor dimension: 128

\- Patch dimension: 384

\- Depth: 6

Results:

\- Parameters: 11,272,176

\- Test accuracy: 42.21%

\- Inference: 212.667 ms

\- Peak memory: 2619.68 MB

\- Throughput: 601.88 images/sec

A 32-anchor version was also tested.

Results:

\- Parameters: 11,274,224

\- Test accuracy: 39.73%

\- Inference: 215.096 ms

\- Peak memory: 2802.46 MB

\- Throughput: 595.08 images/sec

**## Anchor Collapse Investigation**

The lower performance led to an investigation of anchor representations.

For the trained 32-anchor model:

Final anchor cosine similarity:

\- Block 1: 0.9887

\- Block 2: 0.9865

\- Block 3: 0.9900

\- Block 4: 0.9920

\- Block 5: 0.9935

\- Block 6: 0.9950

Patch-to-anchor attention was nearly uniform:

\- Mean attention entropy: 5.2720

\- Maximum possible entropy for 196 patches: approximately 5.278

\- Effective patches attended: 194.82 / 196

\- Mean similarity between anchor attention maps: 0.9995

This indicated that different anchors were receiving almost identical global summaries.

The collapse originated at Patch → Anchor routing.

**## Spatial Routing Investigation**

Raw regional patch features were already diverse:

\- Mean regional feature similarity: 0.3466

A hard local 5×5 routing constraint produced:

\- Mean anchor attention-map similarity: 0.0389

\- Effective patches attended: 16 / 196

This demonstrated that spatial constraints could force anchor specialization.

**## Local-Global-Local Architecture**

A Local → Global → Local architecture was developed:

1\. Local patch attention

2\. Local Patch → Anchor routing

3\. Global Anchor → Anchor self-attention

4\. Local Anchor → Patch routing

5\. Patch FFN

The 5×5 local-routing version achieved:

\- Parameters: 11,272,176

\- Test accuracy: 45.38%

\- Inference: 399.040 ms

\- Peak memory: 4000.50 MB

\- Throughput: 320.77 images/sec

Anchor similarity after training:

\- Block 1: 0.2619

\- Block 2: 0.2550

\- Block 3: 0.2976

\- Block 4: 0.3295

\- Block 5: 0.3682

\- Block 6: 0.4227

This showed substantially greater anchor specialization than the unrestricted routing model.

**## Partition Routing**

The next architecture replaced overlapping local routing with a fixed non-overlapping spatial partition.

For a 14×14 patch grid:

\- 196 total patches

\- 16 spatial regions

\- Each patch belongs to exactly one region

\- Region sizes range from 9 to 16 patches

\- Mean region size: 12.25 patches

The partitioned architecture uses:

Local Patch Attention

    ↓

Patch → Assigned Regional Anchor

    ↓

Global Anchor Self-Attention

    ↓

Assigned Anchor → Patch

    ↓

FFN

The partition is cached and shared across all blocks.

**## Partition RART Results**

Configuration:

\- Image size: 224×224

\- Patch size: 16

\- 196 patches

\- Embed dimension: 384

\- Anchor dimension: 128

\- Depth: 6

\- Patch heads: 6

\- Anchor heads: 4

\- Anchors: 16

\- MLP ratio: 3

Parameters:

\- 11,272,176

Inference:

\- 221.430 ms

\- 3982.33 MB peak GPU memory

\- 578.06 images/sec

Training:

\- Epoch 1: 16.60% test accuracy

\- Epoch 2: 24.19%

\- Epoch 3: 28.90%

\- Epoch 4: 32.55%

\- Epoch 5: 35.73%

\- Epoch 6: 38.35%

\- Epoch 7: 40.08%

\- Epoch 8: 42.19%

\- Epoch 9: 42.85%

\- Epoch 10: 44.62%

Final:

\- Train accuracy: 53.62%

\- Test accuracy: 44.62%

**## Current Comparison**

\| Model | Parameters | Test Accuracy | Inference |

\|---|---:|---:|---:|

\| ViT | 11.06M | 45.44% | 201.3 ms |

\| MeanPool ViT | 11.06M | 44.59% | — |

\| RART-16 | 11.27M | 42.21% | 212.7 ms |

\| RART-32 | 11.27M | 39.73% | 215.1 ms |

\| Local-Global-Local | 11.27M | 45.38% | 399.0 ms |

\| Partition RART | 11.27M | 44.62% | 221.4 ms |

**## Current Interpretation**

The unrestricted anchor routing mechanism caused different anchors to converge toward highly similar representations. This corresponded with lower classification performance.

Introducing explicit spatial routing substantially improved anchor diversity and recovered near-baseline classification accuracy.

The partitioned model reaches 44.62% test accuracy compared with 45.44% for the baseline ViT, a difference of 0.82 percentage points.

The current results therefore support the hypothesis that spatially constrained anchor routing can preserve useful regional specialization while still providing global communication through anchor self-attention.

**## Computational Observation**

At 224×224 with a patch size of 16×16, the input is represented by 196 image patches arranged on a 14×14 grid.

**### ViT**

Standard ViT self-attention allows every patch to interact with every other patch. Therefore, each attention layer computes approximately:

$$

196^2 = 38,416

$$

patch-to-patch attention interactions.

**### Partition RART**

Partition RART divides the 196 patches into 16 non-overlapping spatial regions. Each patch is assigned to exactly one regional anchor.

The regions contain between 9 and 16 patches, with an average of:

$$

\frac{196}{16}=12.25

$$

patches per region.

Consequently, Patch-to-Anchor routing requires only one regional interaction per patch, resulting in approximately:

$$

196

$$

Patch-to-Anchor interactions.

Global communication is then performed through self-attention among the 16 anchors:

$$

16^2 = 256

$$

Anchor-to-Anchor interactions.

Thus, the partitioned routing mechanism involves approximately:

$$

196 + 256 = 452

$$

pairwise interactions for Patch-to-Anchor and Anchor-to-Anchor communication, compared with 38,416 patch-to-patch interactions in dense ViT attention.

This represents approximately an **\*\*85× reduction in these global communication interactions\*\*** at 224×224.

However, this interaction-count reduction does not directly translate into wall-clock speedup. The current implementation still performs local attention, projection layers, feed-forward networks, and other operations, while GPU kernel and implementation overhead also contribute to runtime.

Measured inference performance at 224×224 is:

\* ViT: **\*\*201.3 ms\*\***, 635.7 images/sec

\* Partition RART: **\*\*221.4 ms\*\***, 578.1 images/sec

Therefore, although Partition RART substantially reduces the number of pairwise interactions involved in global communication, **\*\*computational superiority in actual wall-clock inference has not yet been demonstrated at 224×224\*\***.

The central computational hypothesis is instead that this reduced global interaction structure will scale more favorably as the number of image patches increases. This will be evaluated through experiments at higher spatial resolutions.



**## Important Literature Consideration**

Anchor-based token bottlenecks are not themselves novel. AnchorFormer and related architectures already investigate anchor tokens as a mechanism for reducing the cost of global ViT attention.

Therefore, the current work should not claim novelty simply from:

\- using anchors

\- replacing quadratic attention with anchors

\- global communication through a small latent set

The potentially differentiating component is the explicit spatial partitioning of patch-to-anchor communication combined with the Local → Global → Local architecture and the resulting preservation of anchor specialization.

This requires further literature comparison and direct implementation comparisons before making a novelty claim.

**## Next Experiment**

The original hypothesis concerns resolution scaling.

The next experiment should compare ViT and Partition RART at increasing image resolutions:

\- 224×224

\- 384×384

\- 512×512

\- potentially 768×768

The number of anchors should scale with spatial regions rather than remaining fixed indefinitely.

Metrics:

\- Accuracy

\- Inference latency

\- Throughput

\- Peak GPU memory

\- Training time

\- Number of patches

\- Effective global interactions

\- Anchor diversity

The key hypothesis is:

At lower resolution, dense ViT attention may remain competitive or superior.

At higher resolution, partitioned local-global-local routing may scale more favorably because global communication is performed through a much smaller anchor graph rather than dense patch-to-patch attention.

This remains a hypothesis and must be experimentally validated.

**## Conclusion**

The experiments show that naive anchor-based compression can suffer from anchor collapse, where multiple anchors learn nearly identical global summaries. Explicit spatial partitioning changes this behavior by assigning each patch to a single regional anchor, forcing the anchors to encode different spatial regions before communicating globally.

The resulting Partition RART achieves 44.62% test accuracy on CIFAR-100 at 224×224, compared with 45.44% for the baseline ViT, while using a similar parameter budget.

The strongest current evidence is therefore not that the proposed architecture already outperforms ViT, but that spatially constrained anchor routing provides a viable alternative mechanism for local-to-global information exchange without the severe anchor collapse observed in unrestricted routing.

The central unresolved question is whether this advantage becomes more significant as spatial resolution and patch count increase. High-resolution scaling experiments are required to determine whether the proposed architecture offers a genuine computational or representational advantage over dense ViT attention.
## High-Resolution Scaling Experiments

The resolution-scaling experiment was extended beyond the original 224×224 evaluation using resolution-agnostic versions of Partition RART and ViT.

The models used:

- Embed dimension: 384
- Depth: 6
- Patch size: 16×16
- Patch heads: 6
- Anchor dimension: 128
- Anchor heads: 4
- Partition RART MLP ratio: 3
- ViT MLP ratio: 4

### Resolution Scaling Results

| Resolution | Patches | Anchors | RART latency | ViT latency | RART throughput | ViT throughput |
|---|---:|---:|---:|---:|---:|---:|
| 224×224 | 196 | 16 | 38.80 ms | 6.63 ms | 51.55/s | 301.66/s |
| 384×384 | 576 | 36 | 77.92 ms | 12.90 ms | 25.67/s | 155.05/s |
| 512×512 | 1024 | 64 | 128.92 ms | 28.41 ms | 15.51/s | 70.41/s |
| 768×768 | 2304 | 144 | 286.45 ms | 101.09 ms | 6.98/s | 19.78/s |
| 1024×1024 | 4096 | 256 | 505.65 ms | 265.06 ms | 3.96/s | 7.55/s |
| 1280×1280 | 6400 | 400 | 790.15 ms | 641.85 ms | 2.53/s | 3.12/s |
| 1536×1536 | 9216 | 576 | 1155.42 ms | 1416.91 ms | 1.73/s | 1.41/s |

At 1536×1536, Partition RART became faster than the dense ViT baseline:

- RART: 1155.42 ms
- ViT: 1416.91 ms
- RART peak memory: 1253.19 MB
- ViT peak memory: 8897.27 MB

The measured latency crossover therefore occurred between 1280×1280 and 1536×1536.

### 2048×2048 Experiment

At 2048×2048:

- Image patches: 16,384
- Anchors: 1,024
- Patches per anchor: 16
- Partition RART parameters: 17,617,392
- ViT parameters: 17,273,572

Dense ViT attention requires:

$$
16,384^2 = 268,435,456
$$

global patch-patch interactions, while RART anchor self-attention requires:

$$
1,024^2 = 1,048,576
$$

anchor-anchor interactions.

This is approximately a **256× reduction in global pairwise interactions**.

The ViT could not execute a batch-2 forward pass at 2048×2048 on the 14.56 GB T4 GPU. It successfully executed at batch size 1.

For the apples-to-apples batch-1 comparison:

| Model | Parameters | Latency | Peak GPU memory | Throughput |
|---|---:|---:|---:|---:|
| Partition RART | 17.62M | 1755.51 ms | 2428.88 MB | 0.57/s |
| ViT | 17.27M | 2519.30 ms | 14494.73 MB | 0.40/s |

The batch-2 RART measurement was:

- Latency: 1983.66 ms
- Peak GPU memory: 1925.11 MB
- Throughput: 1.01 images/sec

The batch-2 ViT measurement resulted in CUDA out-of-memory.

These measurements provide evidence that Partition RART remains executable at a substantially higher resolution while dense ViT attention becomes strongly memory constrained.

## Comparison With Efficient Transformer Baselines

Additional high-resolution baselines were evaluated using `timm`.

### PVTv2-B1

- Parameters: 13,547,300
- 2048×2048 forward pass: successful
- Batch size: 1
- Latency: 778.923 ms
- Peak GPU memory: 3835.38 MB
- Throughput: 1.28 images/sec

### EfficientViT-B2

- Parameters: 22,025,812
- 2048×2048 forward pass: successful
- Batch size: 1
- Latency: 204.758 ms
- Peak GPU memory: 2907.38 MB
- Throughput: 4.88 images/sec

### 2048×2048 Comparison

| Model | Parameters | Latency | Peak GPU memory | Throughput |
|---|---:|---:|---:|---:|
| Partition RART | 17.62M | 1755.51 ms | 2428.88 MB | 0.57/s |
| ViT | 17.27M | 2519.30 ms | 14494.73 MB | 0.40/s |
| PVTv2-B1 | 13.55M | 778.92 ms | 3835.38 MB | 1.28/s |
| EfficientViT-B2 | 22.03M | 204.76 ms | 2907.38 MB | 4.88/s |

All four measurements in this table use batch size 1.

These results show that Partition RART is not currently the fastest efficient transformer at 2048×2048. EfficientViT-B2 and PVTv2-B1 have substantially lower measured latency in the tested implementations.

The strongest current computational observation for Partition RART is its memory behavior relative to dense ViT. At 2048×2048, RART used 2428.88 MB compared with 14494.73 MB for ViT.

At the same time, the current implementation does not establish a general inference-speed advantage over existing efficient transformer architectures.

### Swin-Tiny Investigation

Swin-Tiny was also investigated as a baseline.

The available `timm` model was `swin_tiny_patch4_window7_224` with 27,596,254 parameters and a native input size of 224×224.

At 224×224 it measured:

- Latency: 10.573 ms
- Peak GPU memory: 2106.88 MB
- Throughput: 94.58 images/sec

Attempts to run the fixed-resolution implementation at 512×512 encountered internal window-shape incompatibilities. The model was therefore not used for the high-resolution comparison rather than modifying the Swin implementation sufficiently to alter the baseline.

## Updated Interpretation

The comparison with efficient transformer baselines changes the computational framing of the work.

The contribution should not be presented simply as replacing quadratic attention with anchors, since existing architectures already provide alternatives to dense global attention.

The more specific research question is whether **spatially partitioned anchors combined with Local → Global → Local communication provide a useful accuracy, memory, and scaling trade-off compared with existing efficient transformer architectures.**

Current evidence supports:

1. Spatial partitioning prevents the severe anchor collapse observed with unrestricted anchor routing.
2. Partition RART recovers near-baseline CIFAR-100 accuracy at 224×224.
3. The number of global anchor interactions grows much more slowly than dense patch-to-patch interactions.
4. The measured latency crossover against dense ViT occurs between 1280×1280 and 1536×1536.
5. At 2048×2048, Partition RART uses substantially less GPU memory than dense ViT and remains executable at batch size 2.
6. PVTv2-B1 and EfficientViT-B2 currently achieve lower inference latency than the RART implementation at 2048×2048.
7. Further work should focus on both architectural efficiency and implementation efficiency before making broader efficiency claims.

## Revised Next Experiments

The next stage should compare Partition RART against efficient transformer baselines across multiple resolutions.

Important experiments include:

- PVTv2-B1 at 224×224, 512×512, 1024×1024, 1536×1536 and 2048×2048
- EfficientViT-B2 at the same resolutions
- Accuracy comparison under matched training conditions
- Peak memory scaling
- Latency scaling
- Throughput scaling
- Parameter-matched comparisons
- Anchor count ablations
- Region-size ablations
- Anchor diversity measurements
- Ablation of local patch attention
- Ablation of anchor self-attention
- Ablation of the partition constraint
- Training-time and inference-time profiling

A particularly important experiment is to determine whether current RART latency is dominated by Python-level indexing and partition routing rather than by the theoretical attention structure. A more optimized implementation could materially change measured efficiency.

The central hypothesis remains:

> Spatially partitioned anchors can provide specialized regional summaries and global communication while avoiding dense patch-to-patch global attention.

Whether this produces a useful practical advantage over existing efficient transformer architectures remains unresolved and requires controlled accuracy and efficiency comparisons.
