# 07 — High resolution

**Question.** On images where downsampling destroys the signal, do focus anchors beat the fixed
partition, dense attention with local windows, and ViTs on downsampled inputs? And at what
resolution, if any, do anchors save memory or time against dense attention with a fused
(FlashAttention) kernel?

**Status:** code ready, not yet run.

## Tasks

| Task | Images | Classes / metric | Why downsampling hurts |
|---|---|---|---|
| Megapixel MNIST (`mpmnist`), after Katharopoulos & Fleuret (2019) | 1024² canvas; 5000 train / 1000 val / 1000 test | 10, accuracy | Five native 28 px digits (three of the target class, two of another) among 50 stroke-like noise patches. At 256² a digit is 7 px. |
| DFG traffic signs (`dfg`), Tabernik & Skočaj (2019) | 1920×1080 street scenes (padded to 1920×1152); 5254 train / 1703 test frames | 20 most frequent sign types, multi-label mAP | Median sign about 120–150 px wide, many much smaller. At 480 px wide a small sign is a few pixels. |

Details:

* **Megapixel MNIST.** Our generator follows the original recipe at 1024² instead of 1500², so
  the canvas divides into 16 px patches and 8×8 windows. The noise patches are our own: one or two
  random line segments each. Validation digits come from MNIST training images 50,000–59,999, and
  test digits from the MNIST test set. Layouts are fixed per split.
* **DFG.** The official train/test split is kept. Note that it interleaves frames from the same
  video sequences, so train and test images can be near-duplicates. The validation split takes
  every 7th block of 100 consecutive frame numbers from the training images, so neighbouring
  frames stay together. Sign annotations flagged `ignore` are excluded from the loss and the mAP
  for that image and class. There is no augmentation, because flipping would change the meaning of
  some signs.

Both tasks have a validation split, and every run reports validation and test scores per
epoch. Settings are fixed in advance:

* transformers and PVTv2: AdamW, learning rate 2e-4, batch 32;
* ResNet-18: learning rate 1e-3;
* all models: gradient clip 1.0, 2-epoch warmup, cosine decay, bf16;
* epochs: 30 for Megapixel MNIST and 20 for DFG.

## Models

| Name | What it is |
|---|---|
| `focus_A64` | 16 coverage anchors (4×4) and 48 focus anchors, exclusive Voronoi regions |
| `fixed_A64` | 64 anchors on a fixed 8×8 grid (Partition RART) |
| `densewin` | The same blocks with the anchor pathway replaced by global attention among all patches, local windows kept. This is the dense baseline with windows that the reviews asked for. |
| `vit_native` | Dense ViT on the full-resolution input |
| `vit512`, `vit256` / `vit960`, `vit480` | Dense ViT on an area-downsampled input (long side in pixels) |
| `pvt`, `resnet18_lr1e-3` | PVTv2-B0 and ResNet-18 (`timm`, no pretraining) |

The patch grids are rectangular. Anchor positions are measured in units of the long side, so
Voronoi regions stay isotropic.

## Cost sweep (phase C)

Untrained models are timed alone on the GPU at 512², 1024², 1536² and 2048² (Megapixel MNIST
canvases) and at full HD, at batch 1 and 8. The sweep records:

* GFLOPs;
* training step time;
* inference throughput;
* peak training and inference memory;
* out-of-memory, if it happens.

The dense ViT is measured twice: with the fused FlashAttention kernel (`RART_SDPA=flash`) and
with plain attention (`RART_SDPA=math`). This checks whether experiment 02's memory advantage
(measured without FlashAttention) holds against a fused kernel.

## Running

On a rented GPU:

```bash
bash setup_box.sh                                   # deps, MNIST, DFG (7.5 GB), smoke tests
RART_PHASES=C RART_JOBS_PER_GPU=1 python run_suite.py
RART_PHASES=M,D RART_JOBS_PER_GPU=2 python run_suite.py
```

To fit one day on 2× A100, DFG can run at 1280×720 without training the full-resolution dense ViT
(the cost sweep still measures it):

```bash
RART_DFG_SCALE=0.6667 RART_SKIP_TAGS=D_vit_native_seed42,D_vit_native_seed43,D_vit_native_seed44   RART_PHASES=M,D RART_JOBS_PER_GPU=2 python run_suite.py
```

`build_data.py dfg --limit-mb 200` fetches only the first part of the image archive, for a
local smoke test. On Windows, set `RART_WORKERS=0` for DFG, because data-loader worker processes
re-import the script.

| Variable | Meaning | Default |
|---|---|---|
| `RART_DATASET` | `mpmnist` or `dfg` | mpmnist |
| `RART_KIND` | `focus`, `fixed`, `densewin`, `vit`, `pvt`, `resnet` | focus |
| `RART_ANCHORS` | Anchor count (square number) | 64 |
| `RART_CANVAS` | Megapixel MNIST canvas side | 1024 |
| `RART_DFG_SCALE` | DFG resize factor (1.0 = 1920×1080) | 1.0 |
| `RART_VIT_INPUT` | Dense ViT input long side; 0 = native | 0 |
| `RART_SDPA` | Attention kernel: `auto`, `flash`, `efficient`, `math` | auto |
| `RART_EPOCHS`, `RART_LR`, `RART_BATCH`, `RART_SEED` | Training settings | 30, 2e-4, 32, 42 |
| `RART_MEASURE_ONLY`, `RART_MEASURE_BATCH` | Time an untrained model instead of training | 0, batch |

## Results

Not run yet.
