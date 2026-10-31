"""Experiment 06 — one training run on Cluttered CIFAR, configured by environment variables.

The anchor models are the ones from experiment 04 (same code, same defaults), plus:
  * baselines: a dense ViT that can downsample its input first (`vit`), and PVTv2-B0 (`pvt`)
  * focus ablations: coverage-grid size, suppression strength, home prior, soft assignment
  * compute skipping with a bounded importance gate (`RART_GATE=bounded`)
  * efficiency measurements: GFLOPs including attention, inference throughput, peak memory

Each run writes one JSON file, `<RART_OUT_DIR>/<RART_TAG>.json`.
"""
import gc
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def env(name, default):
    return os.environ.get(name, default)


KIND = env("RART_KIND", "focus")
NUM_ANCHORS = int(env("RART_ANCHORS", "16"))
SEED = int(env("RART_SEED", "42"))
SMOKE = env("RART_SMOKE", "0") == "1"
NUM_EPOCHS = 1 if SMOKE else int(env("RART_EPOCHS", "10"))
DATASET = env("RART_DATASET", "cifar10")
TAG = env("RART_TAG", f"{KIND}_A{NUM_ANCHORS}_seed{SEED}")
SAVE_CHECKPOINT = env("RART_SAVE_CHECKPOINT", "0") == "1"
# Efficiency pass: no training; time training steps and inference of an untrained model.
# Run one job per GPU at a time so the timings are not shared with other jobs.
MEASURE_ONLY = env("RART_MEASURE_ONLY", "0") == "1"

# Baselines: `vit` resizes the 256² canvas to VIT_INPUT before patching.
VIT_INPUT = int(env("RART_VIT_INPUT", "256"))
VIT_PATCH = int(env("RART_VIT_PATCH", "16"))

# Compute skipping (experiment 05). RART_GATE: `ratio` = importance / importance.detach()
# as in experiment 05; `bounded` = 1 + importance - importance.detach().
KEEP_RATIO = float(env("RART_KEEP_RATIO", "1.0"))
SKIP_FROM = int(env("RART_SKIP_FROM", "2"))
SKIP_SCORE = env("RART_SKIP_SCORE", "importance")
GATE = env("RART_GATE", "ratio")

# Focus ablations. Defaults reproduce experiment 04.
COVERAGE_SIDE = env("RART_COVERAGE_SIDE", "auto")   # auto = max(1, isqrt(A // 4)); 0 = no coverage grid
SUPPRESSION = float(env("RART_SUPPRESSION", "10"))
HOME_WIDTH = float(env("RART_HOME_WIDTH", "0.35"))  # inf = no home prior
ASSIGN = env("RART_ASSIGN", "hard")                  # hard (straight-through) or soft

# Training recipe. Defaults reproduce experiment 04 (no clipping, no warmup, cosine per
# epoch). Phase B (30 epochs) uses clip 1.0 and a 2-epoch linear warmup: without them,
# 2 of its first 6 runs were knocked off course by gradient spikes.
GRAD_CLIP = float(env("RART_GRAD_CLIP", "0"))       # max gradient norm; 0 = off
WARMUP_EPOCHS = float(env("RART_WARMUP_EPOCHS", "0"))
PRECISION = env("RART_PRECISION", "fp16")             # autocast dtype: fp16 (with GradScaler) or bf16
AMP_DTYPE = torch.bfloat16 if PRECISION == "bf16" else torch.float16
LR = float(env("RART_LR", "3e-4"))
# Diagnostics: when a step's gradient norm exceeds SPIKE_FACTOR x the running median,
# print the parameters with the largest gradient norms (0 = off).
SPIKE_FACTOR = float(env("RART_SPIKE_FACTOR", "0"))

CANVAS, OBJECT_SIZE, NUM_DISTRACTORS, CROP, DISTRACTOR_SIZE = 256, 64, 8, 16, 32
# Tiny ImageNet images are 64 px natively: the object is pasted without upsampling and the
# distractors are native 32 px crops, so the canvas carries real detail at full resolution.
if DATASET == "tinyimagenet":
    CROP = 32
PATCH_SIZE, BATCH_SIZE = 16, 128
GRID = CANVAS // PATCH_SIZE
EMBED_DIM, DEPTH, HEADS, MLP_RATIO, WINDOW = 384, 6, 6, 3.0, 8
ANCHOR_DIM, ANCHOR_HEADS = 128, 4
METRIC_IMAGES = 256 if SMOKE else 1024
TEST_LAYOUT_SEED = 1234
ROUTER_KINDS = ("free", "fixed", "moving", "focus")

OUT_DIR = env("RART_OUT_DIR", os.path.abspath("results"))
DATA_DIR = env("RART_DATA_DIR", os.path.abspath("data"))
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(OUT_DIR, TAG + ".json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

if KIND in ROUTER_KINDS and math.isqrt(NUM_ANCHORS) ** 2 != NUM_ANCHORS:
    raise ValueError(f"anchor counts must be square numbers, got {NUM_ANCHORS}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


print(f"[{TAG}] {KIND} | anchors {NUM_ANCHORS} | seed {SEED} | epochs {NUM_EPOCHS} | "
      f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
      f"{' | SMOKE' if SMOKE else ''}", flush=True)

# ---------------------------------------------------------------- Cluttered CIFAR
DATASETS = {
    "cifar10": (torchvision.datasets.CIFAR10, "cifar-10-batches-py", 10,
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": (torchvision.datasets.CIFAR100, "cifar-100-python", 100,
                 (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    # Built by build_extra_data.py from the Hugging Face mirror; no download fallback.
    "tinyimagenet": (None, "tiny-imagenet", 200,
                     (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}
dataset_cls, dataset_folder, NUM_CLASSES, mean, std = DATASETS[DATASET]


def find_data_root():
    if os.path.isdir("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            if dataset_folder in dirs:
                return root, False
    return DATA_DIR, not os.path.isdir(os.path.join(DATA_DIR, dataset_folder))


data_root, need_download = find_data_root()


def load_split(train):
    """Reads the python-pickle batch files the way torchvision does, but without its MD5
    check, so a copy rebuilt from a mirror in the same format also loads."""
    folder = os.path.join(data_root, dataset_folder)
    if need_download:
        dataset_cls(data_root, train=train, download=True)
    if DATASET == "cifar10":
        files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        label_key = "labels"
    elif DATASET == "tinyimagenet":
        files = ["train" if train else "test"]
        label_key = "labels"
    else:
        files = ["train" if train else "test"]
        label_key = "fine_labels"
    data, targets = [], []
    for name in files:
        with open(os.path.join(folder, name), "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        data.append(entry["data"])
        targets.extend(entry[label_key])
    side = math.isqrt(data[0].shape[1] // 3)
    images = torch.from_numpy(np.vstack(data).reshape(-1, 3, side, side)).contiguous()
    labels = torch.tensor(targets, dtype=torch.long)
    return images, labels


train_x, train_y = load_split(True)
test_x, test_y = load_split(False)
if SMOKE:
    train_x, train_y = train_x[:512], train_y[:512]
    test_x, test_y = test_x[:256], test_y[:256]
train_x, train_y, test_x, test_y = (t.to(DEVICE) for t in (train_x, train_y, test_x, test_y))

MEAN = torch.tensor(mean, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(std, device=DEVICE).view(1, 3, 1, 1)
CHANNELS = torch.arange(3, device=DEVICE)


def paste(canvas, tiles, batch_index, top, left):
    height, width = tiles.shape[-2:]
    rows = top[:, None] + torch.arange(height, device=DEVICE)
    cols = left[:, None] + torch.arange(width, device=DEVICE)
    canvas[batch_index[:, None, None, None], CHANNELS[None, :, None, None],
           rows[:, None, :, None], cols[:, None, None, :]] = tiles


def compose(objects, pool, generator, flip):
    """objects: uint8 (B, 3, S, S), S = 32 (CIFAR) or 64 (Tiny ImageNet). Returns normalised
    canvases and object boxes (B, 4) as (top, left, bottom, right) in pixels."""
    batch = objects.shape[0]
    canvas = torch.zeros(batch, 3, CANVAS, CANVAS, device=DEVICE)

    count = batch * NUM_DISTRACTORS
    source = pool[torch.randint(len(pool), (count,), generator=generator).to(DEVICE)].float().div_(255)
    offsets = torch.randint(pool.shape[-1] - CROP + 1, (count, 2), generator=generator).to(DEVICE)
    rows = offsets[:, :1] + torch.arange(CROP, device=DEVICE)
    cols = offsets[:, 1:] + torch.arange(CROP, device=DEVICE)
    crops = source[torch.arange(count, device=DEVICE)[:, None, None, None], CHANNELS[None, :, None, None],
                   rows[:, None, :, None], cols[:, None, None, :]]
    crops = F.interpolate(crops, size=(DISTRACTOR_SIZE, DISTRACTOR_SIZE), mode="bilinear", align_corners=False)
    places = torch.randint(CANVAS - DISTRACTOR_SIZE + 1, (count, 2), generator=generator).to(DEVICE)
    paste(canvas, crops, torch.arange(batch, device=DEVICE).repeat_interleave(NUM_DISTRACTORS),
          places[:, 0], places[:, 1])

    objects = objects.float().div_(255)
    if flip:
        mirror = (torch.rand(batch, generator=generator) < 0.5).to(DEVICE)
        objects = torch.where(mirror.view(-1, 1, 1, 1), objects.flip(-1), objects)
    objects = F.interpolate(objects, size=(OBJECT_SIZE, OBJECT_SIZE), mode="bilinear", align_corners=False)
    corners = torch.randint(CANVAS - OBJECT_SIZE + 1, (batch, 2), generator=generator).to(DEVICE)
    paste(canvas, objects, torch.arange(batch, device=DEVICE), corners[:, 0], corners[:, 1])

    boxes = torch.cat([corners, corners + OBJECT_SIZE], dim=1)
    return (canvas - MEAN) / STD, boxes


def iterate_batches(images, labels, train, generator=None):
    n = images.shape[0]
    if train:
        order = torch.randperm(n, generator=generator).to(DEVICE)
    else:
        order = torch.arange(n, device=DEVICE)
        generator = torch.Generator().manual_seed(TEST_LAYOUT_SEED)
    for start in range(0, n, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        x, boxes = compose(images[idx], images, generator, flip=train)
        yield x, labels[idx], boxes


def object_patch_mask(boxes):
    """(B, N) bool: patches whose centre lies inside the object box."""
    centres = (torch.arange(GRID, device=DEVICE) + 0.5) * PATCH_SIZE
    inside_rows = (centres[None] >= boxes[:, :1]) & (centres[None] < boxes[:, 2:3])
    inside_cols = (centres[None] >= boxes[:, 1:2]) & (centres[None] < boxes[:, 3:4])
    return (inside_rows[:, :, None] & inside_cols[:, None, :]).flatten(1)


sample_x, sample_y, sample_boxes = next(iterate_batches(test_x, test_y, train=False))

# ---------------------------------------------------------------- Shared backbone
def split_heads(t, heads):
    batch, length, dim = t.shape
    return t.view(batch, length, heads, dim // heads).transpose(1, 2)


def merge_heads(t):
    batch, heads, length, dim = t.shape
    return t.transpose(1, 2).reshape(batch, length, heads * dim)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = (split_heads(t, self.heads) for t in self.qkv(x).chunk(3, dim=-1))
        return self.proj(merge_heads(F.scaled_dot_product_attention(q, k, v)))


class LocalWindowAttention(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        self.window = window
        self.attention = SelfAttention(dim, heads)

    def forward(self, x, grid):
        batch, num_patches, dim = x.shape
        w, n = self.window, grid // self.window
        windows = x.view(batch, n, w, n, w, dim).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, dim)
        windows = self.attention(windows)
        return windows.view(batch, n, n, w, w, dim).permute(0, 1, 3, 2, 4, 5).reshape(batch, num_patches, dim)


def grid_positions(side):
    """Cell centres of a side × side grid in normalised [0, 1] coordinates, row-major."""
    centres = (torch.arange(side, dtype=torch.float32) + 0.5) / side
    rows, cols = torch.meshgrid(centres, centres, indexing="ij")
    return torch.stack([rows, cols], dim=-1).reshape(-1, 2)


# ---------------------------------------------------------------- Anchor pathways
class AnchorRouter(nn.Module):
    def __init__(self, num_anchors):
        super().__init__()
        da = ANCHOR_DIM
        self.num_anchors = num_anchors
        self.heads = ANCHOR_HEADS
        self.scale = (da // ANCHOR_HEADS) ** -0.5
        self.gather_in = nn.Linear(EMBED_DIM, da)
        self.gather_q = nn.Linear(da, da)
        self.gather_kv = nn.Linear(da, da * 2)
        self.gather_out = nn.Linear(da, da)
        self.anchor_norm = nn.LayerNorm(da)
        self.anchor_attention = SelfAttention(da, ANCHOR_HEADS)
        self.scatter_norm = nn.LayerNorm(da)
        self.scatter_in = nn.Linear(EMBED_DIM, da)
        self.scatter_q = nn.Linear(da, da)
        self.scatter_kv = nn.Linear(da, da * 2)
        self.scatter_out = nn.Linear(da, EMBED_DIM)
        self.record = False
        self.last = None

    def keep(self, pooling, pooled, positions):
        if self.record:
            self.last = (pooling.detach().float(), pooled.detach().float(),
                         None if positions is None else positions.detach().float())

    def gather_logits(self, x, anchors):
        k, v = (split_heads(t, self.heads) for t in self.gather_kv(self.gather_in(x)).chunk(2, dim=-1))
        q = split_heads(self.gather_q(anchors), self.heads)
        return (q @ k.transpose(-2, -1)).float() * self.scale, v

    def update_anchors(self, anchors, weights, v):
        pooled = merge_heads(weights.to(v.dtype) @ v)
        anchors = anchors + self.gather_out(pooled)
        anchors = anchors + self.anchor_attention(self.anchor_norm(anchors))
        return anchors, pooled

    def scatter_inputs(self, x, anchors):
        k, v = (split_heads(t, self.heads) for t in self.scatter_kv(self.scatter_norm(anchors)).chunk(2, dim=-1))
        q = split_heads(self.scatter_q(self.scatter_in(x)), self.heads)
        return q, k, v


class FreeRouter(AnchorRouter):
    """RART-16/32 routing: learned anchor queries, softmax over all patches."""

    def forward(self, x, state):
        anchors, positions = state
        logits, v = self.gather_logits(x, anchors)
        weights = logits.softmax(dim=-1)
        anchors, pooled = self.update_anchors(anchors, weights, v)
        q, k, v = self.scatter_inputs(x, anchors)
        update = self.scatter_out(merge_heads(F.scaled_dot_product_attention(q, k, v)))
        self.keep(weights.mean(1), pooled, None)
        return update, (anchors, positions)


class VoronoiRouter(AnchorRouter):
    """Anchors with 2-D positions and an exclusive Voronoi partition of the patches.
    move=False keeps the positions on a fixed grid (Partition RART); move=True pulls each
    anchor to the importance-weighted centroid of its region every block."""

    def __init__(self, num_anchors, move):
        super().__init__(num_anchors)
        self.move = move
        self.register_buffer("coords", grid_positions(GRID), persistent=False)
        self.tau = (0.5 / math.isqrt(num_anchors)) ** 2
        self.position_embed = nn.Sequential(
            nn.Linear(2, ANCHOR_DIM), nn.GELU(), nn.Linear(ANCHOR_DIM, ANCHOR_DIM)
        )
        if move:
            self.importance = nn.Linear(EMBED_DIM, 1)

    def assignment(self, positions):
        """(B, A, N) patch → anchor assignment and the (B, A, N) bool hard partition.
        With ASSIGN=hard the assignment is one-hot in the forward pass with the gradient of
        a soft distance-softmax (straight-through); with ASSIGN=soft it is the softmax."""
        distance = (positions[:, :, None, :] - self.coords[None, None]).square().sum(dim=-1)
        soft = (-distance / self.tau).softmax(dim=1)
        hard = F.one_hot(soft.argmax(dim=1), self.num_anchors).transpose(1, 2).to(soft.dtype)
        assign = soft if ASSIGN == "soft" else hard + soft - soft.detach()
        return assign, hard > 0.5

    def place(self, x, positions):
        """Returns the anchor positions for this block and the patch importance
        (None when the router does not use importance)."""
        if not self.move:
            return positions, None
        importance = F.softplus(self.importance(x).squeeze(-1).float()) + 1e-3
        mass = self.assignment(positions)[0] * importance[:, None, :]
        total = mass.sum(dim=-1, keepdim=True)
        centroid = (mass @ self.coords) / total.clamp_min(1e-6)
        return torch.where(total > 1e-4, centroid, positions), importance

    def forward(self, x, state):
        anchors, positions = state
        positions, importance = self.place(x, positions)
        self.current_importance = importance

        assign, hard = self.assignment(positions)
        located = anchors + self.position_embed(positions).to(anchors.dtype)
        logits, v = self.gather_logits(x, located)
        if importance is not None:
            logits = logits + importance.log()[:, None, None, :]
        peak = logits.masked_fill(~hard[:, None], float("-inf")).amax(dim=-1, keepdim=True)
        peak = torch.nan_to_num(peak.detach(), neginf=0.0)
        scores = (logits - peak).clamp_max(0).exp() * assign[:, None]
        # A non-empty region always sums to >= 1 (its top patch scores exp(0) = 1), so the
        # clamp only affects empty regions, where a tiny denominator would blow up the
        # straight-through gradient (focus anchors can leave a region empty).
        weights = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0)
        anchors, pooled = self.update_anchors(anchors, weights, v)

        q, k, v = self.scatter_inputs(x, anchors)
        gate = torch.sigmoid((q @ k.transpose(-2, -1)).float() * self.scale)
        read = (gate * assign.transpose(1, 2)[:, None]).to(v.dtype)
        update = self.scatter_out(merge_heads(read @ v))

        self.keep(weights.mean(1), pooled, positions)
        return update, (anchors, positions)


class FocusRouter(VoronoiRouter):
    """Coverage + focus anchors. A coverage_side² grid of anchors stays fixed so every
    region keeps a global summary; the rest are placed each block at the peaks of the
    learned importance map, anywhere in the image. Peaks are found by a soft-argmax, and
    each chosen peak is suppressed with a Gaussian so the next focus anchor lands
    elsewhere. Each focus anchor also has a spread-out home position with a weak Gaussian
    prior. All anchors then share one exclusive Voronoi partition."""

    SUPPRESSION_WIDTH = 1.5 / GRID

    def __init__(self, num_anchors):
        super().__init__(num_anchors, move=True)
        coverage_side = max(1, math.isqrt(num_anchors // 4)) if COVERAGE_SIDE == "auto" else int(COVERAGE_SIDE)
        self.num_focus = num_anchors - coverage_side ** 2
        if self.num_focus < 1:
            raise ValueError(f"coverage grid {coverage_side}² leaves no focus anchors out of {num_anchors}")
        self.register_buffer("coverage", grid_positions(coverage_side), persistent=False)
        spread = torch.linspace(0, num_anchors - 1, self.num_focus).round().long()
        homes = grid_positions(math.isqrt(num_anchors))[spread]
        home_distance = (self.coords[None] - homes[:, None]).square().sum(dim=-1)
        home_prior = torch.zeros_like(home_distance) if math.isinf(HOME_WIDTH) else \
            -home_distance / (2 * HOME_WIDTH ** 2)
        self.register_buffer("home_prior", home_prior, persistent=False)

    def place(self, x, positions):
        scores = self.importance(x).squeeze(-1).float()
        importance = F.softplus(scores) + 1e-3
        peaks = []
        for k in range(self.num_focus):
            peak = (scores + self.home_prior[k]).softmax(dim=-1) @ self.coords
            peaks.append(peak)
            if SUPPRESSION > 0:
                distance = (self.coords[None] - peak[:, None]).square().sum(dim=-1)
                scores = scores - SUPPRESSION * torch.exp(-distance / (2 * self.SUPPRESSION_WIDTH ** 2))
        coverage = self.coverage.expand(x.shape[0], -1, -1)
        return torch.cat([coverage, torch.stack(peaks, dim=1)], dim=1), importance


ROUTERS = {
    "free": lambda a: FreeRouter(a),
    "fixed": lambda a: VoronoiRouter(a, move=False),
    "moving": lambda a: VoronoiRouter(a, move=True),
    "focus": lambda a: FocusRouter(a),
}


class Block(nn.Module):
    """Local window attention → anchor pathway → FFN. From block SKIP_FROM on, when
    KEEP_RATIO < 1, only the top KEEP_RATIO of patches (by SKIP_SCORE) go through the FFN;
    the rest skip it but have already received their anchor's global update."""

    def __init__(self, router, index):
        super().__init__()
        self.index = index
        self.local_norm = nn.LayerNorm(EMBED_DIM)
        self.local = LocalWindowAttention(EMBED_DIM, HEADS, WINDOW)
        self.global_norm = nn.LayerNorm(EMBED_DIM)
        self.router = router
        self.ffn_norm = nn.LayerNorm(EMBED_DIM)
        hidden = int(EMBED_DIM * MLP_RATIO)
        self.ffn = nn.Sequential(nn.Linear(EMBED_DIM, hidden), nn.GELU(), nn.Linear(hidden, EMBED_DIM))

    def forward(self, x, state):
        x = x + self.local(self.local_norm(x), GRID)
        update, state = self.router(self.global_norm(x), state)
        x = x + update
        if KEEP_RATIO < 1 and self.index >= SKIP_FROM:
            x = x + self.sparse_ffn(x)
        else:
            x = x + self.ffn(self.ffn_norm(x))
        return x, state

    def sparse_ffn(self, x):
        batch, num_patches, dim = x.shape
        keep = max(1, math.ceil(num_patches * KEEP_RATIO))
        importance = getattr(self.router, "current_importance", None)
        if SKIP_SCORE == "importance":
            if importance is None:
                raise ValueError("SKIP_SCORE=importance needs a router with an importance map")
            score = importance
        elif SKIP_SCORE == "norm":
            score = x.detach().float().norm(dim=-1)
        elif SKIP_SCORE == "random":
            score = torch.rand(batch, num_patches, device=x.device)
        else:
            raise ValueError(f"unknown SKIP_SCORE {SKIP_SCORE}")
        index = score.topk(keep, dim=1).indices
        kept = x.gather(1, index[:, :, None].expand(-1, -1, dim))
        out = self.ffn(self.ffn_norm(kept))
        if SKIP_SCORE == "importance":
            chosen = importance.gather(1, index)
            # Both gates are exactly 1 in the forward pass. `ratio` has gradient 1/importance
            # (large for small importance); `bounded` has gradient 1.
            gate = 1 + chosen - chosen.detach() if GATE == "bounded" else chosen / chosen.detach()
            out = out * gate[:, :, None].to(out.dtype)
        return torch.zeros(batch, num_patches, dim, dtype=out.dtype, device=x.device).scatter(
            1, index[:, :, None].expand(-1, -1, dim), out
        )


class AnchorViT(nn.Module):
    def __init__(self, kind, num_anchors):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, EMBED_DIM, PATCH_SIZE, PATCH_SIZE)
        self.pos_embed = nn.Parameter(torch.randn(1, GRID * GRID, EMBED_DIM) * 0.02)
        self.anchor_init = nn.Parameter(torch.randn(1, num_anchors, ANCHOR_DIM) * 0.02)
        self.register_buffer("start_positions", grid_positions(math.isqrt(num_anchors)), persistent=False)
        self.blocks = nn.ModuleList(Block(ROUTERS[kind](num_anchors), i) for i in range(DEPTH))
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def routers(self):
        return [block.router for block in self.blocks]

    def forward(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed
        batch = x.shape[0]
        state = (self.anchor_init.expand(batch, -1, -1),
                 self.start_positions.expand(batch, -1, -1).clone())
        for block in self.blocks:
            x, state = block(x, state)
        return self.head(self.norm(x).mean(dim=1))


# ---------------------------------------------------------------- Baselines
class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(EMBED_DIM)
        self.attention = SelfAttention(EMBED_DIM, HEADS)
        self.ffn_norm = nn.LayerNorm(EMBED_DIM)
        hidden = int(EMBED_DIM * MLP_RATIO)
        self.ffn = nn.Sequential(nn.Linear(EMBED_DIM, hidden), nn.GELU(), nn.Linear(hidden, EMBED_DIM))

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class DenseViT(nn.Module):
    """Same width, depth and mean-pooled head as AnchorViT, with global attention in every
    block. The canvas is area-downsampled to `input_size` before patching."""

    def __init__(self, input_size, patch):
        super().__init__()
        if input_size % patch:
            raise ValueError(f"input {input_size} is not divisible by patch {patch}")
        self.input_size = input_size
        self.patch_embed = nn.Conv2d(3, EMBED_DIM, patch, patch)
        self.pos_embed = nn.Parameter(torch.randn(1, (input_size // patch) ** 2, EMBED_DIM) * 0.02)
        self.blocks = nn.ModuleList(DenseBlock() for _ in range(DEPTH))
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def routers(self):
        return []

    def forward(self, images):
        if self.input_size != images.shape[-1]:
            images = F.interpolate(images, size=(self.input_size, self.input_size), mode="area")
        x = self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x).mean(dim=1))


class TimmModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        import timm
        self.net = timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)

    def routers(self):
        return []

    def forward(self, images):
        return self.net(images)


def build_model():
    if KIND in ROUTER_KINDS:
        return AnchorViT(KIND, NUM_ANCHORS)
    if KIND == "vit":
        return DenseViT(VIT_INPUT, VIT_PATCH)
    if KIND == "pvt":
        return TimmModel("pvt_v2_b0")
    if KIND == "resnet":
        return TimmModel("resnet18")  # standard ImageNet stem; the 256² canvas is ImageNet-sized
    raise ValueError(f"unknown RART_KIND {KIND}")


# ---------------------------------------------------------------- Metrics
def off_diagonal_mean(similarity):
    size = similarity.shape[-1]
    mask = ~torch.eye(size, dtype=torch.bool, device=similarity.device)
    return similarity[:, mask].mean().item()


def router_stats(pooling, pooled, positions, on_object):
    pooling_unit = F.normalize(pooling, dim=-1)
    centred = pooled - pooled.mean(dim=1, keepdim=True)
    spread = centred.square().sum(dim=(1, 2)) / pooled.square().sum(dim=(1, 2)).clamp_min(1e-12)
    stats = {
        "map_similarity": off_diagonal_mean(pooling_unit @ pooling_unit.transpose(-2, -1)),
        "anchor_spread": spread.mean().item(),
        "object_mass": (pooling * on_object[:, None, :]).sum(dim=(1, 2)).mean().item(),
    }
    if positions is not None:
        coords = grid_positions(GRID).to(positions.device)
        owner = (positions[:, :, None, :] - coords[None, None]).square().sum(-1).argmin(dim=1)
        owns_object = F.one_hot(owner, positions.shape[1]) * on_object[:, :, None]
        stats["object_anchor_share"] = owns_object.amax(dim=1).float().mean().item()
    return stats


@torch.no_grad()
def anchor_metrics(model):
    routers = model.routers()
    if not routers:
        return None
    model.eval()
    for router in routers:
        router.record = True
    totals = [defaultdict(float) for _ in routers]
    seen = 0
    for x, _, boxes in iterate_batches(test_x[:METRIC_IMAGES], test_y[:METRIC_IMAGES], train=False):
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
            model(x)
        on_object = object_patch_mask(boxes)
        for total, router in zip(totals, routers):
            for name, value in router_stats(*router.last, on_object).items():
                total[name] += value * x.shape[0]
        seen += x.shape[0]
    for router in routers:
        router.record = False
        router.last = None
    return [{name: value / seen for name, value in total.items()} for total in totals]


@torch.no_grad()
def count_gflops_linear(model, images):
    """Experiment 05's measure: 2 × multiply-accumulates of every Linear and Conv2d that
    runs (attention score/value matmuls excluded)."""
    macs = 0

    def linear_hook(module, inputs, output):
        nonlocal macs
        macs += inputs[0].numel() // module.in_features * module.in_features * module.out_features

    def conv_hook(module, inputs, output):
        nonlocal macs
        kernel = module.kernel_size[0] * module.kernel_size[1]
        macs += output.numel() * module.in_channels * kernel // module.groups

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
    model.eval()
    model(images)
    for handle in handles:
        handle.remove()
    return 2 * macs / images.shape[0] / 1e9


def count_gflops_total(model, images):
    """All matmuls, convolutions and attention kernels, from torch's FlopCounterMode
    (counts attention correctly on CUDA). Elementwise ops are not counted. Runs with
    autograd on because the counter's module tracker fails under no_grad for modules
    that take tuple inputs; only the forward pass is counted."""
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    with torch.enable_grad(), FlopCounterMode(display=False) as counter:
        model(images)
    return counter.get_total_flops() / images.shape[0] / 1e9


@torch.no_grad()
def throughput(model, images, iterations=20):
    """Inference images per second at batch len(images), fp16 autocast."""
    if DEVICE.type != "cuda":
        return None
    model.eval()
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        for _ in range(5):
            model(images)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(images)
        torch.cuda.synchronize()
    return images.shape[0] * iterations / (time.perf_counter() - start)


# ---------------------------------------------------------------- Training
@torch.no_grad()
def evaluate(model, criterion):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y, _ in iterate_batches(test_x, test_y, train=False):
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * y.shape[0]
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += y.shape[0]
    return total_loss / seen, 100.0 * correct / seen


def run():
    set_seed(SEED)
    model = build_model().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"[{TAG}] {params:,} parameters", flush=True)

    metrics_init = anchor_metrics(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    steps_per_epoch = math.ceil(train_x.shape[0] / BATCH_SIZE)
    if WARMUP_EPOCHS > 0:
        # Linear warmup, then cosine decay to 0, stepped every batch.
        warmup, total = int(WARMUP_EPOCHS * steps_per_epoch), NUM_EPOCHS * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warmup if step < warmup
                                                      else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=DEVICE.type == "cuda" and PRECISION == "fp16")
    generator = torch.Generator().manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        start = time.perf_counter()
        total_loss, correct, seen = 0.0, 0, 0
        grad_norms, skipped = [], 0
        for x, y, _ in iterate_batches(train_x, train_y, train=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP if GRAD_CLIP > 0 else float("inf")
            ).item()
            scaler.step(optimizer)
            scaler.update()
            if WARMUP_EPOCHS > 0:
                scheduler.step()
            if SPIKE_FACTOR > 0 and len(grad_norms) > 50 and grad_norm > SPIKE_FACTOR * float(np.median(grad_norms[-200:])):
                top = sorted(((p.grad.norm().item(), n) for n, p in model.named_parameters() if p.grad is not None), reverse=True)[:6]
                print(f"[{TAG}] SPIKE epoch {epoch} step {len(grad_norms)}: norm {grad_norm:.1f} | "
                      + " ; ".join(f"{n} {g:.1f}" for g, n in top), flush=True)
            if math.isfinite(grad_norm):
                grad_norms.append(grad_norm)
            else:
                skipped += 1
            total_loss += loss.item() * y.shape[0]
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += y.shape[0]
        if WARMUP_EPOCHS <= 0:
            scheduler.step()
        test_loss, test_acc = evaluate(model, criterion)
        minutes = (time.perf_counter() - start) / 60
        norms = np.array(grad_norms) if grad_norms else np.array([float("nan")])
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "train_acc": 100.0 * correct / seen,
                        "test_loss": test_loss, "test_acc": test_acc, "minutes": minutes,
                        "grad_norm_median": float(np.median(norms)), "grad_norm_max": float(norms.max()),
                        "steps_skipped": skipped})
        print(f"[{TAG}] epoch {epoch:02d}/{NUM_EPOCHS} | train acc {100.0 * correct / seen:5.2f}% | "
              f"test acc {test_acc:5.2f}% | grad norm median {np.median(norms):.2f} max {norms.max():.2f} | "
              f"skipped steps {skipped} | {minutes:.2f} min", flush=True)

    peak_train_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else None
    config = {"kind": KIND, "num_anchors": NUM_ANCHORS if KIND in ROUTER_KINDS else 0, "seed": SEED,
              "epochs": NUM_EPOCHS, "dataset": DATASET, "vit_input": VIT_INPUT, "vit_patch": VIT_PATCH,
              "keep_ratio": KEEP_RATIO, "skip_from": SKIP_FROM, "skip_score": SKIP_SCORE, "gate": GATE,
              "coverage_side": COVERAGE_SIDE, "suppression": SUPPRESSION, "home_width": HOME_WIDTH,
              "assign": ASSIGN, "grad_clip": GRAD_CLIP, "warmup_epochs": WARMUP_EPOCHS, "precision": PRECISION, "lr": LR}
    result = {"tag": TAG, **config, "params": params, "history": history,
              "final_test_acc": history[-1]["test_acc"],
              "metrics_init": metrics_init, "metrics_trained": anchor_metrics(model),
              "gflops_linear": count_gflops_linear(model, sample_x[:16]),
              "gflops_total": count_gflops_total(model, sample_x[:16]),
              "throughput_img_s": throughput(model, sample_x),
              "peak_train_memory_mb": peak_train_mb,
              "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
              "total_minutes": sum(h["minutes"] for h in history)}
    if SAVE_CHECKPOINT:
        torch.save(model.state_dict(), os.path.join(OUT_DIR, TAG + ".pth"))
    return result


def measure(steps=20, warmup=5):
    """Compute cost does not depend on the weights, so an untrained model gives the same
    GFLOPs, step time, throughput and memory as a trained one."""
    set_seed(SEED)
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=DEVICE.type == "cuda" and PRECISION == "fp16")
    generator = torch.Generator().manual_seed(SEED)
    batches = iterate_batches(train_x, train_y, train=True, generator=generator)
    x, y, _ = next(batches)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(warmup + steps):
        if step == warmup:
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    step_ms = (time.perf_counter() - start) / steps * 1000
    peak_train_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else None
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scaler
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    images_per_s = throughput(model, sample_x, iterations=50)
    peak_inference_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else None
    return {"tag": TAG, "measure_only": True, "kind": KIND,
            "num_anchors": NUM_ANCHORS if KIND in ROUTER_KINDS else 0, "vit_input": VIT_INPUT,
            "vit_patch": VIT_PATCH, "keep_ratio": KEEP_RATIO, "skip_score": SKIP_SCORE,
            "params": sum(p.numel() for p in model.parameters()), "batch_size": x.shape[0],
            "train_step_ms": step_ms, "peak_train_memory_mb": peak_train_mb,
            "throughput_img_s": images_per_s, "peak_inference_memory_mb": peak_inference_mb,
            "gflops_linear": count_gflops_linear(model, sample_x[:16]),
            "gflops_total": count_gflops_total(model, sample_x[:16]),
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
            "final_test_acc": None}


if __name__ == "__main__":
    if os.path.exists(RESULTS_PATH) and not SMOKE:
        print(f"[{TAG}] already done: {RESULTS_PATH}")
    elif MEASURE_ONLY:
        result = measure()
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[{TAG}] measured: {result['gflops_total']:.2f} GFLOPs | {result['train_step_ms']:.1f} ms/step | "
              f"{result['throughput_img_s'] or 0:.0f} img/s | {result['peak_train_memory_mb'] or 0:.0f} MB", flush=True)
    else:
        result = run()
        with open(RESULTS_PATH + ".tmp", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        os.replace(RESULTS_PATH + ".tmp", RESULTS_PATH)
        print(f"[{TAG}] done: {result['final_test_acc']:.2f}% | {result['gflops_total']:.2f} GFLOPs | "
              f"{result['throughput_img_s'] or 0:.0f} img/s | saved {RESULTS_PATH}", flush=True)
        gc.collect()
