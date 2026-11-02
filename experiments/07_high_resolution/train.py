"""Experiment 07 — one training run on a high-resolution task, configured by environment variables.

The anchor models are experiment 06's (same blocks, same focus placement), generalised to
rectangular patch grids, on two tasks where downsampling destroys the signal:
  * `mpmnist`: Megapixel MNIST (Katharopoulos & Fleuret, 2019) at RART_CANVAS² (default 1024²):
    five native 28 px MNIST digits, three of one class and two of another, among 50 stroke-like
    noise patches; the label is the majority digit. 5000 train / 1000 val / 1000 test images.
  * `dfg`: DFG traffic signs (Tabernik & Skočaj, 2019), full-HD street scenes, recognising which
    of the 20 most frequent sign types are present (multi-label, mean average precision).
Both tasks have a validation split, so settings can be chosen without looking at the test set.

Each run writes one JSON file, `<RART_OUT_DIR>/<RART_TAG>.json`. With RART_MEASURE_ONLY=1 it
times an untrained model instead (GFLOPs, step time, throughput, peak memory; records OOM).
"""
import gc
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def env(name, default):
    return os.environ.get(name, default)


KIND = env("RART_KIND", "focus")
NUM_ANCHORS = int(env("RART_ANCHORS", "64"))
SEED = int(env("RART_SEED", "42"))
SMOKE = env("RART_SMOKE", "0") == "1"
NUM_EPOCHS = 1 if SMOKE else int(env("RART_EPOCHS", "30"))
DATASET = env("RART_DATASET", "mpmnist")
TAG = env("RART_TAG", f"{DATASET}_{KIND}_A{NUM_ANCHORS}_seed{SEED}")
SAVE_CHECKPOINT = env("RART_SAVE_CHECKPOINT", "0") == "1"
MEASURE_ONLY = env("RART_MEASURE_ONLY", "0") == "1"
# Dense ViT input: the long side in pixels (the short side keeps the aspect ratio); 0 = native.
VIT_INPUT = int(env("RART_VIT_INPUT", "0"))
VIT_PATCH = int(env("RART_VIT_PATCH", "16"))
COVERAGE_SIDE = env("RART_COVERAGE_SIDE", "auto")   # auto = max(1, isqrt(A // 4))
SUPPRESSION = float(env("RART_SUPPRESSION", "10"))
SUPPRESSION_PATCHES = float(env("RART_SUPPRESSION_WIDTH", "1.5"))  # Gaussian width, in patches
HOME_WIDTH = float(env("RART_HOME_WIDTH", "0.35"))
GRAD_CLIP = float(env("RART_GRAD_CLIP", "1.0"))
WARMUP_EPOCHS = float(env("RART_WARMUP_EPOCHS", "2"))
PRECISION = env("RART_PRECISION", "bf16")
AMP_DTYPE = torch.bfloat16 if PRECISION == "bf16" else torch.float16
LR = float(env("RART_LR", "2e-4"))
BATCH_SIZE = int(env("RART_BATCH", "32"))
# Attention kernel for the dense pathways: auto, flash, efficient or math (no fused kernel).
SDPA = env("RART_SDPA", "auto")
WORKERS = int(env("RART_WORKERS", "8"))

PATCH_SIZE, WINDOW = 16, 8
EMBED_DIM, DEPTH, HEADS, MLP_RATIO = 384, 6, 6, 3.0
ANCHOR_DIM, ANCHOR_HEADS = 128, 4
METRIC_IMAGES = 64 if SMOKE else 256
ROUTER_KINDS = ("fixed", "focus", "densewin")

OUT_DIR = env("RART_OUT_DIR", os.path.abspath("results"))
DATA_DIR = env("RART_DATA_DIR", os.path.abspath("data"))
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(OUT_DIR, TAG + ".json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

if KIND in ("fixed", "focus") and math.isqrt(NUM_ANCHORS) ** 2 != NUM_ANCHORS:
    raise ValueError(f"anchor counts must be square numbers, got {NUM_ANCHORS}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def padded(size):
    """Rounds a side up so the patch grid divides into local windows."""
    unit = PATCH_SIZE * WINDOW
    return math.ceil(size / unit) * unit


if DATASET == "mpmnist":
    IMAGE_H = IMAGE_W = int(env("RART_CANVAS", "1024"))
    NUM_CLASSES, MULTILABEL = 10, False
elif DATASET == "dfg":
    DFG_SCALE = float(env("RART_DFG_SCALE", "1.0"))
    IMAGE_H, IMAGE_W = round(1080 * DFG_SCALE), round(1920 * DFG_SCALE)
    NUM_CLASSES, MULTILABEL = 20, True
else:
    raise ValueError(f"unknown RART_DATASET {DATASET}")
CANVAS_H, CANVAS_W = padded(IMAGE_H), padded(IMAGE_W)
GRID_H, GRID_W = CANVAS_H // PATCH_SIZE, CANVAS_W // PATCH_SIZE
GRID = max(GRID_H, GRID_W)   # positions are in units of the long side, so distances are isotropic

print(f"[{TAG}] {KIND} | {DATASET} {CANVAS_H}x{CANVAS_W} ({GRID_H}x{GRID_W} patches) | anchors "
      f"{NUM_ANCHORS} | seed {SEED} | epochs {NUM_EPOCHS} | batch {BATCH_SIZE} | device "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}{' | SMOKE' if SMOKE else ''}",
      flush=True)


# ---------------------------------------------------------------- Megapixel MNIST
class MegapixelMNIST:
    """Fixed sets of layouts, composed on the GPU batch by batch. Every item (digit or noise
    patch) sits in its own 32 px cell with a random 0–4 px offset, so items never overlap."""

    CELL, ITEM, NOISE, BANK = 32, 28, 50, 4096
    SIZES = {"train": 5000, "val": 1000, "test": 1000}
    MEAN, STD = 0.01, 0.08

    def __init__(self):
        import torchvision
        root = DATA_DIR
        train = torchvision.datasets.MNIST(root, train=True, download=True)
        test = torchvision.datasets.MNIST(root, train=False, download=True)
        # Validation digits are held out of the MNIST training set.
        pools = {"train": (train.data[:50000], train.targets[:50000]),
                 "val": (train.data[50000:], train.targets[50000:]),
                 "test": (test.data, test.targets)}
        generator = torch.Generator().manual_seed(2019)
        self.bank = self.noise_bank(generator).to(DEVICE)
        self.splits = {}
        for offset, (name, size) in enumerate(self.SIZES.items()):
            digits, labels = pools[name]
            if SMOKE:
                size = 64
            self.splits[name] = self.layouts(digits, labels, size, torch.Generator().manual_seed(7 + offset))

    def noise_bank(self, generator):
        """Stroke-like noise: one or two random line segments, about 2 px thick."""
        coords = torch.stack(torch.meshgrid(torch.arange(self.ITEM), torch.arange(self.ITEM),
                                            indexing="ij"), dim=-1).float()
        bank = torch.zeros(self.BANK, self.ITEM, self.ITEM)
        for strokes in (1, 2):
            ends = torch.rand(self.BANK, 2, 2, generator=generator) * (self.ITEM - 1)
            a, b = ends[:, None, None, 0], ends[:, None, None, 1]
            ab = b - a
            t = (((coords[None] - a) * ab).sum(-1) / ab.square().sum(-1).clamp_min(1e-6)).clamp(0, 1)
            distance = (coords[None] - (a + t[..., None] * ab)).norm(dim=-1)
            stroke = (1.5 - distance).clamp(0, 1)
            use = (torch.rand(self.BANK, generator=generator) < (1.0 if strokes == 1 else 0.5)).float()
            bank = torch.maximum(bank, stroke * use[:, None, None])
        return bank

    def layouts(self, digits, labels, size, generator):
        by_class = [torch.nonzero(labels == c).squeeze(1) for c in range(10)]
        target = torch.randint(10, (size,), generator=generator)
        other = (target + torch.randint(1, 10, (size,), generator=generator)) % 10
        classes = torch.cat([target[:, None].expand(-1, 3), other[:, None].expand(-1, 2)], dim=1)
        picks = torch.empty(size, 5, dtype=torch.long)
        for c in range(10):
            mask = classes == c
            pool = by_class[c]
            picks[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), generator=generator)]
        cells_per_side = IMAGE_H // self.CELL
        cells = torch.rand(size, cells_per_side ** 2, generator=generator).argsort(dim=1)[:, :5 + self.NOISE]
        jitter = torch.randint(self.CELL - self.ITEM + 1, (size, 5 + self.NOISE, 2), generator=generator)
        corners = torch.stack([cells // cells_per_side, cells % cells_per_side], dim=-1) * self.CELL + jitter
        noise = torch.randint(self.BANK, (size, self.NOISE), generator=generator)
        return {"digits": digits[picks].to(DEVICE), "corners": corners.to(DEVICE), "noise": noise.to(DEVICE),
                "labels": target.to(DEVICE)}

    def batches(self, split, train, generator=None):
        data = self.splits[split]
        n = data["labels"].shape[0]
        order = torch.randperm(n, generator=generator).to(DEVICE) if train else torch.arange(n, device=DEVICE)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            yield self.compose({k: v[idx] for k, v in data.items()})

    def compose(self, data):
        batch = data["labels"].shape[0]
        items = torch.cat([data["digits"].float() / 255, self.bank[data["noise"]]], dim=1)  # (B, 55, 28, 28)
        canvas = torch.zeros(batch, CANVAS_H, CANVAS_W, device=DEVICE)
        rows = data["corners"][..., :1] + torch.arange(self.ITEM, device=DEVICE)    # (B, 55, 28)
        cols = data["corners"][..., 1:] + torch.arange(self.ITEM, device=DEVICE)
        image = torch.arange(batch, device=DEVICE)[:, None, None, None]
        canvas[image, rows[..., :, None], cols[..., None, :]] = items
        x = ((canvas - self.MEAN) / self.STD)[:, None].expand(-1, 3, -1, -1)
        boxes = torch.cat([data["corners"][:, :3], data["corners"][:, :3] + self.ITEM], dim=-1)  # target digits
        return x, data["labels"], torch.ones_like(data["labels"], dtype=torch.bool), boxes


# ---------------------------------------------------------------- DFG traffic signs
class DFGImages(torch.utils.data.Dataset):
    MAX_BOXES = 32

    def __init__(self, entries, folder):
        self.entries, self.folder = entries, folder

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        import torchvision.io as io
        entry = self.entries[i]
        image = io.decode_jpeg(io.read_file(os.path.join(self.folder, entry["file"])))
        if (IMAGE_H, IMAGE_W) != tuple(image.shape[-2:]):
            image = F.interpolate(image[None].float(), size=(IMAGE_H, IMAGE_W), mode="bilinear",
                                  antialias=True, align_corners=False)[0].round().clamp(0, 255).byte()
        target = torch.zeros(NUM_CLASSES)
        valid = torch.ones(NUM_CLASSES, dtype=torch.bool)
        target[entry["labels"]] = 1
        valid[[c for c in entry["ignore"] if c not in entry["labels"]]] = False  # only ignore-flagged signs
        boxes = torch.full((self.MAX_BOXES, 4), -1.0)
        if entry["boxes"]:
            kept = torch.tensor(entry["boxes"][:self.MAX_BOXES], dtype=torch.float32) * (IMAGE_W / 1920)
            boxes[:len(kept)] = kept
        return image, target, valid, boxes


class DFG:
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255

    def __init__(self):
        root = os.path.join(DATA_DIR, "dfg")
        with open(os.path.join(root, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        self.class_names = index["class_names"]
        folder = os.path.join(root, "JPEGImages")
        self.splits = {}
        for name in ("train", "val", "test"):
            entries = index[name][:64] if SMOKE else index[name]
            self.splits[name] = DFGImages(entries, folder)
        self.mean, self.std = self.MEAN.to(DEVICE), self.STD.to(DEVICE)

    def batches(self, split, train, generator=None):
        loader = torch.utils.data.DataLoader(
            self.splits[split], batch_size=BATCH_SIZE, shuffle=train, generator=generator,
            num_workers=WORKERS, pin_memory=DEVICE.type == "cuda", persistent_workers=False)
        for image, target, valid, boxes in loader:
            x = (image.to(DEVICE, non_blocking=True).float() - self.mean) / self.std
            x = F.pad(x, (0, CANVAS_W - IMAGE_W, 0, CANVAS_H - IMAGE_H))   # 0 = mean colour
            yield x, target.to(DEVICE), valid.to(DEVICE), boxes.to(DEVICE)


data = MegapixelMNIST() if DATASET == "mpmnist" else DFG()


def object_patch_mask(boxes):
    """(B, N) bool: patches whose centre lies inside any object box; boxes (B, K, 4) are
    (top, left, bottom, right) in pixels, padded with -1."""
    rows = (torch.arange(GRID_H, device=DEVICE) + 0.5) * PATCH_SIZE
    cols = (torch.arange(GRID_W, device=DEVICE) + 0.5) * PATCH_SIZE
    in_rows = (rows[None, None] >= boxes[..., :1]) & (rows[None, None] < boxes[..., 2:3])
    in_cols = (cols[None, None] >= boxes[..., 1:2]) & (cols[None, None] < boxes[..., 3:4])
    return (in_rows[..., :, None] & in_cols[..., None, :]).any(dim=1).flatten(1)


sample_x, sample_y, _, _ = next(data.batches("test", train=False))

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

    def forward(self, x, grid_h, grid_w):
        batch, num_patches, dim = x.shape
        w, nh, nw = self.window, grid_h // self.window, grid_w // self.window
        windows = x.view(batch, nh, w, nw, w, dim).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, dim)
        windows = self.attention(windows)
        return windows.view(batch, nh, nw, w, w, dim).permute(0, 1, 3, 2, 4, 5).reshape(batch, num_patches, dim)


def patch_coords():
    """Patch centres, row-major, in units of the long side: rows in [0, GRID_H / GRID]."""
    rows = (torch.arange(GRID_H, dtype=torch.float32) + 0.5) / GRID
    cols = (torch.arange(GRID_W, dtype=torch.float32) + 0.5) / GRID
    r, c = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([r, c], dim=-1).reshape(-1, 2)


def grid_positions(side):
    """Cell centres of a side × side grid stretched over the image, in patch_coords units."""
    centres = (torch.arange(side, dtype=torch.float32) + 0.5) / side
    r, c = torch.meshgrid(centres * GRID_H / GRID, centres * GRID_W / GRID, indexing="ij")
    return torch.stack([r, c], dim=-1).reshape(-1, 2)


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


class VoronoiRouter(AnchorRouter):
    """Anchors with 2-D positions and an exclusive Voronoi partition of the patches, with a
    straight-through hard assignment. Positions stay on a fixed grid (Partition RART)."""

    def __init__(self, num_anchors):
        super().__init__(num_anchors)
        self.register_buffer("coords", patch_coords(), persistent=False)
        self.tau = (0.5 / math.isqrt(num_anchors)) ** 2
        self.position_embed = nn.Sequential(
            nn.Linear(2, ANCHOR_DIM), nn.GELU(), nn.Linear(ANCHOR_DIM, ANCHOR_DIM)
        )

    def assignment(self, positions):
        distance = (positions[:, :, None, :] - self.coords[None, None]).square().sum(dim=-1)
        soft = (-distance / self.tau).softmax(dim=1)
        hard = F.one_hot(soft.argmax(dim=1), self.num_anchors).transpose(1, 2).to(soft.dtype)
        return hard + soft - soft.detach(), hard > 0.5

    def place(self, x, positions):
        return positions, None

    def forward(self, x, state):
        anchors, positions = state
        positions, importance = self.place(x, positions)
        assign, hard = self.assignment(positions)
        located = anchors + self.position_embed(positions).to(anchors.dtype)
        logits, v = self.gather_logits(x, located)
        if importance is not None:
            logits = logits + importance.log()[:, None, None, :]
        peak = logits.masked_fill(~hard[:, None], float("-inf")).amax(dim=-1, keepdim=True)
        peak = torch.nan_to_num(peak.detach(), neginf=0.0)
        scores = (logits - peak).clamp_max(0).exp() * assign[:, None]
        # A non-empty region always sums to >= 1 (its top patch scores exp(0) = 1), so the
        # clamp only affects empty regions.
        weights = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0)
        anchors, pooled = self.update_anchors(anchors, weights, v)

        q, k, v = self.scatter_inputs(x, anchors)
        gate = torch.sigmoid((q @ k.transpose(-2, -1)).float() * self.scale)
        read = (gate * assign.transpose(1, 2)[:, None]).to(v.dtype)
        update = self.scatter_out(merge_heads(read @ v))

        self.keep(weights.mean(1), pooled, positions)
        return update, (anchors, positions)


class FocusRouter(VoronoiRouter):
    """Coverage + focus anchors, as in experiment 06: a coverage_side² grid stays fixed, the
    rest are placed one at a time at the soft-argmax of the learned importance scores, each
    placement suppressing the scores around it; a weak prior pulls each focus anchor towards
    a spread-out home position. All anchors share one exclusive Voronoi partition."""

    SUPPRESSION_WIDTH = SUPPRESSION_PATCHES / GRID

    def __init__(self, num_anchors):
        super().__init__(num_anchors)
        self.importance = nn.Linear(EMBED_DIM, 1)
        coverage_side = max(1, math.isqrt(num_anchors // 4)) if COVERAGE_SIDE == "auto" else int(COVERAGE_SIDE)
        self.num_focus = num_anchors - coverage_side ** 2
        if self.num_focus < 1:
            raise ValueError(f"coverage grid {coverage_side}² leaves no focus anchors out of {num_anchors}")
        self.register_buffer("coverage", grid_positions(coverage_side), persistent=False)
        spread = torch.linspace(0, num_anchors - 1, self.num_focus).round().long()
        homes = grid_positions(math.isqrt(num_anchors))[spread]
        home_distance = (self.coords[None] - homes[:, None]).square().sum(dim=-1)
        self.register_buffer("home_prior", -home_distance / (2 * HOME_WIDTH ** 2), persistent=False)

    def place(self, x, positions):
        scores = self.importance(x).squeeze(-1).float()
        importance = F.softplus(scores) + 1e-3
        peaks = []
        for k in range(self.num_focus):
            peak = (scores + self.home_prior[k]).softmax(dim=-1) @ self.coords
            peaks.append(peak)
            distance = (self.coords[None] - peak[:, None]).square().sum(dim=-1)
            scores = scores - SUPPRESSION * torch.exp(-distance / (2 * self.SUPPRESSION_WIDTH ** 2))
        coverage = self.coverage.expand(x.shape[0], -1, -1)
        return torch.cat([coverage, torch.stack(peaks, dim=1)], dim=1), importance


class DenseRouter(nn.Module):
    """No anchors: global self-attention among all patches in place of the anchor pathway."""

    def __init__(self, num_anchors):
        super().__init__()
        self.attention = SelfAttention(EMBED_DIM, HEADS)

    def forward(self, x, state):
        return self.attention(x), state


ROUTERS = {"densewin": DenseRouter, "fixed": VoronoiRouter, "focus": FocusRouter}


class Block(nn.Module):
    """Local window attention → anchor pathway → FFN."""

    def __init__(self, router):
        super().__init__()
        self.local_norm = nn.LayerNorm(EMBED_DIM)
        self.local = LocalWindowAttention(EMBED_DIM, HEADS, WINDOW)
        self.global_norm = nn.LayerNorm(EMBED_DIM)
        self.router = router
        self.ffn_norm = nn.LayerNorm(EMBED_DIM)
        hidden = int(EMBED_DIM * MLP_RATIO)
        self.ffn = nn.Sequential(nn.Linear(EMBED_DIM, hidden), nn.GELU(), nn.Linear(hidden, EMBED_DIM))

    def forward(self, x, state):
        x = x + self.local(self.local_norm(x), GRID_H, GRID_W)
        update, state = self.router(self.global_norm(x), state)
        x = x + update
        return x + self.ffn(self.ffn_norm(x)), state


class AnchorViT(nn.Module):
    def __init__(self, kind, num_anchors):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, EMBED_DIM, PATCH_SIZE, PATCH_SIZE)
        self.pos_embed = nn.Parameter(torch.randn(1, GRID_H * GRID_W, EMBED_DIM) * 0.02)
        anchors = num_anchors if kind != "densewin" else 1
        self.anchor_init = nn.Parameter(torch.randn(1, anchors, ANCHOR_DIM) * 0.02)
        self.register_buffer("start_positions", grid_positions(math.isqrt(anchors)), persistent=False)
        self.blocks = nn.ModuleList(Block(ROUTERS[kind](num_anchors)) for _ in range(DEPTH))
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def routers(self):
        return [block.router for block in self.blocks if isinstance(block.router, AnchorRouter)]

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
    block. The canvas is area-downsampled so its long side is `long_side` before patching."""

    def __init__(self, long_side, patch):
        super().__init__()
        long_side = long_side or max(CANVAS_H, CANVAS_W)
        scale = long_side / max(CANVAS_H, CANVAS_W)
        self.size = (round(CANVAS_H * scale / patch) * patch, round(CANVAS_W * scale / patch) * patch)
        self.patch_embed = nn.Conv2d(3, EMBED_DIM, patch, patch)
        tokens = (self.size[0] // patch) * (self.size[1] // patch)
        self.pos_embed = nn.Parameter(torch.randn(1, tokens, EMBED_DIM) * 0.02)
        self.blocks = nn.ModuleList(DenseBlock() for _ in range(DEPTH))
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def routers(self):
        return []

    def forward(self, images):
        if self.size != tuple(images.shape[-2:]):
            images = F.interpolate(images, size=self.size, mode="area")
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
        return TimmModel("resnet18")
    raise ValueError(f"unknown RART_KIND {KIND}")


def sdpa_context():
    from contextlib import nullcontext
    if SDPA == "auto" or DEVICE.type != "cuda":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel({"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                        "math": SDPBackend.MATH}[SDPA])


# ---------------------------------------------------------------- Loss and metrics
def loss_fn(logits, target, valid):
    if not MULTILABEL:
        return F.cross_entropy(logits.float(), target)
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp_min(1)


def average_precision(scores, target, valid):
    """Mean over classes of average precision, skipping ignore-flagged entries."""
    aps = []
    for c in range(scores.shape[1]):
        keep = valid[:, c]
        s, t = scores[keep, c], target[keep, c]
        if t.sum() == 0:
            continue
        hits = t[s.argsort(descending=True)]
        precision = hits.cumsum(0) / torch.arange(1, len(hits) + 1)
        aps.append((precision * hits).sum().item() / hits.sum().item())
    return 100.0 * float(np.mean(aps)) if aps else float("nan")


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
        coords = patch_coords().to(positions.device)
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
    for x, _, _, boxes in data.batches("test", train=False):
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
            model(x)
        on_object = object_patch_mask(boxes)
        for total, router in zip(totals, routers):
            for name, value in router_stats(*router.last, on_object).items():
                total[name] += value * x.shape[0]
        seen += x.shape[0]
        if seen >= METRIC_IMAGES:
            break
    for router in routers:
        router.record = False
        router.last = None
    return [{name: value / seen for name, value in total.items()} for total in totals]


def count_gflops_total(model, images):
    """All matmuls, convolutions and attention kernels (forward pass), from torch's
    FlopCounterMode. Elementwise ops are not counted."""
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    with torch.enable_grad(), FlopCounterMode(display=False) as counter:
        model(images)
    return counter.get_total_flops() / images.shape[0] / 1e9


@torch.no_grad()
def throughput(model, images, iterations=10):
    if DEVICE.type != "cuda":
        return None
    model.eval()
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        for _ in range(3):
            model(images)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(images)
        torch.cuda.synchronize()
    return images.shape[0] * iterations / (time.perf_counter() - start)


# ---------------------------------------------------------------- Training
@torch.no_grad()
def evaluate(model, split):
    model.eval()
    total_loss, seen = 0.0, 0
    scores, targets, valids = [], [], []
    for x, y, valid, _ in data.batches(split, train=False):
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
            logits = model(x)
        total_loss += loss_fn(logits, y, valid).item() * y.shape[0]
        seen += y.shape[0]
        scores.append(logits.float().cpu())
        targets.append(y.cpu())
        valids.append(valid.cpu())
    scores, targets, valids = torch.cat(scores), torch.cat(targets), torch.cat(valids)
    if MULTILABEL:
        score = average_precision(scores, targets, valids)
    else:
        score = 100.0 * (scores.argmax(dim=1) == targets).float().mean().item()
    return total_loss / seen, score


def run():
    set_seed(SEED)
    model = build_model().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"[{TAG}] {params:,} parameters", flush=True)

    metrics_init = anchor_metrics(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    steps_per_epoch = math.ceil(len(data.splits["train"]["labels"] if DATASET == "mpmnist"
                                    else data.splits["train"]) / BATCH_SIZE)
    warmup, total = max(1, int(WARMUP_EPOCHS * steps_per_epoch)), NUM_EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warmup if step < warmup
                                                  else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=DEVICE.type == "cuda" and PRECISION == "fp16")
    generator = torch.Generator().manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        start = time.perf_counter()
        total_loss, seen = 0.0, 0
        grad_norms, skipped = [], 0
        for x, y, valid, _ in data.batches("train", train=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
                loss = loss_fn(model(x), y, valid)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP if GRAD_CLIP > 0 else float("inf")).item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if math.isfinite(grad_norm):
                grad_norms.append(grad_norm)
            else:
                skipped += 1
            total_loss += loss.item() * y.shape[0]
            seen += y.shape[0]
        val_loss, val_score = evaluate(model, "val")
        test_loss, test_score = evaluate(model, "test")
        minutes = (time.perf_counter() - start) / 60
        norms = np.array(grad_norms) if grad_norms else np.array([float("nan")])
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "val_loss": val_loss,
                        "val_score": val_score, "test_loss": test_loss, "test_score": test_score,
                        "minutes": minutes, "grad_norm_median": float(np.median(norms)),
                        "grad_norm_max": float(norms.max()), "steps_skipped": skipped})
        print(f"[{TAG}] epoch {epoch:02d}/{NUM_EPOCHS} | train loss {total_loss / seen:.4f} | "
              f"val {val_score:5.2f} | test {test_score:5.2f} | grad norm median {np.median(norms):.2f} "
              f"max {norms.max():.2f} | {minutes:.2f} min", flush=True)

    peak_train_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else None
    config = {"kind": KIND, "num_anchors": NUM_ANCHORS if KIND in ("fixed", "focus") else 0, "seed": SEED,
              "epochs": NUM_EPOCHS, "dataset": DATASET, "canvas": [CANVAS_H, CANVAS_W],
              "image": [IMAGE_H, IMAGE_W], "vit_input": VIT_INPUT, "vit_patch": VIT_PATCH,
              "coverage_side": COVERAGE_SIDE, "suppression": SUPPRESSION,
              "suppression_width": SUPPRESSION_PATCHES, "home_width": HOME_WIDTH, "grad_clip": GRAD_CLIP,
              "warmup_epochs": WARMUP_EPOCHS, "precision": PRECISION, "lr": LR, "batch": BATCH_SIZE,
              "metric": "mAP" if MULTILABEL else "accuracy"}
    result = {"tag": TAG, **config, "params": params, "history": history,
              "final_val_score": history[-1]["val_score"], "final_test_score": history[-1]["test_score"],
              # final_test_acc keeps the experiment 06 runner's field name (mAP for DFG)
              "final_test_acc": history[-1]["test_score"],
              "metrics_init": metrics_init, "metrics_trained": anchor_metrics(model),
              "gflops_total": count_gflops_total(model, sample_x[:2]),
              "throughput_img_s": throughput(model, sample_x[:8]),
              "peak_train_memory_mb": peak_train_mb,
              "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
              "total_minutes": sum(h["minutes"] for h in history)}
    if SAVE_CHECKPOINT:
        torch.save(model.state_dict(), os.path.join(OUT_DIR, TAG + ".pth"))
    return result


def measure(steps=10, warmup=3):
    """Cost of an untrained model at batch RART_MEASURE_BATCH: GFLOPs, training step time,
    inference throughput and peak memory. An out-of-memory error is recorded, not raised."""
    set_seed(SEED)
    batch = int(env("RART_MEASURE_BATCH", str(BATCH_SIZE)))
    base = {"tag": TAG, "measure_only": True, "kind": KIND, "dataset": DATASET, "canvas": [CANVAS_H, CANVAS_W],
            "num_anchors": NUM_ANCHORS if KIND in ("fixed", "focus") else 0, "vit_input": VIT_INPUT,
            "sdpa": SDPA, "batch_size": batch, "final_test_acc": None,
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"}
    model = build_model().to(DEVICE)
    base["params"] = sum(p.numel() for p in model.parameters())
    x = sample_x[:1].expand(batch, -1, -1, -1).contiguous()
    y = sample_y[:1].expand(batch, *sample_y.shape[1:]).contiguous()
    valid = torch.ones_like(y, dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    try:
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
                loss = loss_fn(model(x), y, valid)
            loss.backward()
            optimizer.step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        base["train_step_ms"] = (time.perf_counter() - start) / steps * 1000
        base["peak_train_memory_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else None
    except torch.cuda.OutOfMemoryError:
        base["train_oom"] = True
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        base["throughput_img_s"] = throughput(model, x)
        base["peak_inference_memory_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else None
    except torch.cuda.OutOfMemoryError:
        base["inference_oom"] = True
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    try:
        base["gflops_total"] = count_gflops_total(model, x[:1])
    except torch.cuda.OutOfMemoryError:
        base["gflops_total"] = None
    return base


if __name__ == "__main__":
    if os.path.exists(RESULTS_PATH) and not SMOKE:
        print(f"[{TAG}] already done: {RESULTS_PATH}")
    else:
        with sdpa_context():
            result = measure() if MEASURE_ONLY else run()
        with open(RESULTS_PATH + ".tmp", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        os.replace(RESULTS_PATH + ".tmp", RESULTS_PATH)
        if MEASURE_ONLY:
            print(f"[{TAG}] measured: {result.get('gflops_total') or 0:.1f} GFLOPs | "
                  f"{result.get('train_step_ms', float('nan')):.1f} ms/step | {result.get('throughput_img_s') or 0:.1f} img/s | "
                  f"{result.get('peak_train_memory_mb') or 0:.0f} MB{' | TRAIN OOM' if result.get('train_oom') else ''}",
                  flush=True)
        else:
            print(f"[{TAG}] done: val {result['final_val_score']:.2f} | test {result['final_test_score']:.2f} | "
                  f"{result['gflops_total']:.1f} GFLOPs | {result['throughput_img_s'] or 0:.1f} img/s | "
                  f"saved {RESULTS_PATH}", flush=True)
        gc.collect()
