"""Experiment 03 as a script: the notebook's code cells in order, with CIFAR-100 read from the
pickle files directly. usage: RART_SEED=44 python anchor_collapse.py"""
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
import torchvision


def env(name, default):
    return os.environ.get(name, default)


SEED = int(env("RART_SEED", "42"))
SMOKE = env("RART_SMOKE", "0") == "1"
NUM_EPOCHS = 1 if SMOKE else int(env("RART_EPOCHS", "10"))
NUM_ANCHORS = int(env("RART_ANCHORS", "16"))
MODELS = [
    m.strip()
    for m in env("RART_MODELS", "perceiver,competitive,anchorformer,paca,partition").split(",")
    if m.strip()
]

IMG_SIZE, PATCH_SIZE, BATCH_SIZE, NUM_CLASSES = 224, 16, 128, 100
EMBED_DIM, DEPTH, HEADS, MLP_RATIO, WINDOW = 384, 6, 6, 3.0, 7
ANCHOR_DIM, ANCHOR_HEADS = 128, 4
METRIC_IMAGES = 256 if SMOKE else 1024

WORK_DIR = env(
    "RART_WORK_DIR",
    "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("working"),
)
os.makedirs(WORK_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(
    WORK_DIR,
    f"step0_results_A{NUM_ANCHORS}_seed{SEED}{'_smoke' if SMOKE else ''}.json",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


print("PyTorch:", torch.__version__, "| device:", DEVICE,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("Models:", MODELS, "| anchors:", NUM_ANCHORS, "| epochs:", NUM_EPOCHS,
      "| seed:", SEED, "(SMOKE TEST)" if SMOKE else "")
print("Results file:", RESULTS_PATH)

def find_cifar_root():
    if os.path.isdir("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            if "cifar-100-python" in dirs:
                return root, False
    root = os.path.join(WORK_DIR, "data")
    return root, not os.path.isdir(os.path.join(root, "cifar-100-python"))


cifar_root, need_download = find_cifar_root()


def load_split(train):
    """Reads the pickle files directly (no torchvision MD5 check), so a copy rebuilt from the
    Hugging Face mirror by ../06_main_comparison/build_extra_data.py also loads."""
    if need_download:
        torchvision.datasets.CIFAR100(cifar_root, train=train, download=True)
    import pickle
    with open(os.path.join(cifar_root, "cifar-100-python", "train" if train else "test"), "rb") as f:
        entry = pickle.load(f, encoding="latin1")
    images = torch.from_numpy(entry["data"].reshape(-1, 3, 32, 32)).contiguous()
    labels = torch.tensor(entry["fine_labels"], dtype=torch.long)
    return images, labels


train_x, train_y = load_split(True)
test_x, test_y = load_split(False)
if SMOKE:
    train_x, train_y = train_x[:512], train_y[:512]
    test_x, test_y = test_x[:256], test_y[:256]
train_x, train_y, test_x, test_y = (t.to(DEVICE) for t in (train_x, train_y, test_x, test_y))

MEAN = torch.tensor([0.5071, 0.4867, 0.4408], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.2675, 0.2565, 0.2761], device=DEVICE).view(1, 3, 1, 1)


def iterate_batches(images, labels, train, generator=None):
    n = images.shape[0]
    if train:
        order = torch.randperm(n, generator=generator).to(DEVICE)
    else:
        order = torch.arange(n, device=DEVICE)
    for start in range(0, n, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        x = images[idx].float().div_(255)
        if train:
            flip = (torch.rand(idx.shape[0], generator=generator) < 0.5).to(DEVICE)
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        yield (x - MEAN) / STD, labels[idx]


print("Train:", tuple(train_x.shape), "| Test:", tuple(test_x.shape), "| CIFAR root:", cifar_root)

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


def balanced_partition(grid, num_anchors):
    side = math.isqrt(num_anchors)
    if side * side != num_anchors:
        raise ValueError("the partition control needs a square number of anchors")
    band = torch.arange(grid) * side // grid
    return (band[:, None] * side + band[None, :]).flatten()

class Router(nn.Module):
    stateful = False

    def __init__(self):
        super().__init__()
        self.record = False
        self.last = None

    def keep(self, pooling, summaries):
        if self.record:
            self.last = (pooling.detach().float(), summaries.detach().float())


class AnchorStateRouter(Router):
    """Gather patches into anchors, let anchors attend to each other, scatter back
    (the RART Local → Global → Local pathway). Subclasses choose how anchors pool."""

    stateful = True

    def __init__(self, num_anchors, grid):
        super().__init__()
        da = ANCHOR_DIM
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
        self.grid = grid
        self.local = None   # optional 3x3 depth-wise conv bypass on the patch values

    def pool_weights(self, logits):
        raise NotImplementedError

    def scatter(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    def forward(self, x, anchors):
        k, v = (split_heads(t, self.heads) for t in self.gather_kv(self.gather_in(x)).chunk(2, dim=-1))
        q = split_heads(self.gather_q(anchors), self.heads)
        weights = self.pool_weights((q @ k.transpose(-2, -1)).float() * self.scale)
        pooled = merge_heads(weights.to(v.dtype) @ v)
        anchors = anchors + self.gather_out(pooled)
        anchors = anchors + self.anchor_attention(self.anchor_norm(anchors))

        k2, v2 = (split_heads(t, self.heads) for t in self.scatter_kv(self.scatter_norm(anchors)).chunk(2, dim=-1))
        q2 = split_heads(self.scatter_q(self.scatter_in(x)), self.heads)
        read = merge_heads(self.scatter(q2, k2, v2))
        if self.local is not None:
            values = merge_heads(v).transpose(1, 2).reshape(x.shape[0], -1, self.grid, self.grid)
            read = read + self.local(values).flatten(2).transpose(1, 2)
        update = self.scatter_out(read)

        self.keep(weights.mean(1), pooled)
        return update, anchors


class PerceiverRouter(AnchorStateRouter):
    """RART-16/32 routing: each anchor softmaxes over all patches."""

    def pool_weights(self, logits):
        return logits.softmax(dim=-1)


class CompetitiveRouter(AnchorStateRouter):
    """Slot-attention style: patches are shared out between anchors (softmax over
    anchors), then each anchor takes the weighted mean of what it won."""

    def pool_weights(self, logits):
        weights = logits.softmax(dim=-2)
        return weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)


class PartitionRouter(AnchorStateRouter):
    """Partition RART: each anchor pools only its own fixed region, and each patch
    reads back only its own anchor through a sigmoid gate."""

    def __init__(self, num_anchors, grid):
        super().__init__(num_anchors, grid)
        region = balanced_partition(grid, num_anchors)
        self.register_buffer("region", region, persistent=False)
        self.register_buffer(
            "allowed", region[None, :] == torch.arange(num_anchors)[:, None], persistent=False
        )

    def pool_weights(self, logits):
        return logits.masked_fill(~self.allowed, float("-inf")).softmax(dim=-1)

    def scatter(self, q, k, v):
        k, v = k[:, :, self.region], v[:, :, self.region]
        gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * self.scale)
        return gate * v


class PartitionLocalRouter(PartitionRouter):
    """Ablation: the fixed partition plus Representative Attention's 3x3 depth-wise
    convolution bypass on the patch values, added before the output projection."""

    def __init__(self, num_anchors, grid):
        super().__init__(num_anchors, grid)
        self.local = nn.Conv2d(ANCHOR_DIM, ANCHOR_DIM, 3, padding=1, groups=ANCHOR_DIM)


class AnchorFormerRouter(Router):
    """AnchorFormer: learned anchors, token→anchor softmax over anchors, and the
    Markov round trip H = A Δ⁻¹ Aᵀ V. No anchor state or anchor self-attention."""

    def __init__(self, num_anchors, grid):
        super().__init__()
        da = ANCHOR_DIM
        self.heads = ANCHOR_HEADS
        self.scale = (da // ANCHOR_HEADS) ** -0.5
        self.proj_in = nn.Linear(EMBED_DIM, da)
        self.kv = nn.Linear(da, da * 2)
        self.anchors = nn.Parameter(torch.randn(ANCHOR_HEADS, num_anchors, da // ANCHOR_HEADS) * 0.02)
        self.out = nn.Linear(da, EMBED_DIM)

    def forward(self, x, anchors):
        k, v = (split_heads(t, self.heads) for t in self.kv(self.proj_in(x)).chunk(2, dim=-1))
        token_to_anchor = ((k.float() @ self.anchors.float().transpose(-2, -1)) * self.scale).softmax(dim=-1)
        column_mass = token_to_anchor.sum(dim=-2, keepdim=True) + 1e-6
        pooling = (token_to_anchor / column_mass).transpose(-2, -1)
        summaries = pooling @ v.float()
        update = self.out(merge_heads((token_to_anchor @ summaries).to(x.dtype)))
        self.keep(pooling.mean(1), merge_heads(summaries))
        return update, anchors


class PaCaRouter(Router):
    """PaCa-ViT: an MLP predicts cluster assignments (softmax over patches), clusters
    are assignment-weighted means of patches, and patches attend to the clusters.
    The assignment MLP uses a hidden width of EMBED_DIM rather than PaCa's 4× to keep
    the parameter count close to the other variants."""

    def __init__(self, num_anchors, grid):
        super().__init__()
        da = ANCHOR_DIM
        self.heads = ANCHOR_HEADS
        self.assign = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU(), nn.Linear(EMBED_DIM, num_anchors)
        )
        self.cluster_norm = nn.LayerNorm(EMBED_DIM)
        self.q = nn.Linear(EMBED_DIM, da)
        self.kv = nn.Linear(EMBED_DIM, da * 2)
        self.out = nn.Linear(da, EMBED_DIM)

    def forward(self, x, anchors):
        pooling = self.assign(x).float().softmax(dim=1).transpose(1, 2)
        clusters = self.cluster_norm(pooling.to(x.dtype) @ x)
        q = split_heads(self.q(x), self.heads)
        k, v = (split_heads(t, self.heads) for t in self.kv(clusters).chunk(2, dim=-1))
        update = self.out(merge_heads(F.scaled_dot_product_attention(q, k, v)))
        self.keep(pooling, clusters)
        return update, anchors


class RPAttentionRouter(Router):
    """Representative Attention (Li et al., 2026), implemented from the paper's equations;
    the official repository had no code. Per-head learned representatives W_g; tokens are
    routed by a softmax over representatives of K W_g (Eq. 5, unscaled as in Eq. 14), each
    representative's assignments are normalised by its total mass (Eq. 6), and latent keys
    and values are the mass-weighted means (Eq. 7). The latents are layer-normalised and
    refined by residual self-attention driven by the values (Eqs. 8-9); patches then read
    them by query attention over the normalised latent keys, plus a 3x3 depth-wise
    convolution bypass on the values (Eq. 10). Width and heads match the other pathways."""

    use_local = True

    def __init__(self, num_anchors, grid):
        super().__init__()
        da = ANCHOR_DIM
        head_dim = da // ANCHOR_HEADS
        self.heads = ANCHOR_HEADS
        self.grid = grid
        self.qkv = nn.Linear(EMBED_DIM, da * 3)
        self.gather = nn.Parameter(torch.randn(ANCHOR_HEADS, head_dim, num_anchors) * head_dim ** -0.5)
        self.key_norm = nn.LayerNorm(head_dim)
        self.value_norm = nn.LayerNorm(head_dim)
        self.latent_qkv = nn.Linear(head_dim, head_dim * 3)
        self.local = nn.Conv2d(da, da, 3, padding=1, groups=da) if self.use_local else None
        self.out = nn.Linear(da, EMBED_DIM)

    def forward(self, x, anchors):
        batch = x.shape[0]
        q, k, v = (split_heads(t, self.heads) for t in self.qkv(x).chunk(3, dim=-1))
        assign = (k.float() @ self.gather.float()).softmax(dim=-1)
        pooling = (assign / (assign.sum(dim=-2, keepdim=True) + 1e-6)).transpose(-2, -1)
        latent_k, latent_v = pooling @ k.float(), pooling @ v.float()
        latent_k, latent_v = self.key_norm(latent_k), self.value_norm(latent_v)
        lq, lk, lv = self.latent_qkv(latent_v).chunk(3, dim=-1)
        refined = latent_v + F.scaled_dot_product_attention(lq, lk, lv).float()
        read = F.scaled_dot_product_attention(q, latent_k.to(q.dtype), refined.to(q.dtype))
        read = merge_heads(read)
        if self.local is not None:
            values = merge_heads(v).transpose(1, 2).reshape(batch, -1, self.grid, self.grid)
            read = read + self.local(values).flatten(2).transpose(1, 2)
        update = self.out(read)
        self.keep(pooling.mean(1), merge_heads(latent_v))
        return update, anchors


class RPAttentionNoLocalRouter(RPAttentionRouter):
    """Ablation: Representative Attention without its depth-wise convolution bypass."""

    use_local = False


class DenseRouter(Router):
    """Reference without a bottleneck: global self-attention over all patches in place of
    the anchor pathway (same local attention, FFN and head). Has no anchors to measure."""

    def __init__(self, num_anchors, grid):
        super().__init__()
        self.attention = SelfAttention(EMBED_DIM, HEADS)

    def forward(self, x, anchors):
        return self.attention(x), anchors


ROUTERS = {
    "dense": DenseRouter,
    "perceiver": PerceiverRouter,
    "competitive": CompetitiveRouter,
    "anchorformer": AnchorFormerRouter,
    "rpattention": RPAttentionRouter,
    "rpattention_nolocal": RPAttentionNoLocalRouter,
    "partition_local": PartitionLocalRouter,
    "paca": PaCaRouter,
    "partition": PartitionRouter,
}


class Block(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.local_norm = nn.LayerNorm(EMBED_DIM)
        self.local = LocalWindowAttention(EMBED_DIM, HEADS, WINDOW)
        self.global_norm = nn.LayerNorm(EMBED_DIM)
        self.router = router
        self.ffn_norm = nn.LayerNorm(EMBED_DIM)
        hidden = int(EMBED_DIM * MLP_RATIO)
        self.ffn = nn.Sequential(nn.Linear(EMBED_DIM, hidden), nn.GELU(), nn.Linear(hidden, EMBED_DIM))

    def forward(self, x, anchors, grid):
        x = x + self.local(self.local_norm(x), grid)
        update, anchors = self.router(self.global_norm(x), anchors)
        x = x + update
        x = x + self.ffn(self.ffn_norm(x))
        return x, anchors


class AnchorBottleneckViT(nn.Module):
    def __init__(self, kind, num_anchors):
        super().__init__()
        self.grid = IMG_SIZE // PATCH_SIZE
        router_cls = ROUTERS[kind]
        self.patch_embed = nn.Conv2d(3, EMBED_DIM, PATCH_SIZE, PATCH_SIZE)
        self.pos_embed = nn.Parameter(torch.randn(1, self.grid ** 2, EMBED_DIM) * 0.02)
        self.anchor_init = (
            nn.Parameter(torch.randn(1, num_anchors, ANCHOR_DIM) * 0.02) if router_cls.stateful else None
        )
        self.blocks = nn.ModuleList(Block(router_cls(num_anchors, self.grid)) for _ in range(DEPTH))
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def routers(self):
        return [block.router for block in self.blocks]

    def forward(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed
        anchors = None
        if self.anchor_init is not None:
            anchors = self.anchor_init.expand(x.shape[0], -1, -1)
        for block in self.blocks:
            x, anchors = block(x, anchors, self.grid)
        return self.head(self.norm(x).mean(dim=1))


for kind in ROUTERS:
    if kind.startswith("partition") and math.isqrt(NUM_ANCHORS) ** 2 != NUM_ANCHORS:
        continue
    probe = AnchorBottleneckViT(kind, NUM_ANCHORS)
    print(f"{kind:13s} {sum(p.numel() for p in probe.parameters()):>12,d} parameters")
    del probe

def off_diagonal_mean(similarity):
    size = similarity.shape[-1]
    mask = ~torch.eye(size, dtype=torch.bool, device=similarity.device)
    return similarity[:, mask].mean().item()


def anchor_stats(pooling, summaries):
    entropy = -(pooling * pooling.clamp_min(1e-12).log()).sum(dim=-1)
    pooling_unit = F.normalize(pooling, dim=-1)
    summary_unit = F.normalize(summaries, dim=-1)
    centred = summaries - summaries.mean(dim=1, keepdim=True)
    spread = centred.square().sum(dim=(1, 2)) / summaries.square().sum(dim=(1, 2)).clamp_min(1e-12)
    return {
        "map_similarity": off_diagonal_mean(pooling_unit @ pooling_unit.transpose(-2, -1)),
        "effective_patches": entropy.exp().mean().item(),
        "summary_similarity": off_diagonal_mean(summary_unit @ summary_unit.transpose(-2, -1)),
        "anchor_spread": spread.mean().item(),
    }


@torch.no_grad()
def collapse_metrics(model):
    model.eval()
    routers = model.routers()
    if any(isinstance(router, DenseRouter) for router in routers):
        return []
    for router in routers:
        router.record = True
    totals = [defaultdict(float) for _ in routers]
    seen = 0
    for x, _ in iterate_batches(test_x[:METRIC_IMAGES], test_y[:METRIC_IMAGES], train=False):
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            model(x)
        for total, router in zip(totals, routers):
            for name, value in anchor_stats(*router.last).items():
                total[name] += value * x.shape[0]
        seen += x.shape[0]
    for router in routers:
        router.record = False
        router.last = None
    return [{name: value / seen for name, value in total.items()} for total in totals]

@torch.no_grad()
def evaluate(model, criterion):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in iterate_batches(test_x, test_y, train=False):
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * y.shape[0]
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += y.shape[0]
    return total_loss / seen, 100.0 * correct / seen


def run_experiment(kind):
    set_seed(SEED)
    model = AnchorBottleneckViT(kind, NUM_ANCHORS).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {kind} | {params:,} parameters ===")

    metrics_init = collapse_metrics(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=DEVICE.type == "cuda")
    generator = torch.Generator().manual_seed(SEED)

    history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        start = time.perf_counter()
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in iterate_batches(train_x, train_y, train=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * y.shape[0]
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += y.shape[0]
        scheduler.step()
        test_loss, test_acc = evaluate(model, criterion)
        minutes = (time.perf_counter() - start) / 60
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "train_acc": 100.0 * correct / seen,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "minutes": minutes,
        })
        print(f"{kind} epoch {epoch:02d} | train acc {100.0 * correct / seen:5.2f}% | "
              f"test acc {test_acc:5.2f}% | {minutes:.2f} min")

    metrics_trained = collapse_metrics(model)
    result = {
        "kind": kind,
        "num_anchors": NUM_ANCHORS,
        "seed": SEED,
        "epochs": NUM_EPOCHS,
        "params": params,
        "history": history,
        "metrics_init": metrics_init,
        "metrics_trained": metrics_trained,
    }
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

results = {}
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    print("Loaded finished runs:", sorted(results))

for kind in MODELS:
    if kind in results:
        print(f"Skipping {kind}: already in {RESULTS_PATH}")
        continue
    if kind == "partition" and math.isqrt(NUM_ANCHORS) ** 2 != NUM_ANCHORS:
        print(f"Skipping partition: {NUM_ANCHORS} anchors is not a square number")
        continue
    results[kind] = run_experiment(kind)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved:", RESULTS_PATH)

METRIC_NAMES = ["map_similarity", "effective_patches", "summary_similarity", "anchor_spread"]


def block_mean(block_metrics, name):
    return float(np.mean([block[name] for block in block_metrics]))


lines = [
    f"| Model | Params | Final test acc | When | Map sim (mean / last block) | "
    f"Effective patches (of {(IMG_SIZE // PATCH_SIZE) ** 2}) | Summary sim | Anchor spread |",
    "|---|---:|---:|---|---:|---:|---:|---:|",
]
for kind, result in results.items():
    for when in ("init", "trained"):
        blocks = result[f"metrics_{when}"]
        acc = f"{result['history'][-1]['test_acc']:.2f}%" if when == "trained" else ""
        params = f"{result['params'] / 1e6:.2f}M" if when == "init" else ""
        if not blocks:
            lines.append(f"| {kind if when == 'init' else ''} | {params} | {acc} | {when} | — | — | — | — |")
            continue
        lines.append(
            f"| {kind if when == 'init' else ''} | {params} | {acc} | {when} | "
            f"{block_mean(blocks, 'map_similarity'):.3f} / {blocks[-1]['map_similarity']:.3f} | "
            f"{block_mean(blocks, 'effective_patches'):.1f} | "
            f"{block_mean(blocks, 'summary_similarity'):.3f} | "
            f"{block_mean(blocks, 'anchor_spread'):.3f} |"
        )
table = "\n".join(lines)
print(table)

with open(RESULTS_PATH.replace(".json", ".md"), "w", encoding="utf-8") as f:
    f.write(f"# Experiment 03 results ({NUM_ANCHORS} anchors, seed {SEED}, {NUM_EPOCHS} epochs)\n\n{table}\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(METRIC_NAMES), figsize=(5 * len(METRIC_NAMES), 4))
blocks_axis = np.arange(1, DEPTH + 1)
for kind, result in results.items():
    if not result["metrics_trained"]:
        continue
    for ax, name in zip(axes, METRIC_NAMES):
        line, = ax.plot(blocks_axis, [b[name] for b in result["metrics_trained"]], marker="o", label=kind)
        ax.plot(blocks_axis, [b[name] for b in result["metrics_init"]], linestyle="--",
                color=line.get_color(), alpha=0.5)
for ax, name in zip(axes, METRIC_NAMES):
    ax.set_title(name.replace("_", " ") + " (solid: trained, dashed: init)")
    ax.set_xlabel("block")
    ax.grid(alpha=0.3)
axes[0].legend()
plt.tight_layout()
plt.savefig(RESULTS_PATH.replace(".json", ".png"), dpi=120)

