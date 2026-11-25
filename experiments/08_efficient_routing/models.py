"""Experiment 08: an efficient implementation of exclusive-region anchor routing.

Experiment 07's routers compute the gather and scatter steps with dense anchors x patches
matrices for every head. With exclusive (Voronoi) regions, each patch only exchanges
information with its own anchor in the forward pass, so both steps can use per-patch anchor
indices instead: memory and time O(patches) rather than O(patches x anchors).

The only part that needs every anchor-patch pair is the straight-through (ST) gradient that
moves focus anchors. It is recomputed in chunks during the backward pass instead of being
stored, so the indexed version computes the *same* gradients as the reference (see
test_equivalence.py). Fixed anchors have constant positions, so for them the ST term is skipped.

  impl="reference": experiment 07's maths, kept for the equivalence test and as a baseline.
  impl="indexed":   the efficient version.
  placement="serial":   experiment 07's focus placement (one anchor at a time, suppression).
  placement="parallel": all focus anchors placed at once, without suppression. This *changes the
                        model*: it is experiment 06's "no suppression" ablation, which matched
                        the full model's accuracy when trained with gradient clipping.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    grid_h: int
    grid_w: int
    embed_dim: int = 384
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 3.0
    anchor_dim: int = 128
    anchor_heads: int = 4
    window: int = 8
    patch: int = 16
    num_anchors: int = 64
    num_classes: int = 20
    suppression: float = 10.0
    suppression_patches: float = 1.5
    home_width: float = 0.35
    impl: str = "indexed"        # reference | indexed
    placement: str = "serial"    # serial | parallel
    st_chunk: int = 4096         # patches per chunk when recomputing the ST gradient
    # Reference only. Experiment 07 wrote the straight-through assignment as
    # `hard + soft - soft.detach()`, which evaluates to (hard + soft) - soft: for a patch's own
    # anchor that is 1 +- one rounding step, not exactly 1. In a single-patch region a value just
    # below 1 drops the region's sum under the clamp at 1, so that patch's logit gets a spurious
    # gradient (the true gradient is 0: the weight is always 1). exact_st=True uses
    # `hard + (soft - soft.detach())`, which is exactly `hard` in the forward pass; the indexed
    # version always behaves like this.
    exact_st: bool = False
    # Recompute the router in the backward pass instead of storing its activations
    # (torch.utils.checkpoint). Same maths, less memory, some extra compute.
    checkpoint_router: bool = False

    @property
    def grid(self):
        return max(self.grid_h, self.grid_w)


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


def patch_coords(cfg):
    """Patch centres, row-major, in units of the long side."""
    rows = (torch.arange(cfg.grid_h, dtype=torch.float32) + 0.5) / cfg.grid
    cols = (torch.arange(cfg.grid_w, dtype=torch.float32) + 0.5) / cfg.grid
    r, c = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([r, c], dim=-1).reshape(-1, 2)


def grid_positions(cfg, side):
    centres = (torch.arange(side, dtype=torch.float32) + 0.5) / side
    r, c = torch.meshgrid(centres * cfg.grid_h / cfg.grid, centres * cfg.grid_w / cfg.grid, indexing="ij")
    return torch.stack([r, c], dim=-1).reshape(-1, 2)


# ---------------------------------------------------------------- Straight-through terms
# Each returns zeros in the forward pass; added to a per-head output, its backward pass
# receives that output's gradient and returns the reference model's gradient for the soft
# assignment, recomputed over all anchor-patch pairs in chunks of patches.
class _GatherST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft, q, k, v, log_importance, peak, denom, pooled, chunk):
        ctx.save_for_backward(q, k, v, log_importance, peak, denom, pooled)
        ctx.chunk = chunk
        return torch.zeros_like(pooled)

    @staticmethod
    def backward(ctx, grad_pooled):
        q, k, v, log_importance, peak, denom, pooled = ctx.saved_tensors
        scale = q.shape[-1] ** -0.5
        g = grad_pooled.float()                                   # (B, H, A, dh)
        c = (g * pooled).sum(-1, keepdim=True)                    # (B, H, A, 1)
        qf, gd = q.float(), g / denom[..., None]
        out = []
        for start in range(0, k.shape[2], ctx.chunk):
            sl = slice(start, start + ctx.chunk)
            logits = qf @ k[:, :, sl].float().transpose(-2, -1) * scale     # (B, H, A, n)
            if log_importance is not None:
                logits = logits + log_importance[:, None, None, sl]
            e = (logits - peak[..., None]).clamp_max(0).exp()
            p = gd @ v[:, :, sl].float().transpose(-2, -1) - c / denom[..., None]
            out.append((e * p).sum(1))                            # (B, A, n)
        return torch.cat(out, dim=-1), None, None, None, None, None, None, None, None


class _ScatterST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft, q, k, v, out_like, chunk):
        ctx.save_for_backward(q, k, v)
        ctx.chunk = chunk
        return torch.zeros_like(out_like)

    @staticmethod
    def backward(ctx, grad_read):
        q, k, v = ctx.saved_tensors
        scale = q.shape[-1] ** -0.5
        kf, vf = k.float(), v.float()                             # (B, H, A, dh)
        out = []
        for start in range(0, q.shape[2], ctx.chunk):
            sl = slice(start, start + ctx.chunk)
            gate = torch.sigmoid(q[:, :, sl].float() @ kf.transpose(-2, -1) * scale)   # (B, H, n, A)
            u = grad_read[:, :, sl].float() @ vf.transpose(-2, -1)                    # (B, H, n, A)
            out.append((gate * u).sum(1).transpose(1, 2))                             # (B, A, n)
        return torch.cat(out, dim=-1), None, None, None, None, None


# ---------------------------------------------------------------- Anchor pathways
class VoronoiRouter(nn.Module):
    """Exclusive Voronoi partition of the patches among anchors on a fixed grid."""

    def __init__(self, cfg):
        super().__init__()
        da = cfg.anchor_dim
        self.cfg = cfg
        self.num_anchors = cfg.num_anchors
        self.heads = cfg.anchor_heads
        self.scale = (da // cfg.anchor_heads) ** -0.5
        # Experiment 07's gather_in and scatter_in (two 384 -> 128 layers on the same input),
        # fused into one layer: same function, and autocast keeps one bf16 copy of the input.
        self.input_proj = nn.Linear(cfg.embed_dim, 2 * da)
        self.gather_q = nn.Linear(da, da)
        self.gather_kv = nn.Linear(da, da * 2)
        self.gather_out = nn.Linear(da, da)
        self.anchor_norm = nn.LayerNorm(da)
        self.anchor_attention = SelfAttention(da, cfg.anchor_heads)
        self.scatter_norm = nn.LayerNorm(da)
        self.scatter_q = nn.Linear(da, da)
        self.scatter_kv = nn.Linear(da, da * 2)
        self.scatter_out = nn.Linear(da, cfg.embed_dim)
        self.register_buffer("coords", patch_coords(cfg), persistent=False)
        self.tau = (0.5 / math.isqrt(cfg.num_anchors)) ** 2
        self.position_embed = nn.Sequential(nn.Linear(2, da), nn.GELU(), nn.Linear(da, da))

    def place(self, x, positions):
        return positions, None

    def soft_assignment(self, positions):
        distance = (positions[:, :, None, :] - self.coords[None, None]).square().sum(dim=-1)
        return (-distance / self.tau).softmax(dim=1)              # (B, A, N)

    def update_anchors(self, anchors, pooled_heads):
        pooled = merge_heads(pooled_heads)
        anchors = anchors + self.gather_out(pooled)
        return anchors + self.anchor_attention(self.anchor_norm(anchors))

    def forward(self, x, state):
        anchors, positions = state
        positions, importance = self.place(x, positions)
        located = anchors + self.position_embed(positions).to(anchors.dtype)
        gather_x, scatter_x = self.input_proj(x).chunk(2, dim=-1)
        k, v = (split_heads(t, self.heads) for t in self.gather_kv(gather_x).chunk(2, dim=-1))
        q = split_heads(self.gather_q(located), self.heads)
        route = self.route_reference if self.cfg.impl == "reference" else self.route_indexed
        update, anchors = route(scatter_x, anchors, positions, importance, q, k, v)
        return update, (anchors, positions)

    # Experiment 07's computation (dense anchors x patches matrices). `scatter_x` is the patches
    # projected to the anchor width for the scatter step.
    def route_reference(self, scatter_x, anchors, positions, importance, q, k, v):
        soft = self.soft_assignment(positions)
        hard = F.one_hot(soft.argmax(dim=1), self.num_anchors).transpose(1, 2).to(soft.dtype)
        st = hard + (soft - soft.detach()) if self.cfg.exact_st else hard + soft - soft.detach()
        assign, mask = st, hard > 0.5
        logits = (q @ k.transpose(-2, -1)).float() * self.scale
        if importance is not None:
            logits = logits + importance.log()[:, None, None, :]
        peak = logits.masked_fill(~mask[:, None], float("-inf")).amax(dim=-1, keepdim=True)
        peak = torch.nan_to_num(peak.detach(), neginf=0.0)
        scores = (logits - peak).clamp_max(0).exp() * assign[:, None]
        weights = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0)
        anchors = self.update_anchors(anchors, weights.to(v.dtype) @ v)

        k2, v2 = (split_heads(t, self.heads) for t in self.scatter_kv(self.scatter_norm(anchors)).chunk(2, dim=-1))
        q2 = split_heads(self.scatter_q(scatter_x), self.heads)
        gate = torch.sigmoid((q2 @ k2.transpose(-2, -1)).float() * self.scale)
        read = (gate * assign.transpose(1, 2)[:, None]).to(v2.dtype)
        return self.scatter_out(merge_heads(read @ v2)), anchors

    # Per-patch indices: each patch only talks to its own anchor.
    def route_indexed(self, scatter_x, anchors, positions, importance, q, k, v):
        batch, heads, num_patches, dh = k.shape
        num_anchors, device = self.num_anchors, k.device
        acc = torch.promote_types(k.dtype, torch.float32)        # fp32 sums (fp64 in the test)
        moving = positions.requires_grad and torch.is_grad_enabled()
        soft = self.soft_assignment(positions) if moving else None
        with torch.no_grad():
            distance = (positions[:, :, None, :] - self.coords[None, None]).square().sum(dim=-1)
            owner = distance.argmin(dim=1)                        # (B, N)
        # Flat segment ids over (batch, head, anchor), so every reduction is a 1-D index_add.
        offsets = (torch.arange(batch * heads, device=device) * num_anchors).view(batch, heads, 1)
        segment = (owner[:, None, :] + offsets).reshape(-1)        # (B*H*N,)
        owner_bh = owner[:, None, :].expand(batch, heads, num_patches)

        # Gather: softmax of each patch's logit within its own region, with experiment 07's
        # max-shift and clamp, then a weighted sum of values per region. Products stay in the
        # compute dtype (bf16 under autocast, as in experiment 07's matmuls); sums are fp32.
        q_own = q.gather(2, owner_bh[..., None].expand(-1, -1, -1, dh))           # (B, H, N, dh)
        logits = (q_own * k).sum(-1, dtype=acc) * self.scale                         # (B, H, N)
        log_importance = importance.log() if importance is not None else None
        if log_importance is not None:
            logits = logits + log_importance[:, None, :]
        peak = torch.full((batch * heads * num_anchors,), float("-inf"), device=device, dtype=acc)
        peak = peak.scatter_reduce(0, segment, logits.detach().reshape(-1), "amax")
        peak = torch.nan_to_num(peak, neginf=0.0)
        e = (logits - peak[segment].view_as(logits)).clamp_max(0).exp()
        denom = torch.zeros(batch * heads * num_anchors, device=device, dtype=acc).index_add(0, segment, e.reshape(-1))
        denom = denom.clamp_min(1.0)
        w = e / denom[segment].view_as(e)
        pooled = torch.zeros(batch * heads * num_anchors, dh, device=device, dtype=acc)
        pooled = pooled.index_add(0, segment, (w.to(v.dtype)[..., None] * v).to(acc).reshape(-1, dh))
        pooled = pooled.view(batch, heads, num_anchors, dh)
        if moving:
            pooled = pooled + _GatherST.apply(
                soft, q.detach(), k.detach(), v.detach(),
                None if log_importance is None else log_importance.detach(),
                peak.view(batch, heads, num_anchors), denom.view(batch, heads, num_anchors).detach(),
                pooled.detach(), self.cfg.st_chunk)
        anchors = self.update_anchors(anchors, pooled.to(v.dtype))

        # Scatter: each patch reads its own anchor through a sigmoid gate.
        k2, v2 = (split_heads(t, self.heads) for t in self.scatter_kv(self.scatter_norm(anchors)).chunk(2, dim=-1))
        q2 = split_heads(self.scatter_q(scatter_x), self.heads)
        index = owner_bh[..., None].expand(-1, -1, -1, dh)
        k2_own, v2_own = k2.gather(2, index), v2.gather(2, index)
        gate = torch.sigmoid((q2 * k2_own).sum(-1, dtype=acc) * self.scale)        # (B, H, N)
        read = gate[..., None].to(v2.dtype) * v2_own                               # (B, H, N, dh)
        if moving:
            read = read + _ScatterST.apply(soft, q2.detach(), k2.detach(), v2.detach(), read,
                                           self.cfg.st_chunk)
        return self.scatter_out(merge_heads(read)), anchors


class FocusRouter(VoronoiRouter):
    """Coverage anchors on a fixed coarse grid plus focus anchors placed at the soft-argmax of
    learned importance scores; all anchors share one exclusive Voronoi partition."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.importance = nn.Linear(cfg.embed_dim, 1)
        coverage_side = max(1, math.isqrt(cfg.num_anchors // 4))
        self.num_focus = cfg.num_anchors - coverage_side ** 2
        self.suppression_width = cfg.suppression_patches / cfg.grid
        self.register_buffer("coverage", grid_positions(cfg, coverage_side), persistent=False)
        spread = torch.linspace(0, cfg.num_anchors - 1, self.num_focus).round().long()
        homes = grid_positions(cfg, math.isqrt(cfg.num_anchors))[spread]
        home_distance = (self.coords[None] - homes[:, None]).square().sum(dim=-1)
        self.register_buffer("home_prior", -home_distance / (2 * cfg.home_width ** 2), persistent=False)

    def place(self, x, positions):
        scores = self.importance(x).squeeze(-1).float()
        importance = F.softplus(scores) + 1e-3
        if self.cfg.placement == "parallel":
            peaks = (scores[:, None, :] + self.home_prior[None]).softmax(dim=-1) @ self.coords
        else:
            found = []
            for k in range(self.num_focus):
                peak = (scores + self.home_prior[k]).softmax(dim=-1) @ self.coords
                found.append(peak)
                distance = (self.coords[None] - peak[:, None]).square().sum(dim=-1)
                scores = scores - self.cfg.suppression * torch.exp(-distance / (2 * self.suppression_width ** 2))
            peaks = torch.stack(found, dim=1)
        coverage = self.coverage.expand(x.shape[0], -1, -1)
        return torch.cat([coverage, peaks], dim=1), importance


class DenseRouter(nn.Module):
    """No anchors: global self-attention among all patches (densewin)."""

    def __init__(self, cfg):
        super().__init__()
        self.attention = SelfAttention(cfg.embed_dim, cfg.heads)

    def forward(self, x, state):
        return self.attention(x), state


ROUTERS = {"fixed": VoronoiRouter, "focus": FocusRouter, "densewin": DenseRouter}


class Block(nn.Module):
    def __init__(self, cfg, router):
        super().__init__()
        self.cfg = cfg
        self.local_norm = nn.LayerNorm(cfg.embed_dim)
        self.local = LocalWindowAttention(cfg.embed_dim, cfg.heads, cfg.window)
        self.global_norm = nn.LayerNorm(cfg.embed_dim)
        self.router = router
        self.ffn_norm = nn.LayerNorm(cfg.embed_dim)
        hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(cfg.embed_dim, hidden), nn.GELU(), nn.Linear(hidden, cfg.embed_dim))

    def forward(self, x, state):
        x = x + self.local(self.local_norm(x), self.cfg.grid_h, self.cfg.grid_w)
        if self.cfg.checkpoint_router and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            update, state = checkpoint(self.router, self.global_norm(x), state, use_reentrant=False)
        else:
            update, state = self.router(self.global_norm(x), state)
        x = x + update
        return x + self.ffn(self.ffn_norm(x)), state


class AnchorViT(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        self.patch_embed = nn.Conv2d(3, d, cfg.patch, cfg.patch)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.grid_h * cfg.grid_w, d) * 0.02)
        anchors = cfg.num_anchors if kind != "densewin" else 1
        self.anchor_init = nn.Parameter(torch.randn(1, anchors, cfg.anchor_dim) * 0.02)
        self.register_buffer("start_positions", grid_positions(cfg, math.isqrt(anchors)), persistent=False)
        self.blocks = nn.ModuleList(Block(cfg, ROUTERS[kind](cfg)) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.num_classes)

    def forward(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed
        batch = x.shape[0]
        state = (self.anchor_init.expand(batch, -1, -1), self.start_positions.expand(batch, -1, -1).clone())
        for block in self.blocks:
            x, state = block(x, state)
        return self.head(self.norm(x).amax(dim=1))


class DenseBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_norm = nn.LayerNorm(cfg.embed_dim)
        self.attention = SelfAttention(cfg.embed_dim, cfg.heads)
        self.ffn_norm = nn.LayerNorm(cfg.embed_dim)
        hidden = int(cfg.embed_dim * cfg.mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(cfg.embed_dim, hidden), nn.GELU(), nn.Linear(hidden, cfg.embed_dim))

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class DenseViT(nn.Module):
    """Global attention in every block, native resolution (the 'vit_native' baseline)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.embed_dim
        self.patch_embed = nn.Conv2d(3, d, cfg.patch, cfg.patch)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.grid_h * cfg.grid_w, d) * 0.02)
        self.blocks = nn.ModuleList(DenseBlock(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.num_classes)

    def forward(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x).amax(dim=1))


def build(kind, cfg):
    return DenseViT(cfg) if kind == "vit" else AnchorViT(cfg, kind)
