"""v2 backbone: PVTv2 with its spatial-reduction attention replaced by exclusive-region anchors.

PVTv2 (timm `pvt_v2_b0` / `pvt_v2_b1`) is a 4-stage pyramid: overlapping conv patch embedding,
blocks of [LayerNorm -> global mixer] + [LayerNorm -> FFN with a 3x3 depthwise conv], and a
global mixer called spatial-reduction attention (SRA): a strided conv pools each sr x sr block of
patches into one key/value token, and every patch attends to all of them. SRA is itself a fixed
exclusive partition with a learned linear pooling.

`AnchorMixer` replaces SRA in stages 1-3 and keeps everything else identical (stage 4, where the
map is 7x7 at 224 px, keeps PVTv2's full attention, as PVTv2 itself does). Per block:
  1. anchors start from a learned seed plus an MLP of their 2-D position;
  2. gather: each anchor pools *only its own Voronoi region*, with a softmax of its query
     against the patch keys inside the region (experiment 07's routing, indexed implementation);
  3. the anchors attend to each other (the only long-range step);
  4. scatter: each patch reads its own anchor through a sigmoid gate (read="own", experiment
     07), or attends to all anchors (read="all", like SRA's read side);
  5. an output projection.
kind="fixed": anchors on a regular grid. kind="focus": a quarter of the anchors on a coarse
grid, the rest placed at the soft-argmax of a learned importance map (parallel placement, with a
home prior), with the straight-through gradient of experiment 07.

The anchor grid is `anchor_grid` (rows, cols) per stage, or, with `region`, one anchor per
region x region patches (so the anchor count grows with the input, keeping regions the same size).
"""
import math
from dataclasses import dataclass, field

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.pvt_v2 import Attention as PVTAttention

from models import _GatherST, _ScatterST


@dataclass
class MixerConfig:
    kind: str = "fixed"            # fixed | focus (global placement, v1) | local (local focus)
    read: str = "own"              # own | all
    anchor_grid: tuple = (7, 7)    # per block, if region is None
    region: int | None = None      # patches per anchor side (overrides anchor_grid)
    home_width: float = 0.35       # focus: home prior width, in long-side units
    radius: int = 1                # local: a focus anchor searches its home cell +- radius cells
    home_sigma_cells: float = 1.5  # local: home prior width, in cells
    impl: str = "indexed"          # indexed | reference (dense, for the test)
    st_chunk: int = 4096
    cell_st: bool = True           # local: per-cell batched matmuls for the ST terms when possible


# ---------------------------------------------------------------- local straight-through terms
# When the patch grid splits exactly into the anchor cells (the usual case: 56/8, 28/4, 14/2 at
# 224 px; 480/8 x 288/8 at full HD), all patches of a cell share the same candidate anchors, so
# the per-(patch, candidate) products become one batched matmul per cell: (K x dh) @ (dh x P).
def _to_cells(t, blocks):
    """(B, H, N, d) -> (B, H, cells, P, d), patches grouped by cell."""
    ah, aw, ph, pw = blocks[:4]
    b, h, _, d = t.shape
    return t.view(b, h, ah, ph, aw, pw, d).permute(0, 1, 2, 4, 3, 5, 6).reshape(b, h, ah * aw, ph * pw, d)


def _from_cells(t, blocks):
    """(B, cells, K, P) -> (B, N, K)."""
    ah, aw, ph, pw = blocks[:4]
    b, _, k, _ = t.shape
    return t.view(b, ah, aw, k, ph, pw).permute(0, 1, 4, 2, 5, 3).reshape(b, ah * ph * aw * pw, k)


def _cell_gather_st(cand, valid, blocks, q, k, v, log_importance, peak, denom, pooled, grad_pooled):
    cand_cell, valid_cell = blocks[4], blocks[5]                      # (cells, K)
    scale = q.shape[-1] ** -0.5
    g = grad_pooled.float()
    gd, cd = g / denom[..., None], (g * pooled).sum(-1) / denom       # (B,H,A,dh), (B,H,A)
    kc, vc = _to_cells(k.float(), blocks), _to_cells(v.float(), blocks)          # (B,H,C,P,dh)
    logits = q.float()[:, :, cand_cell] @ kc.transpose(-2, -1) * scale         # (B,H,C,K,P)
    if log_importance is not None:
        li = log_importance[:, None, :, None].expand(-1, 1, -1, 1)
        logits = logits + _to_cells(li, blocks).transpose(-2, -1)
    e = (logits - peak[:, :, cand_cell][..., None]).clamp_max(0).exp()
    p = gd[:, :, cand_cell] @ vc.transpose(-2, -1) - cd[:, :, cand_cell][..., None]
    out = (e * p).sum(1) * valid_cell[None, :, :, None]                         # (B,C,K,P)
    return _from_cells(out, blocks)


def _cell_scatter_st(cand, valid, blocks, q, k, v, grad_read):
    cand_cell, valid_cell = blocks[4], blocks[5]
    scale = q.shape[-1] ** -0.5
    qc, gc = _to_cells(q.float(), blocks), _to_cells(grad_read.float(), blocks)  # (B,H,C,P,dh)
    gate = torch.sigmoid(k.float()[:, :, cand_cell] @ qc.transpose(-2, -1) * scale)   # (B,H,C,K,P)
    u = v.float()[:, :, cand_cell] @ gc.transpose(-2, -1)
    out = (gate * u).sum(1) * valid_cell[None, :, :, None]
    return _from_cells(out, blocks)



# As models._GatherST / _ScatterST, but the soft assignment of patch n is over its candidate
# anchors cand[n, :] only (B, N, K), so the recomputed gradient costs O(N K), not O(N A).
class _GatherSTLocal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft, cand, valid, blocks, q, k, v, log_importance, peak, denom, pooled, chunk):
        ctx.save_for_backward(cand, valid, q, k, v, log_importance, peak, denom, pooled)
        ctx.chunk, ctx.blocks = chunk, blocks
        return torch.zeros_like(pooled)

    @staticmethod
    def backward(ctx, grad_pooled):
        cand, valid, q, k, v, log_importance, peak, denom, pooled = ctx.saved_tensors
        if ctx.blocks is not None:
            out = _cell_gather_st(cand, valid, ctx.blocks, q, k, v, log_importance, peak, denom, pooled, grad_pooled)
            return (out,) + (None,) * 11
        scale = q.shape[-1] ** -0.5
        g = grad_pooled.float()                                       # (B, H, A, dh)
        gd = g / denom[..., None]
        cd = (g * pooled).sum(-1) / denom                             # (B, H, A)
        kf, vf = k.float(), v.float()                                 # (B, H, N, dh)
        qf = q.float()
        out = torch.zeros(cand.shape[0], cand.shape[1], k.shape[0], device=k.device, dtype=g.dtype)
        # One candidate slot at a time: every temporary is (B, H, N, dh) at most.
        for slot in range(cand.shape[1]):
            a = cand[:, slot]                                         # (N,)
            logits = (qf[:, :, a] * kf).sum(-1) * scale               # (B, H, N)
            if log_importance is not None:
                logits = logits + log_importance[:, None, :]
            e = (logits - peak[:, :, a]).clamp_max(0).exp()
            p = (gd[:, :, a] * vf).sum(-1) - cd[:, :, a]
            out[:, slot] = ((e * p).sum(1) * valid[:, slot]).transpose(0, 1)
        return (out.permute(2, 0, 1),) + (None,) * 11


class _ScatterSTLocal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft, cand, valid, blocks, q, k, v, out_like, chunk):
        ctx.save_for_backward(cand, valid, q, k, v)
        ctx.chunk, ctx.blocks = chunk, blocks
        return torch.zeros_like(out_like)

    @staticmethod
    def backward(ctx, grad_read):
        cand, valid, q, k, v = ctx.saved_tensors
        if ctx.blocks is not None:
            return (_cell_scatter_st(cand, valid, ctx.blocks, q, k, v, grad_read),) + (None,) * 8
        scale = q.shape[-1] ** -0.5
        kf, vf = k.float(), v.float()                                 # (B, H, A, dh)
        qf, gr = q.float(), grad_read.float()                         # (B, H, N, dh)
        out = torch.zeros(cand.shape[0], cand.shape[1], q.shape[0], device=q.device, dtype=qf.dtype)
        for slot in range(cand.shape[1]):
            a = cand[:, slot]
            gate = torch.sigmoid((qf * kf[:, :, a]).sum(-1) * scale)  # (B, H, N)
            u = (gr * vf[:, :, a]).sum(-1)
            out[:, slot] = ((gate * u).sum(1) * valid[:, slot]).transpose(0, 1)
        return (out.permute(2, 0, 1),) + (None,) * 8


def _heads(t, h):
    b, n, c = t.shape
    return t.view(b, n, h, c // h).transpose(1, 2)


def _merge(t):
    b, h, n, d = t.shape
    return t.transpose(1, 2).reshape(b, n, h * d)


class AnchorMixer(nn.Module):
    def __init__(self, dim, num_heads, cfg: MixerConfig):
        super().__init__()
        self.cfg, self.heads, self.dim = cfg, num_heads, dim
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)              # patch queries for the scatter step
        self.kv = nn.Linear(dim, 2 * dim)         # patch keys / values for the gather step
        self.anchor_seed = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embed = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor_q = nn.Linear(dim, dim)
        self.anchor_norm = nn.LayerNorm(dim)
        self.anchor_qkv = nn.Linear(dim, 3 * dim)
        self.anchor_proj = nn.Linear(dim, dim)
        self.scatter_norm = nn.LayerNorm(dim)
        self.anchor_kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        if cfg.kind in ("focus", "local"):
            self.importance = nn.Linear(dim, 1)
        self._geometry = {}

    # ---------------------------------------------------------------- geometry (cached per size)
    def geometry(self, h, w, device):
        key = (h, w, str(device), torch.get_default_dtype())
        if key in self._geometry:
            return self._geometry[key]
        g = max(h, w)
        rows = (torch.arange(h, device=device, dtype=torch.get_default_dtype()) + 0.5) / g
        cols = (torch.arange(w, device=device, dtype=torch.get_default_dtype()) + 0.5) / g
        r, c = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([r, c], -1).reshape(-1, 2)                       # (N, 2)
        if self.cfg.region:
            ah, aw = math.ceil(h / self.cfg.region), math.ceil(w / self.cfg.region)
        else:
            ah, aw = self.cfg.anchor_grid

        def lattice(nr, nc):
            rr = (torch.arange(nr, device=device, dtype=torch.get_default_dtype()) + 0.5) / nr * h / g
            cc = (torch.arange(nc, device=device, dtype=torch.get_default_dtype()) + 0.5) / nc * w / g
            a, b = torch.meshgrid(rr, cc, indexing="ij")
            return torch.stack([a, b], -1).reshape(-1, 2)

        grid = lattice(ah, aw)                                                 # (A, 2)
        spacing = max(h / (g * ah), w / (g * aw))
        geo = {"coords": coords, "grid": grid, "tau": (0.5 * spacing) ** 2}
        # On a product lattice the nearest anchor is (nearest row, nearest column), so fixed
        # anchors' regions need only h x ah + w x aw distances, not A x N (same argmin tie-break).
        row_near = ((grid.view(ah, aw, 2)[:, 0, 0][None] - rows[:, None]).square()).argmin(1)   # (h,)
        col_near = ((grid.view(ah, aw, 2)[0, :, 1][None] - cols[:, None]).square()).argmin(1)   # (w,)
        geo["owner"] = (row_near[:, None] * aw + col_near[None, :]).reshape(-1)                 # (N,)
        if self.cfg.kind == "focus":
            coverage = lattice(max(1, ah // 2), max(1, aw // 2))
            num_focus = ah * aw - coverage.shape[0]
            spread = torch.linspace(0, ah * aw - 1, num_focus, device=device).round().long()
            homes = grid[spread]
            home_distance = (coords[None] - homes[:, None]).square().sum(-1)
            geo.update(coverage=coverage, home_prior=-home_distance / (2 * self.cfg.home_width ** 2))
        if self.cfg.kind == "local":
            geo.update(self._local_geometry(h, w, ah, aw, g, coords, grid, row_near, col_near, device))
        self._geometry[key] = geo
        return geo

    def _local_geometry(self, h, w, ah, aw, g, coords, grid, row_near, col_near, device):
        """Local focus: anchors at even (row, col) lattice positions stay fixed (coverage, about
        a quarter); every other anchor searches the patches of its home cell +- radius cells.
        Each patch's candidate owners are the anchors homed within +- (radius + 1) cells of the
        cell it falls in, in increasing anchor order."""
        R = self.cfg.radius
        ch, cw = h / ah, w / aw
        ar = torch.arange(ah, device=device).repeat_interleave(aw)
        ac = torch.arange(aw, device=device).repeat(ah)
        focus_ids = (~((ar % 2 == 0) & (ac % 2 == 0))).nonzero().squeeze(1)
        fr, fc = ar[focus_ids], ac[focus_ids]
        r0 = torch.floor((fr - R) * ch).clamp_min(0).long()
        r1 = torch.ceil((fr + R + 1) * ch).clamp_max(h).long()
        c0 = torch.floor((fc - R) * cw).clamp_min(0).long()
        c1 = torch.ceil((fc + R + 1) * cw).clamp_max(w).long()
        kr, kc = int((r1 - r0).max()), int((c1 - c0).max())
        rr = r0[:, None] + torch.arange(kr, device=device)[None]
        cc = c0[:, None] + torch.arange(kc, device=device)[None]
        window = (rr.clamp_max(h - 1)[:, :, None] * w + cc.clamp_max(w - 1)[:, None, :]).reshape(len(focus_ids), -1)
        window_valid = ((rr < r1[:, None])[:, :, None] & (cc < c1[:, None])[:, None, :]).reshape(len(focus_ids), -1)
        sigma = self.cfg.home_sigma_cells * max(ch, cw) / g
        prior = -(coords[window] - grid[focus_ids][:, None]).square().sum(-1) / (2 * sigma ** 2)
        prior = prior.masked_fill(~window_valid, float("-inf"))

        off = torch.arange(-(R + 1), R + 2, device=device)
        cr, cn = row_near[:, None] + off[None], col_near[:, None] + off[None]            # (h, K1), (w, K1)
        cand = cr.clamp(0, ah - 1)[:, None, :, None] * aw + cn.clamp(0, aw - 1)[None, :, None, :]
        cand_valid = ((cr >= 0) & (cr < ah))[:, None, :, None] & ((cn >= 0) & (cn < aw))[None, :, None, :]
        blocks = None
        if self.cfg.cell_st and h % ah == 0 and w % aw == 0:
            ph, pw = h // ah, w // aw
            # every patch of a cell has the same nearest lattice row/column, hence the same candidates
            assert bool((row_near == torch.arange(h, device=device) // ph).all()) and \
                bool((col_near == torch.arange(w, device=device) // pw).all())
            blocks = (ah, aw, ph, pw, cand[::ph, ::pw].reshape(ah * aw, -1), cand_valid[::ph, ::pw].reshape(ah * aw, -1))
        return {"focus_ids": focus_ids, "window": window, "window_prior": prior, "blocks": blocks,
                "cand": cand.reshape(h * w, -1), "cand_valid": cand_valid.reshape(h * w, -1)}

    def place(self, x, geo):
        batch = x.shape[0]
        if self.cfg.kind == "fixed":
            return geo["grid"].expand(batch, -1, -1), None
        scores = self.importance(x).squeeze(-1).float()
        importance = F.softplus(scores) + 1e-3
        if self.cfg.kind == "local":
            weights = (scores[:, geo["window"]] + geo["window_prior"][None]).softmax(-1)      # (B, F, K)
            peaks = (weights[..., None] * geo["coords"][geo["window"]][None]).sum(-2)         # (B, F, 2)
            positions = geo["grid"].expand(batch, -1, -1).index_copy(1, geo["focus_ids"], peaks.to(geo["grid"].dtype))
            return positions, importance
        peaks = (scores[:, None, :] + geo["home_prior"][None]).softmax(-1) @ geo["coords"]
        return torch.cat([geo["coverage"].expand(batch, -1, -1), peaks], 1), importance

    # ---------------------------------------------------------------- forward
    def forward(self, x, feat_size):
        batch, n, c = x.shape
        h, w = feat_size
        geo = self.geometry(h, w, x.device)
        positions, importance = self.place(x, geo)
        anchors = self.anchor_seed + self.position_embed(positions).to(x.dtype)
        k, v = (_heads(t, self.heads) for t in self.kv(x).chunk(2, -1))
        q_anchor = _heads(self.anchor_q(anchors), self.heads)
        route = self._gather_reference if self.cfg.impl == "reference" else self._gather_indexed
        pooled, owner, soft, local = route(q_anchor, k, v, positions, importance, geo)
        anchors = anchors + _merge(pooled.to(anchors.dtype))
        qkv = self.anchor_qkv(self.anchor_norm(anchors))
        aq, ak, av = (_heads(t, self.heads) for t in qkv.chunk(3, -1))
        anchors = anchors + self.anchor_proj(_merge(F.scaled_dot_product_attention(aq, ak, av)))

        k2, v2 = (_heads(t, self.heads) for t in self.anchor_kv(self.scatter_norm(anchors)).chunk(2, -1))
        q2 = _heads(self.q(x), self.heads)
        if self.cfg.read == "all":
            read = F.scaled_dot_product_attention(q2, k2, v2)
        elif self.cfg.impl == "reference":
            hard = F.one_hot(owner, k2.shape[2]).to(q2.dtype)                    # (B, N, A)
            assign = hard + (soft.transpose(1, 2) - soft.transpose(1, 2).detach()) if soft is not None else hard
            gate = torch.sigmoid((q2 @ k2.transpose(-2, -1)).float() * self.scale)
            read = (gate * assign[:, None]).to(v2.dtype) @ v2
        else:
            dh = k2.shape[-1]
            index = owner[:, None, :, None].expand(-1, self.heads, -1, dh)
            k2_own, v2_own = k2.gather(2, index), v2.gather(2, index)
            acc = torch.promote_types(k2.dtype, torch.float32)
            gate = torch.sigmoid((q2 * k2_own).sum(-1, dtype=acc) * self.scale)
            read = gate[..., None].to(v2.dtype) * v2_own
            if soft is not None and local is not None:
                read = read + _ScatterSTLocal.apply(soft, *local, q2.detach(), k2.detach(), v2.detach(), read,
                                                    self.cfg.st_chunk)
            elif soft is not None:
                read = read + _ScatterST.apply(soft, q2.detach(), k2.detach(), v2.detach(), read, self.cfg.st_chunk)
        return self.proj(_merge(read))

    def _assignment(self, positions, geo, dense):
        """Returns (owner (B, N), soft assignment or None, local candidates or None). The soft
        assignment is (B, A, N) when dense, and (B, N, K) over each patch's candidates otherwise."""
        moving = positions.requires_grad and torch.is_grad_enabled()
        batch = positions.shape[0]
        if self.cfg.kind == "fixed" and not moving:
            return geo["owner"].expand(batch, -1), None, None
        if self.cfg.kind == "local" and not dense:
            cand, valid = geo["cand"], geo["cand_valid"]
            distance = (geo["coords"][None, :, None, :] - positions[:, cand]).square().sum(-1)     # (B, N, K)
            distance = distance.masked_fill(~valid, float("inf"))
            best = distance.detach().argmin(-1, keepdim=True)
            owner = cand[None].expand(batch, -1, -1).gather(2, best).squeeze(-1)
            soft = (-distance / geo["tau"]).softmax(-1) if moving else None
            return owner, soft, (cand, valid, geo["blocks"])
        distance = (positions[:, :, None, :] - geo["coords"][None, None]).square().sum(-1)   # (B, A, N)
        if self.cfg.kind == "local":            # dense reference with the same candidate sets
            if "dense_mask" not in geo:
                n_idx = torch.arange(geo["cand"].shape[0], device=distance.device)[:, None].expand_as(geo["cand"])
                mask = torch.zeros(distance.shape[1:], dtype=torch.bool, device=distance.device)
                mask[geo["cand"][geo["cand_valid"]], n_idx[geo["cand_valid"]]] = True
                geo["dense_mask"] = mask
            distance = distance.masked_fill(~geo["dense_mask"][None], float("inf"))
        soft = (-distance / geo["tau"]).softmax(1) if moving else None
        owner = distance.detach().argmin(1)                                                  # (B, N)
        return owner, soft, None

    def _gather_indexed(self, q, k, v, positions, importance, geo):
        batch, heads, n, dh = k.shape
        a = q.shape[2]
        owner, soft, local = self._assignment(positions, geo, dense=False)
        acc = torch.promote_types(k.dtype, torch.float32)
        offsets = (torch.arange(batch * heads, device=k.device) * a).view(batch, heads, 1)
        segment = (owner[:, None, :] + offsets).reshape(-1)
        q_own = q.gather(2, owner[:, None, :, None].expand(-1, heads, -1, dh))
        logits = (q_own * k).sum(-1, dtype=acc) * self.scale
        log_imp = importance.log() if importance is not None else None
        if log_imp is not None:
            logits = logits + log_imp[:, None, :]
        peak = torch.full((batch * heads * a,), float("-inf"), device=k.device, dtype=acc)
        peak = torch.nan_to_num(peak.scatter_reduce(0, segment, logits.detach().reshape(-1), "amax"), neginf=0.0)
        e = (logits - peak[segment].view_as(logits)).clamp_max(0).exp()
        denom = torch.zeros(batch * heads * a, device=k.device, dtype=acc).index_add(0, segment, e.reshape(-1)).clamp_min(1.0)
        wgt = e / denom[segment].view_as(e)
        pooled = torch.zeros(batch * heads * a, dh, device=k.device, dtype=acc)
        pooled = pooled.index_add(0, segment, (wgt.to(v.dtype)[..., None] * v).to(acc).reshape(-1, dh))
        pooled = pooled.view(batch, heads, a, dh)
        if soft is not None:
            st_args = (q.detach(), k.detach(), v.detach(), None if log_imp is None else log_imp.detach(),
                       peak.view(batch, heads, a), denom.view(batch, heads, a).detach(), pooled.detach(),
                       self.cfg.st_chunk)
            if local is not None:
                pooled = pooled + _GatherSTLocal.apply(soft, *local, *st_args)
            else:
                pooled = pooled + _GatherST.apply(soft, *st_args)
        return pooled, owner, soft, local

    def _gather_reference(self, q, k, v, positions, importance, geo):
        owner, soft, _ = self._assignment(positions, geo, dense=True)
        hard = F.one_hot(owner, q.shape[2]).transpose(1, 2).to(k.dtype)              # (B, A, N)
        assign = hard + (soft - soft.detach()) if soft is not None else hard
        logits = (q @ k.transpose(-2, -1)).float() * self.scale
        if importance is not None:
            logits = logits + importance.log()[:, None, None, :]
        peak = logits.masked_fill(~(hard > 0.5)[:, None], float("-inf")).amax(-1, keepdim=True)
        peak = torch.nan_to_num(peak.detach(), neginf=0.0)
        scores = (logits - peak).clamp_max(0).exp() * assign[:, None]
        weights = scores / scores.sum(-1, keepdim=True).clamp_min(1.0)
        return weights.to(v.dtype) @ v, owner, soft, None


def build(name, mixer: MixerConfig | None = None, num_classes=1000, anchor_stages=(0, 1, 2),
          match_sra=True, **timm_kwargs):
    """`name`: pvt_v2_b0 / pvt_v2_b1 etc. With `mixer`, SRA in `anchor_stages` is replaced by
    AnchorMixer; without it, this is timm's PVTv2 unchanged. With match_sra, each stage gets one
    anchor per sr x sr patches (PVTv2's reduction ratio, 8/4/2), i.e. exactly as many anchors as
    SRA has pooled key/value tokens, at any input size (7x7 at 224 px)."""
    import dataclasses
    model = timm.create_model(name, num_classes=num_classes, **timm_kwargs)
    if mixer is not None:
        for s in anchor_stages:
            for block in model.stages[s].blocks:
                old = block.attn
                assert isinstance(old, PVTAttention)
                cfg = mixer
                if match_sra and old.sr is not None:
                    cfg = dataclasses.replace(mixer, region=old.sr.kernel_size[0])
                block.attn = AnchorMixer(old.dim, old.num_heads, cfg)
    return model
