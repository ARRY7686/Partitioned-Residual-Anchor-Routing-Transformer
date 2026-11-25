"""v2 backbone checks.

  python compare_v2.py test      # indexed AnchorMixer == dense reference (float64), fixed + focus
  python compare_v2.py table     # params, GFLOPs at 224, train-step time / memory vs PVTv2
"""
import copy
import json
import os
import subprocess
import sys
import time

import torch

from models_v2 import MixerConfig, build

VARIANTS = {
    # name: (timm name, mixer kwargs or None)
    "pvt_v2_b0": ("pvt_v2_b0", None),
    "anchor_b0_fixed": ("pvt_v2_b0", dict(kind="fixed")),
    "anchor_b0_focus": ("pvt_v2_b0", dict(kind="focus")),
    "anchor_b0_fixed_readall": ("pvt_v2_b0", dict(kind="fixed", read="all")),
    "anchor_b0_local": ("pvt_v2_b0", dict(kind="local")),
    "pvt_v2_b1": ("pvt_v2_b1", None),
    "anchor_b1_fixed": ("pvt_v2_b1", dict(kind="fixed")),
    "anchor_b1_focus": ("pvt_v2_b1", dict(kind="focus")),
    "anchor_b1_local": ("pvt_v2_b1", dict(kind="local")),
}


def make(variant, **kw):
    name, mixer = VARIANTS[variant]
    return build(name, MixerConfig(**{**mixer, **kw}) if mixer is not None else None)


def test():
    torch.set_default_dtype(torch.float64)
    original = torch.Tensor.float
    torch.Tensor.float = lambda self, *a, **k: self if self.dtype == torch.float64 else original(self, *a, **k)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ok = True
    for kind in ("fixed", "focus", "local", "local_slots"):
        sizes = ((128, 128), (96, 160), (256, 256), (112, 144)) if kind.startswith("local") else ((128, 128), (96, 160))
        for size in sizes:
            torch.manual_seed(0)
            extra = {"cell_st": False} if kind == "local_slots" else {}
            mk = "local" if kind == "local_slots" else kind
            ref = build("pvt_v2_b0", MixerConfig(kind=mk, impl="reference", **extra), num_classes=10).to(device)
            fast = copy.deepcopy(ref)
            for m in fast.modules():
                if hasattr(m, "cfg") and isinstance(m.cfg, MixerConfig):
                    m.cfg = MixerConfig(**{**m.cfg.__dict__, "impl": "indexed"})
            x = torch.randn(2, 3, *size, device=device)
            grads = []
            for model in (ref, fast):
                out = model(x)
                out.square().mean().backward()
                grads.append((out.detach(), {n: p.grad for n, p in model.named_parameters() if p.grad is not None}))
            (o_r, g_r), (o_f, g_f) = grads
            err = ((o_f - o_r).abs().max() / o_r.abs().max()).item()
            worst = max(((g_f[n] - g_r[n]).abs().max() / g_r[n].abs().max().clamp_min(1e-30)).item() for n in g_r)
            missing = set(g_r) ^ set(g_f)
            imp = [n for n in g_r if "importance" in n]
            passed = err < 1e-9 and worst < 1e-9 and not missing and (kind == "fixed" or bool(imp))
            ok &= passed
            print(f"{'PASS' if passed else 'FAIL'} {kind:5s} {size}: output err {err:.1e}, worst grad err {worst:.1e}"
                  f"{', importance grads present' if imp else ''}{', missing ' + str(missing) if missing else ''}")
    sys.exit(0 if ok else 1)


def measure(variant, height, width, batch):
    from torch.utils.flop_counter import FlopCounterMode
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = make(variant).to(device)
    result = {"params_m": sum(p.numel() for p in model.parameters()) / 1e6}
    if height == 224:
        with FlopCounterMode(display=False) as counter:
            model.eval()
            model(torch.randn(1, 3, 224, 224, device=device))
        result["gflops_224"] = counter.get_total_flops() / 2 / 1e9   # multiply-adds, as papers report
        model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    images = torch.randn(batch, 3, height, width, device=device)
    labels = torch.randint(0, 1000, (batch,), device=device)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(images).float(), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    try:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        result.update(step_ms=1000 * (time.perf_counter() - start) / 5,
                      peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
    except torch.OutOfMemoryError:
        result["oom"] = True
    return result


def table():
    settings = [(224, 224, 64), (1024, 1024, 2), (1152, 1920, 2)]
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)
    rows = []
    for h, w, b in settings:
        for v in variants:
            cmd = [sys.executable, os.path.abspath(__file__), "one", v, str(h), str(w), str(b)]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
            r = json.loads(line[7:]) if line else {"error": (proc.stderr.strip().splitlines() or ["?"])[-1]}
            rows.append({"variant": v, "h": h, "w": w, "batch": b, **r})
            extra = f"params {r['params_m']:.2f}M  GFLOPs@224 {r['gflops_224']:.2f}  " if "gflops_224" in r else ""
            body = "OOM" if r.get("oom") else (f"{r['step_ms']:7.1f} ms {r['peak_mb']:7.0f} MB" if "step_ms" in r else f"ERROR {r.get('error')}")
            print(f"{h}x{w} b{b:<3d} {v:26s} {extra}{body}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "compare_v2_rtx5060.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "test":
        test()
    elif sys.argv[1] == "table":
        table()
    elif sys.argv[1] == "one":
        _, _, v, h, w, b = sys.argv
        print("RESULT " + json.dumps(measure(v, int(h), int(w), int(b))), flush=True)
