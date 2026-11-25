"""Training-step time and peak memory: reference vs indexed anchor routing vs dense attention.

Each (model, resolution, batch) runs in its own process so peak-memory readings are clean.
One training step = forward + backward + AdamW update, bf16 autocast, untrained weights.

Usage:
  python benchmark.py                              # default sweep, writes results/benchmark_<gpu>.json
  python benchmark.py --sizes 512 --batch 4        # quick check
  python benchmark.py --one fixed_indexed 512x512 4    (internal: one measurement, prints JSON)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

from models import Config, build

VARIANTS = {
    # name: (kind, impl, placement, checkpoint_router)
    "fixed_reference": ("fixed", "reference", "serial", False),
    "fixed_indexed": ("fixed", "indexed", "serial", False),
    "fixed_indexed_ckpt": ("fixed", "indexed", "serial", True),
    "focus_reference": ("focus", "reference", "serial", False),
    "focus_indexed": ("focus", "indexed", "serial", False),
    "focus_indexed_parallel": ("focus", "indexed", "parallel", False),
    "focus_indexed_parallel_ckpt": ("focus", "indexed", "parallel", True),
    "densewin": ("densewin", "indexed", "serial", False),
    "vit": ("vit", "indexed", "serial", False),
}
SIZES = {"512": (512, 512), "1024": (1024, 1024), "1536": (1536, 1536), "fullhd": (1152, 1920),
         "2048": (2048, 2048)}


def measure(variant, height, width, batch, steps=5, warmup=3):
    kind, impl, placement, ckpt = VARIANTS[variant]
    cfg = Config(grid_h=height // 16, grid_w=width // 16, impl=impl, placement=placement,
                 checkpoint_router=ckpt)
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = build(kind, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    images = torch.randn(batch, 3, height, width, device=device)
    target = torch.randn(batch, cfg.num_classes, device=device)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), target.sigmoid())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    return {"step_ms": 1000 * (time.perf_counter() - start) / steps,
            "peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6}


def run_one(variant, size, batch):
    height, width = (int(s) for s in size.split("x"))
    try:
        result = measure(variant, height, width, int(batch))
    except torch.OutOfMemoryError:
        result = {"oom": True}
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", nargs=3)
    parser.add_argument("--sizes", default="512,1024,1536,fullhd,2048")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.one:
        run_one(*args.one)
        return

    gpu = torch.cuda.get_device_name(0)
    rows = []
    for size in args.sizes.split(","):
        height, width = SIZES[size]
        for variant in args.variants.split(","):
            cmd = [sys.executable, os.path.abspath(__file__), "--one", variant, f"{height}x{width}", str(args.batch)]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
            result = json.loads(line[7:]) if line else {"error": proc.stderr.strip().splitlines()[-1:]}
            row = {"variant": variant, "size": size, "height": height, "width": width, "batch": args.batch, **result}
            rows.append(row)
            shown = "OOM" if result.get("oom") else (f"{result['step_ms']:8.1f} ms  {result['peak_mb']:8.0f} MB"
                                                    if "step_ms" in result else f"ERROR {result.get('error')}")
            print(f"{size:>7s} b{args.batch} {variant:24s} {shown}", flush=True)

    os.makedirs("results", exist_ok=True)
    out = args.out or os.path.join("results", f"benchmark_{gpu.replace(' ', '_')}_b{args.batch}.json")
    with open(out, "w") as f:
        json.dump({"gpu": gpu, "torch": torch.__version__, "rows": rows}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
