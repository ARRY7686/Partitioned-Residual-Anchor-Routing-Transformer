"""Checks that impl="indexed" computes the same outputs and gradients as experiment 07's
reference implementation, for fixed and focus anchors (serial placement), on the full model.

Usage: python test_equivalence.py            (fp32, TF32 off; exits non-zero on failure)
"""
import copy
import sys

import torch

from models import Config, build

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOL = 1e-3   # fp32; the float64 run (--float64) is the strict check at 1e-9


def rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def compare(kind, grid_h, grid_w, anchors, batch=2, seed=0, autocast=False, chunk=4096, exact_st=True):
    torch.manual_seed(seed)
    cfg = Config(grid_h=grid_h, grid_w=grid_w, num_anchors=anchors, impl="reference", st_chunk=chunk,
                 exact_st=exact_st)
    ref = build(kind, cfg).to(DEVICE, torch.get_default_dtype())
    fast = copy.deepcopy(ref)
    for module in fast.modules():
        if hasattr(module, "cfg"):
            module.cfg = Config(**{**cfg.__dict__, "impl": "indexed"})
    images = torch.randn(batch, 3, grid_h * cfg.patch, grid_w * cfg.patch, device=DEVICE)
    target = torch.randn(batch, cfg.num_classes, device=DEVICE)

    results = []
    for model in (ref, fast):
        model.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=autocast):
            out = model(images)
        loss = ((out.float() - target) ** 2).mean()
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        results.append((out.detach().float(), grads))

    (out_r, g_r), (out_f, g_f) = results
    worst = ("output", rel_err(out_f, out_r))
    missing = set(g_r) ^ set(g_f)
    for name in g_r:
        if name in g_f:
            err = rel_err(g_f[name], g_r[name])
            if err > worst[1]:
                worst = (name, err)
    # The ST path is the only route for gradients into the focus importance head's positions;
    # report it separately so a silent zero there cannot hide.
    st = [n for n in g_r if "importance" in n]
    st_err = max((rel_err(g_f[n], g_r[n]) for n in st), default=0.0)
    return worst, st_err, missing


def bf16_accuracy(kind, grid=32, anchors=64, seed=0):
    """Under bf16 autocast, how far is each implementation's gradient from the fp32 gradient?
    The indexed version should be no less accurate than experiment 07's code."""
    torch.manual_seed(seed)
    cfg = Config(grid_h=grid, grid_w=grid, num_anchors=anchors, impl="reference", exact_st=True)
    base = build(kind, cfg).to(DEVICE)
    images = torch.randn(2, 3, grid * cfg.patch, grid * cfg.patch, device=DEVICE)
    target = torch.randn(2, cfg.num_classes, device=DEVICE)

    def grads(impl, autocast):
        model = copy.deepcopy(base)
        for module in model.modules():
            if hasattr(module, "cfg"):
                module.cfg = Config(**{**cfg.__dict__, "impl": impl})
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=autocast):
            out = model(images)
        ((out.float() - target) ** 2).mean().backward()
        return {n: p.grad.float() for n, p in model.named_parameters() if p.grad is not None}

    truth = grads("reference", False)
    errors = []
    for impl in ("reference", "indexed"):
        g = grads(impl, True)
        errors.append(torch.tensor([((g[n] - truth[n]).norm() / truth[n].norm().clamp_min(1e-12)).item()
                                    for n in truth]).median().item())
    return errors


def float64_mode():
    """Runs both implementations in float64: the models cast to float32 with .float() in a few
    places, so make .float() a no-op on float64 tensors. Any real difference between the
    implementations survives this; float32 rounding and summation-order effects do not."""
    original = torch.Tensor.float
    torch.Tensor.float = lambda self, *a, **k: self if self.dtype == torch.float64 else original(self, *a, **k)
    torch.set_default_dtype(torch.float64)


def main():
    if "--float64" in sys.argv:
        float64_mode()
        global TOL
        TOL = 1e-9
    cases = [
        ("fixed", 16, 24, 16, {}),
        ("focus", 16, 24, 16, {}),
        ("fixed", 32, 32, 64, {}),
        ("focus", 32, 32, 64, {}),
        ("focus", 24, 40, 64, {"chunk": 100}),    # rectangular grid, ST recomputed in many chunks
        ("focus", 16, 16, 64, {}),                # more anchors than fit: empty regions
    ]
    failed = False
    for kind, gh, gw, a, kw in cases:
        (name, err), st_err, missing = compare(kind, gh, gw, a, **kw)
        ok = err < TOL and st_err < TOL and not missing
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'}  {kind:5s} grid {gh}x{gw} A={a:3d} {kw}  "
              f"max rel err {err:.2e} ({name})  importance-head grad err {st_err:.2e}"
              + (f"  missing grads {sorted(missing)}" if missing else ""))
    (name, err), st_err, _ = compare("focus", 16, 16, 64, exact_st=False)
    print(f"INFO  focus 16x16 A=64 against experiment 07's original ST form: max rel err {err:.2e} "
          f"({name}) - the single-patch-region rounding artefact described in models.Config")
    if DEVICE.type == "cuda" and "--float64" not in sys.argv:
        for kind in ("fixed", "focus"):
            r, f = bf16_accuracy(kind)
            print(f"INFO  {kind} bf16 autocast, error against an fp32 run (median over parameters): "
                  f"reference {r:.2e}, indexed {f:.2e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
