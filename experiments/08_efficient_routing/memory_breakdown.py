"""Which activations does backward keep, and in which module? Counts the bytes of every tensor
saved for backward (each storage once), attributed to the innermost module running when it was
saved, for one block of a model at a given resolution.

Usage: python memory_breakdown.py fixed_indexed fullhd 2
"""
import sys
from collections import defaultdict

import torch

from benchmark import SIZES, VARIANTS
from models import Config, build


def main():
    variant, size, batch = sys.argv[1], sys.argv[2], int(sys.argv[3])
    kind, impl, placement, ckpt = VARIANTS[variant]
    height, width = SIZES[size]
    cfg = Config(grid_h=height // 16, grid_w=width // 16, impl=impl, placement=placement,
                 checkpoint_router=ckpt)
    model = build(kind, cfg).cuda()
    images = torch.randn(batch, 3, height, width, device="cuda")

    stack, names = ["<model>"], {m: n or "<model>" for n, m in model.named_modules()}
    for module in model.modules():
        module.register_forward_pre_hook(lambda m, a: stack.append(names[m]) and None)
        module.register_forward_hook(lambda m, a, o: stack.pop() and None)

    seen, by_module = set(), defaultdict(int)

    def pack(t):
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if t.is_cuda and key not in seen:
            seen.add(key)
            by_module[stack[-1]] += key[1]
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(images)

    # Report block 2 (a middle block) by sub-module, plus totals.
    total = sum(by_module.values())
    block = {k[len("blocks.2."):]: v for k, v in by_module.items() if k.startswith("blocks.2.")}
    print(f"{variant} {size} batch {batch}: saved for backward {total / 2**20:.0f} MB total, "
          f"block 2 {sum(block.values()) / 2**20:.0f} MB")
    for name, nbytes in sorted(block.items(), key=lambda kv: -kv[1]):
        if nbytes > 0.01 * sum(block.values()):
            print(f"  {name:40s} {nbytes / 2**20:7.1f} MB")
    other = {k: v for k, v in by_module.items() if not k.startswith("blocks.")}
    for name, nbytes in other.items():
        print(f"  [outside blocks] {name:23s} {nbytes / 2**20:7.1f} MB")


if __name__ == "__main__":
    main()
