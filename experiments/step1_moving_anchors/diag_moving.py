"""Diagnose a trained positional-anchor model from step1-moving-anchors.ipynb.

Reports, per block, how strongly the learned importance map points at the object and how
far the anchors move from their starting grid. Loads the model definitions from the
notebook next to this file and a checkpoint saved by it.

    python diag_moving.py [kind] [num_anchors]      # e.g. python diag_moving.py moving 4

Uses the same environment variables as the notebook (RART_DATASET, RART_WORK_DIR); the
checkpoint is read from RART_WORK_DIR.
"""
import json
import os
import sys

os.environ.setdefault("RART_SMOKE", "1")
os.environ.setdefault("RART_DATASET", "cifar10")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "step1-moving-anchors.ipynb"), encoding="utf-8") as f:
    cells = json.load(f)["cells"]
code = []
for cell in cells:
    if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("## Metrics"):
        break
    if cell["cell_type"] == "code":
        code.append("".join(cell["source"]))
exec("\n\n".join(code))

kind = sys.argv[1] if len(sys.argv) > 1 else "moving"
num_anchors = int(sys.argv[2]) if len(sys.argv) > 2 else 4
path = os.path.join(WORK_DIR, f"step1_{DATASET}_seed{SEED}_{kind}_A{num_anchors}.pth")
model = AnchorViT(kind, num_anchors).to(DEVICE)
model.load_state_dict(torch.load(path, map_location=DEVICE))
model.eval()

captured = []


def hook(module, inputs, output):
    captured.append(F.softplus(output.squeeze(-1).float()) + 1e-3)


for block in model.blocks:
    block.router.importance.register_forward_hook(hook)
    block.router.record = True

x, y, boxes = next(iterate_batches(test_x, test_y, train=False))
with torch.no_grad(), torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
    model(x)
positions = torch.stack([router.last[2] for router in model.routers()], dim=1)
on_object = object_patch_mask(boxes).float()
start = model.start_positions[None]

print(f"{kind} A={num_anchors}: importance on object vs background, and anchor displacement")
for block, importance in enumerate(captured):
    share = (importance * on_object).sum(1) / importance.sum(1)
    density_object = (importance * on_object).sum(1) / on_object.sum(1)
    density_background = (importance * (1 - on_object)).sum(1) / (1 - on_object).sum(1)
    top_hit = on_object.gather(1, importance.topk(16, dim=1).indices).mean().item()
    moved = (positions[:, block] - start).norm(dim=-1) * GRID
    print(f"block {block + 1}: importance share on object {share.mean():.3f} "
          f"(area share {OBJECT_SIZE ** 2 / CANVAS ** 2:.4f}) | "
          f"object/background ratio {(density_object / density_background).mean():.2f} | "
          f"top-16 patches on object {top_hit:.2f} | "
          f"anchor displacement {moved.mean():.2f} patches (max {moved.max():.2f})")
