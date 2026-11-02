"""Static checks of the DFG pipeline: label stats, box alignment on a real frame, mAP vs sklearn."""
import json, os, sys
import numpy as np
import torch, torchvision.io as io
from torchvision.utils import draw_bounding_boxes, save_image
root = "/root/exp07/data/dfg"
ix = json.load(open(os.path.join(root, "index.json")))
print("classes:", ix["class_names"])
for s in ("train", "val", "test"):
    e = ix[s]
    pos = np.array([[c in x["labels"] for c in range(20)] for x in e]).sum(0)
    ign = sum(len(x["ignore"]) for x in e)
    print(f"{s}: {len(e)} imgs, per-class positives min {pos.min()} median {int(np.median(pos))} max {pos.max()}, "
          f"labels/img {np.mean([len(x['labels']) for x in e]):.2f}, ignore-flags {ign}")
# box sizes of top-20 signs
sides = [max(b[2]-b[0], b[3]-b[1]) for x in ix["train"] for b in x["boxes"]]
print("top-20 sign size px: p10 %d median %d p90 %d" % tuple(np.percentile(sides, [10, 50, 90])))
# render two frames with boxes + label names
cands = [x for x in ix["train"] if len(x["boxes"]) >= 2][:2]
for k, x in enumerate(cands):
    img = io.decode_jpeg(io.read_file(os.path.join(root, "JPEGImages", x["file"])))
    boxes = torch.tensor([[b[1], b[0], b[3], b[2]] for b in x["boxes"]], dtype=torch.float)  # xyxy
    out = draw_bounding_boxes(img, boxes, colors="red", width=4)
    save_image(out.float() / 255, f"/root/exp07/dfg_check_{k}.jpg")
    print(x["file"], "labels", [ix["class_names"][c] for c in x["labels"]], "boxes", x["boxes"])
# AP implementation vs sklearn
sys.path.insert(0, "/root/exp07")
src = open("/root/exp07/train.py").read()
start = src.index("def average_precision"); end = src.index("def off_diagonal_mean")
ns = {"torch": torch, "np": np}; exec(src[start:end], ns)
from sklearn.metrics import average_precision_score
g = torch.Generator().manual_seed(0)
t = (torch.rand(500, 20, generator=g) < 0.05).float(); s = torch.randn(500, 20, generator=g) + 2 * t
v = torch.ones_like(t, dtype=torch.bool)
ours = ns["average_precision"](s, t, v)
ref = 100 * np.mean([average_precision_score(t[:, c], s[:, c]) for c in range(20) if t[:, c].sum() > 0])
print(f"mAP ours {ours:.3f} sklearn {ref:.3f}")
