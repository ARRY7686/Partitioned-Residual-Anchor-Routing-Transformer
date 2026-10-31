"""Rebuilds CIFAR-10 in the original python-pickle batch format from the Hugging Face mirror
uoft-cs/cifar10, for boxes where the original server is too slow.

The test set keeps the original order, so the fixed test canvases match experiments 04-05.
The training set holds the same 50,000 images in the mirror's order.

usage: python build_cifar_from_hf.py DATA_DIR   (writes DATA_DIR/cifar-10-batches-py/)
"""
import glob, io, os, pickle, sys
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

out = os.path.join(sys.argv[1], "cifar-10-batches-py")
os.makedirs(out, exist_ok=True)
src = snapshot_download("uoft-cs/cifar10", repo_type="dataset", allow_patterns="*.parquet")


def read(split):
    files = sorted(glob.glob(os.path.join(src, "**", f"{split}-*.parquet"), recursive=True))
    assert files, f"no {split} parquet under {src}"
    rows_img, rows_lab = [], []
    for f in files:
        t = pq.read_table(f).to_pydict()
        img_col = "img" if "img" in t else "image"
        rows_img += [np.asarray(Image.open(io.BytesIO(x["bytes"])).convert("RGB")) for x in t[img_col]]
        rows_lab += list(t["label"])
    data = np.stack(rows_img).transpose(0, 3, 1, 2).reshape(len(rows_img), -1)  # CHW, as in the original
    return data.astype(np.uint8), rows_lab


def dump(name, data, labels):
    with open(os.path.join(out, name), "wb") as f:
        pickle.dump({"data": data, "labels": labels, "batch_label": name}, f)


x, y = read("train")
assert len(y) == 50000, len(y)
for i in range(5):
    dump(f"data_batch_{i + 1}", x[i * 10000:(i + 1) * 10000], y[i * 10000:(i + 1) * 10000])
x, y = read("test")
assert len(y) == 10000, len(y)
dump("test_batch", x, y)
with open(os.path.join(out, "batches.meta"), "wb") as f:
    pickle.dump({"label_names": ["airplane", "automobile", "bird", "cat", "deer",
                                 "dog", "frog", "horse", "ship", "truck"]}, f)
print("wrote", out)
