"""Builds the extra datasets from Hugging Face mirrors in python-pickle form.

  cifar100      uoft-cs/cifar100      -> DATA_DIR/cifar-100-python/{train,test}  (experiment 03, seed 44)
  tinyimagenet  zh-plus/tiny-imagenet -> DATA_DIR/tiny-imagenet/{train,test}    (native-detail canvas)

Tiny ImageNet's labelled validation split (10,000 images) is used as the test set.

usage: python build_extra_data.py DATASET DATA_DIR
"""
import glob, io, os, pickle, sys
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

dataset, data_dir = sys.argv[1], sys.argv[2]
repo, folder, splits, label_col, label_key = {
    "cifar100": ("uoft-cs/cifar100", "cifar-100-python", {"train": "train", "test": "test"}, "fine_label", "fine_labels"),
    "tinyimagenet": ("zh-plus/tiny-imagenet", "tiny-imagenet", {"train": "train", "test": "valid"}, "label", "labels"),
}[dataset]
out = os.path.join(data_dir, folder)
os.makedirs(out, exist_ok=True)
src = snapshot_download(repo, repo_type="dataset", allow_patterns="*.parquet")

for name, split in splits.items():
    files = sorted(glob.glob(os.path.join(src, "**", f"{split}-*.parquet"), recursive=True))
    assert files, f"no {split} parquet under {src}"
    images, labels = [], []
    for f in files:
        t = pq.read_table(f).to_pydict()
        img_col = "img" if "img" in t else "image"
        images += [np.asarray(Image.open(io.BytesIO(x["bytes"])).convert("RGB")) for x in t[img_col]]
        labels += list(t[label_col])
    data = np.stack(images).transpose(0, 3, 1, 2).reshape(len(images), -1).astype(np.uint8)  # CHW
    with open(os.path.join(out, name), "wb") as fh:
        pickle.dump({"data": data, label_key: labels}, fh)
    print(dataset, name, data.shape, "classes", len(set(labels)))
