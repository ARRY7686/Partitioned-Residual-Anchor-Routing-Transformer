"""Builds ImageNet-100 (or all of ImageNet-1k) as packed, pre-resized JPEG files from the
Hugging Face copy of ILSVRC-2012 (ILSVRC/imagenet-1k, parquet shards; licence accepted on the
user's own account, `huggingface-cli login` done by the user).

Output, in --out:
  {train,val,test}.bin        concatenated JPEG bytes
  {train,val,test}.offsets.npy int64, N+1 byte offsets into .bin
  {train,val,test}.labels.npy  int16 class index (0..C-1, in sorted-synset order of the subset)
Disk: the per-shard part files stay until the end, so allow ~2.5x the final size.
  {train,val,test}.files.txt   source file name of every image, in storage order
  meta.json                    classes, counts, settings, source dataset revision, checksums

Splits (PROTOCOL.md):
  train = ILSVRC-2012 train minus the held-out images
  val   = --val-per-class images per class held out from ILSVRC-2012 train, chosen by a fixed
          seed from the file names (so the split does not depend on shard order or machine)
  test  = the official ILSVRC-2012 validation set (50 per class); never used for any decision

Resizing: images whose short side is above --short are resized to short side --short
(Lanczos) and re-encoded at --quality; smaller images keep their original bytes. All models
see the same files, so this is a fixed part of the data, not a per-model choice.

usage (on the rented box):
  python prepare_data.py --classes imagenet100_classes.txt --out data/in100 --workers 24
  python prepare_data.py --classes all --out data/in1k --val-per-class 10 --workers 48
smoke test (two shards, a few images per class):
  python prepare_data.py --classes imagenet100_classes.txt --out data/smoke --shards 1 --val-per-class 1
"""
import argparse
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

REPO = "ILSVRC/imagenet-1k"
SPLIT_SEED = 20261125   # fixed once, before any run; changing it is a protocol deviation


def all_synsets():
    from huggingface_hub import hf_hub_download
    src = open(hf_hub_download(REPO, "classes.py", repo_type="dataset"), encoding="utf-8").read()
    synsets = re.findall(r'"(n\d{8})"', src)
    assert len(synsets) == 1000 and synsets == sorted(synsets)
    return synsets   # HF label i == synsets[i]


def process_shard(filename, keep_labels, short, quality, scratch, part):
    """Downloads one parquet shard, keeps the wanted classes, resizes, writes them to `part`
    (a pickle of (file name, HF label, JPEG bytes) records) and deletes the shard."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    local = hf_hub_download(REPO, filename, repo_type="dataset", local_dir=scratch)
    keep = set(keep_labels)
    records, resized = [], 0
    pf = pq.ParquetFile(local)
    for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
        labels = batch.column("label").to_pylist()
        images = batch.column("image")
        for row, label in enumerate(labels):
            if label not in keep:
                continue
            item = images[row].as_py()
            data, name = item["bytes"], os.path.basename(item["path"])
            img = Image.open(io.BytesIO(data))
            w, h = img.size
            if img.format != "JPEG" or img.mode != "RGB" or min(w, h) > short:
                img = img.convert("RGB")
                if min(w, h) > short:
                    s = short / min(w, h)
                    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
                    resized += 1
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                data = buf.getvalue()
            records.append((name, label, data))
    del pf
    os.remove(local)
    with open(part + ".tmp", "wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(part + ".tmp", part)
    return len(records), resized


def run_shards(files, keep_labels, args, parts_dir):
    """Processes shards in parallel, one part file per shard, so a rerun resumes. Returns the
    part files in shard order."""
    os.makedirs(parts_dir, exist_ok=True)
    parts = [os.path.join(parts_dir, os.path.basename(f) + ".pkl") for f in files]
    todo = [(f, p) for f, p in zip(files, parts) if not os.path.exists(p)]
    print(f"{len(files) - len(todo)}/{len(files)} shards already done", flush=True)
    scratch = os.path.join(args.out, "_download")
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(process_shard, f, keep_labels, args.short, args.quality, scratch, p): f
                   for f, p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            kept, resized = fut.result()
            rate = i / (time.time() - start) * 3600
            print(f"[{i}/{len(todo)}] {os.path.basename(futures[fut])}: kept {kept}, resized {resized}  "
                  f"({rate:.0f} shards/h)", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return parts


def load(part):
    with open(part, "rb") as fh:
        return pickle.load(fh)


class SplitWriter:
    """Appends records to {split}.bin; storage order = shard order, then row order (fixed by
    the source revision, so identical on every machine)."""

    def __init__(self, out, split, remap):
        self.out, self.split, self.remap = out, split, remap
        self.fh = open(os.path.join(out, f"{split}.bin"), "wb")
        self.offsets, self.labels, self.names = [0], [], []

    def add(self, name, label, data):
        self.fh.write(data)
        self.offsets.append(self.offsets[-1] + len(data))
        self.labels.append(self.remap[label])
        self.names.append(name)

    def close(self):
        self.fh.close()
        labels = np.array(self.labels, dtype=np.int16)
        np.save(os.path.join(self.out, f"{self.split}.offsets.npy"), np.array(self.offsets, dtype=np.int64))
        np.save(os.path.join(self.out, f"{self.split}.labels.npy"), labels)
        with open(os.path.join(self.out, f"{self.split}.files.txt"), "w") as fh:
            fh.write("\n".join(self.names) + "\n")
        sha = hashlib.sha256(labels.tobytes())
        sha.update("\n".join(self.names).encode())
        return {"images": len(labels), "bytes": self.offsets[-1], "files_labels_sha256": sha.hexdigest(),
                "per_class_min": int(np.bincount(labels, minlength=len(self.remap)).min())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classes", required=True, help="file of synset ids, one per line, or 'all'")
    p.add_argument("--out", required=True)
    p.add_argument("--short", type=int, default=288)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--val-per-class", type=int, default=50)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--shards", type=int, default=0, help="smoke test: only the first N shards per split")
    args = p.parse_args()

    from huggingface_hub import HfApi
    synsets = all_synsets()
    chosen = synsets if args.classes == "all" else sorted(l.strip() for l in open(args.classes) if l.strip())
    assert len(set(chosen)) == len(chosen) and set(chosen) <= set(synsets)
    keep_labels = [synsets.index(s) for s in chosen]
    remap = {hf: i for i, hf in enumerate(keep_labels)}
    info = HfApi().dataset_info(REPO)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
    train_files = [f for f in files if "/train-" in f]
    val_files = [f for f in files if "/validation-" in f]
    if args.shards:
        train_files, val_files = train_files[:args.shards], val_files[:args.shards]
    os.makedirs(args.out, exist_ok=True)

    test_parts = run_shards(val_files, keep_labels, args, os.path.join(args.out, "_parts", "validation"))
    train_parts = run_shards(train_files, keep_labels, args, os.path.join(args.out, "_parts", "train"))

    # Hold out val-per-class train images per class: a seeded permutation of each class's sorted
    # file names, so the choice depends only on SPLIT_SEED and the file names. Pass 1 reads names.
    by_class = {}
    for part in train_parts:
        for name, label, _ in load(part):
            by_class.setdefault(label, []).append(name)
    rng = np.random.default_rng(SPLIT_SEED)
    held_out = set()
    for label in sorted(by_class):
        names = sorted(by_class[label])
        k = min(args.val_per_class, max(0, len(names) - 1))
        held_out.update(names[i] for i in rng.permutation(len(names))[:k])

    meta = {"source": REPO, "source_revision": info.sha, "classes": chosen, "short_side": args.short,
            "jpeg_quality": args.quality, "val_per_class": args.val_per_class, "split_seed": SPLIT_SEED,
            "smoke_shards": args.shards, "splits": {}}
    # Pass 2: stream one part at a time, so memory stays at one shard's worth.
    writers = {s: SplitWriter(args.out, s, remap) for s in ("train", "val", "test")}
    for part in train_parts:
        for name, label, data in load(part):
            writers["val" if name in held_out else "train"].add(name, label, data)
    for part in test_parts:
        for record in load(part):
            writers["test"].add(*record)
    for split, w in writers.items():
        meta["splits"][split] = w.close()
        print(split, meta["splits"][split], flush=True)
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    shutil.rmtree(os.path.join(args.out, "_parts"))
    print("done:", args.out)


if __name__ == "__main__":
    main()
