"""Downloads and indexes the experiment 07 datasets into DATA_DIR (default ./data).

    python build_data.py mnist                 # MNIST for Megapixel MNIST (11 MB)
    python build_data.py dfg                   # DFG traffic signs: 7.5 GB of images + annotations
    python build_data.py dfg --limit-mb 200    # only the first ~200 MB of images, for a smoke test

DFG (Tabernik & Skočaj, 2019; CC BY-NC-SA 4.0) has 200 sign classes in 1920×1080 frames. The
task here is multi-label recognition of the 20 classes with the most (non-ignored) training
annotations. The official train/test split is kept. The validation split takes every 7th block
of 100 consecutive frame numbers from the training images, so neighbouring video frames stay on
one side of the train/val split.
"""
import argparse
import collections
import io
import json
import os
import tarfile
import urllib.request
import zipfile

IMAGES_URL = "https://data.vicos.si/skokec/villard/JPEGImages.tar.bz2"
ANNOTATIONS_URL = "https://go.vicos.si/dfgannot"
TOP_CLASSES = 20
VAL_BLOCK, VAL_EVERY = 100, 7


def build_mnist(data_dir):
    import torchvision
    torchvision.datasets.MNIST(data_dir, train=True, download=True)
    torchvision.datasets.MNIST(data_dir, train=False, download=True)
    print("MNIST ready in", data_dir)


class Counted(io.RawIOBase):
    """Wraps a response so reading stops after `limit` bytes (None = no limit)."""

    def __init__(self, response, limit):
        self.response, self.limit, self.read_bytes = response, limit, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.limit is not None and self.read_bytes >= self.limit:
            return 0
        chunk = self.response.read(len(buffer))
        buffer[:len(chunk)] = chunk
        self.read_bytes += len(chunk)
        if self.read_bytes // (256 << 20) != (self.read_bytes - len(chunk)) // (256 << 20):
            print(f"  {self.read_bytes / 2 ** 30:.2f} GB downloaded", flush=True)
        return len(chunk)


def download_images(root, limit_mb):
    print("Downloading and extracting", IMAGES_URL, flush=True)
    limit = None if limit_mb is None else limit_mb << 20
    with urllib.request.urlopen(IMAGES_URL) as response:
        stream = io.BufferedReader(Counted(response, limit), buffer_size=1 << 20)
        with tarfile.open(fileobj=stream, mode="r|bz2") as archive:
            last = None
            try:
                for member in archive:
                    if member.isfile() and member.name.lower().endswith(".jpg"):
                        member.name = os.path.join("JPEGImages", os.path.basename(member.name))
                        archive.extract(member, root)
                        last = os.path.join(root, member.name)
            except (EOFError, tarfile.ReadError):
                if limit is None:
                    raise
                if last is not None:
                    os.remove(last)   # the file being extracted when the download stopped is truncated
                print("  stopped at the download limit (partial archive)")


def load_annotations(root, annotations_zip):
    if annotations_zip is None:
        annotations_zip = os.path.join(root, "annotations.zip")
        if not os.path.exists(annotations_zip):
            print("Downloading", ANNOTATIONS_URL, flush=True)
            urllib.request.urlretrieve(ANNOTATIONS_URL, annotations_zip)
    with zipfile.ZipFile(annotations_zip) as z:
        names = {os.path.basename(n): n for n in z.namelist()}
        return {split: json.loads(z.read(names[f"{split}.json"])) for split in ("train", "test")}


def entries(coco, keep, folder):
    category = {c: i for i, c in enumerate(keep)}
    by_image = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)
    out = []
    for image in sorted(coco["images"], key=lambda i: i["file_name"]):
        if not os.path.exists(os.path.join(folder, image["file_name"])):
            continue
        labels, ignore, boxes = set(), set(), []
        for a in by_image[image["id"]]:
            if a["category_id"] not in category:
                continue
            c = category[a["category_id"]]
            if a.get("ignore"):
                ignore.add(c)
            else:
                labels.add(c)
                x, y, w, h = a["bbox"]
                boxes.append([y, x, y + h, x + w])
        out.append({"file": image["file_name"], "labels": sorted(labels), "ignore": sorted(ignore), "boxes": boxes})
    return out


def build_dfg(data_dir, limit_mb, annotations_zip, index_only=False):
    root = os.path.join(data_dir, "dfg")
    folder = os.path.join(root, "JPEGImages")
    os.makedirs(root, exist_ok=True)
    if not index_only and (limit_mb is not None or not os.path.isdir(folder) or len(os.listdir(folder)) < 6900):
        download_images(root, limit_mb)
    coco = load_annotations(root, annotations_zip)
    counts = collections.Counter(a["category_id"] for a in coco["train"]["annotations"] if not a.get("ignore"))
    keep = [c for c, _ in counts.most_common(TOP_CLASSES)]
    names = {c["id"]: c["name"] for c in coco["train"]["categories"]}
    train = entries(coco["train"], keep, folder)
    is_val = lambda e: (int(os.path.splitext(e["file"])[0]) // VAL_BLOCK) % VAL_EVERY == 0
    index = {"class_names": [names[c] for c in keep], "category_ids": keep,
             "train": [e for e in train if not is_val(e)], "val": [e for e in train if is_val(e)],
             "test": entries(coco["test"], keep, folder)}
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    for split in ("train", "val", "test"):
        positives = sum(bool(e["labels"]) for e in index[split])
        print(f"{split}: {len(index[split])} images, {positives} with at least one top-{TOP_CLASSES} sign")
    print("classes:", ", ".join(index["class_names"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["mnist", "dfg"])
    parser.add_argument("--data-dir", default=os.environ.get("RART_DATA_DIR", os.path.abspath("data")))
    parser.add_argument("--limit-mb", type=int, default=None, help="stop the image download after this many MB")
    parser.add_argument("--annotations", default=None, help="a local copy of the annotation zip")
    parser.add_argument("--index-only", action="store_true", help="rebuild index.json from the images already there")
    args = parser.parse_args()
    if args.dataset == "mnist":
        build_mnist(args.data_dir)
    else:
        build_dfg(args.data_dir, args.limit_mb, args.annotations, args.index_only)
