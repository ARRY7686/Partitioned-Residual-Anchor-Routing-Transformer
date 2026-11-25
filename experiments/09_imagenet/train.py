"""ImageNet-100 / ImageNet-1k training under PROTOCOL.md (the DeiT / PVTv2 recipe).

Reads only the train and val splits written by prepare_data.py; the test split (official
ILSVRC-2012 val) is opened only by evaluate_test.py. The result of a run is its FINAL
checkpoint's validation accuracy (no best-epoch picking). A non-finite loss ends the run with
status "diverged", which is a result, not a reason to rerun (PROTOCOL.md, divergence policy).

  python train.py --data data/in100 --model anchor_b0_fixed --lr 5e-4 --epochs 100 --seed 1
  python train.py --data data/in100 --model timm:resnet18 ...        # any timm model
  python train.py --data data/smoke --model pvt_v2_b0 --epochs 2 --batch 32 --classes 10   # smoke
  python train.py --data data/in100 --bench-loader 200 --workers 16   # data loader speed only

Runs resume from <out>/last.pt automatically (vast.ai boxes can be pre-empted). Every epoch is
seeded from (seed, epoch), so a resumed run sees the same data order and augmentation seeds as
an uninterrupted one (up to non-deterministic GPU kernels).
"""
import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "08_efficient_routing"))


# ----------------------------------------------------------------------------- data

class Packed(torch.utils.data.Dataset):
    """One split of prepare_data.py's output. `classes`: keep only labels < classes (smoke)."""

    def __init__(self, root, split, transform, classes=0):
        assert split in ("train", "val", "test")
        self.path = os.path.join(root, f"{split}.bin")
        self.offsets = np.load(os.path.join(root, f"{split}.offsets.npy"))
        labels = np.load(os.path.join(root, f"{split}.labels.npy")).astype(np.int64)
        self.idx = np.arange(len(labels)) if not classes else np.nonzero(labels < classes)[0]
        self.labels, self.transform, self._mm = labels, transform, None

    def __len__(self):
        return len(self.idx)

    def __getstate__(self):          # workers open their own memmap
        state = dict(self.__dict__)
        state["_mm"] = None
        return state

    def __getitem__(self, i):
        j = self.idx[i]
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        data = self._mm[self.offsets[j]:self.offsets[j + 1]].tobytes()
        return self.transform(Image.open(io.BytesIO(data)).convert("RGB")), self.labels[j]


class EpochSampler(torch.utils.data.Sampler):
    """A fresh permutation per epoch from (seed, epoch); independent of how the run got there."""

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch = n, seed, 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g).tolist())

    def __len__(self):
        return self.n


def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def transforms(args):
    from timm.data import create_transform
    train = create_transform(args.img_size, is_training=True, auto_augment=args.randaug or None,
                             interpolation="bicubic", re_prob=args.reprob, re_mode="pixel", re_count=1)
    test = create_transform(args.img_size, is_training=False, interpolation="bicubic", crop_pct=args.crop_pct)
    return train, test


def loader(dataset, batch, workers, train, sampler=None):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch, sampler=sampler, shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=train, persistent_workers=False,
        prefetch_factor=4 if workers else None)


# ----------------------------------------------------------------------------- model

def make_model(name, num_classes, drop_path):
    import timm
    from compare_v2 import VARIANTS
    from models_v2 import MixerConfig, build
    if name in VARIANTS:
        timm_name, mixer = VARIANTS[name]
        return build(timm_name, MixerConfig(**mixer) if mixer is not None else None,
                     num_classes=num_classes, drop_path_rate=drop_path)
    if name.startswith("timm:"):
        return timm.create_model(name[5:], num_classes=num_classes, drop_path_rate=drop_path)
    raise SystemExit(f"unknown model {name}; use one of {sorted(VARIANTS)} or timm:<name>")


def param_groups(model, weight_decay):
    """No weight decay on biases, norms and learned tokens (timm / DeiT convention); the anchor
    seed is a learned token like DeiT's cls token."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith("anchor_seed") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def gflops(model, size, device):
    from torch.utils.flop_counter import FlopCounterMode
    was = model.training
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(torch.randn(1, 3, size, size, device=device))
    model.train(was)
    return counter.get_total_flops() / 2 / 1e9   # multiply-adds, as papers report


# ----------------------------------------------------------------------------- loop

def lr_at(step, total, warmup, base, minimum):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    n, loss, top1, top5 = 0, 0.0, 0, 0
    for x, y in data:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out = model(x).float()
        loss += F.cross_entropy(out, y, reduction="sum").item()
        top = out.topk(min(5, out.shape[1]), dim=1).indices
        top1 += (top[:, 0] == y).sum().item()
        top5 += (top == y[:, None]).any(1).sum().item()
        n += len(y)
    model.train()
    return {"loss": loss / n, "top1": 100 * top1 / n, "top5": 100 * top5 / n, "images": n}


def save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def bench_loader(args):
    train_tf, _ = transforms(args)
    ds = Packed(args.data, "train", train_tf, args.classes)
    data = loader(ds, args.batch, args.workers, True, EpochSampler(len(ds), 0))
    it, start, n = iter(data), None, 0
    for i in range(args.bench_loader):
        x, _ = next(it)
        if i == 10:
            start = time.perf_counter()
        elif i > 10:
            n += len(x)
    rate = n / (time.perf_counter() - start)
    print(f"loader: {rate:.0f} images/s with {args.workers} workers ({rate / args.workers:.0f} per worker)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="pvt_v2_b0")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4, help="peak LR at --batch (not rescaled)")
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=float, default=5)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.8)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--randaug", default="rand-m9-mstd0.5-inc1")
    p.add_argument("--reprob", type=float, default=0.25)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--crop-pct", type=float, default=0.875)
    p.add_argument("--classes", type=int, default=0, help="smoke test: keep only labels < N")
    p.add_argument("--workers", type=int, default=max(1, num_cpus() - 2))
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps")
    p.add_argument("--bench-loader", type=int, default=0)
    args = p.parse_args()
    if args.bench_loader:
        return bench_loader(args)

    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    num_classes = args.classes or len(meta["classes"])
    name = args.name or f"{args.model.replace(':', '-')}_lr{args.lr:g}_ep{args.epochs}_s{args.seed}"
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)

    train_tf, test_tf = transforms(args)
    train_ds = Packed(args.data, "train", train_tf, args.classes)
    val_ds = Packed(args.data, "val", test_tf, args.classes)
    sampler = EpochSampler(len(train_ds), args.seed)
    val_data = loader(val_ds, 256, args.workers, False)
    steps_per_epoch = len(train_ds) // args.batch
    assert steps_per_epoch > 0, "batch larger than the training set"
    total = steps_per_epoch * args.epochs
    warmup = int(args.warmup_epochs * steps_per_epoch)

    torch.manual_seed(args.seed)
    model = make_model(args.model, num_classes, args.drop_path).to(device)
    info = {"params_m": sum(p.numel() for p in model.parameters()) / 1e6,
            "gflops": gflops(model, args.img_size, device)}
    optimizer = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, fused=device.type == "cuda")
    mixup = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, label_smoothing=args.smoothing,
                  num_classes=num_classes) if (args.mixup or args.cutmix) else None
    criterion = SoftTargetCrossEntropy()

    ckpt_path, start_epoch, resumes = os.path.join(out, "last.pt"), 0, 0
    history = []
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, resumes, history = ckpt["epoch"], ckpt["resumes"] + 1, ckpt["history"]
        print(f"resumed {out} at epoch {start_epoch}", flush=True)
    else:
        json.dump({"args": vars(args), "data_meta": {k: v for k, v in meta.items() if k != "classes"},
                   "model_info": info, "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
                  open(os.path.join(out, "config.json"), "w"), indent=1)
    if os.path.exists(os.path.join(out, "result.json")):
        print(f"{out} already finished"); return
    step_model = torch.compile(model) if args.compile else model
    print(f"{name}: {info['params_m']:.2f}M params, {info['gflops']:.2f} GFLOPs, "
          f"{len(train_ds)} train / {len(val_ds)} val images, {steps_per_epoch} steps/epoch", flush=True)

    status, step = "finished", start_epoch * steps_per_epoch
    run_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        # seed everything that draws random numbers this epoch (Mixup uses numpy's global RNG;
        # DataLoader workers take their seeds from torch's RNG when the iterator is created)
        torch.manual_seed(args.seed * 100003 + epoch)
        np.random.seed((args.seed * 100003 + epoch) % 2 ** 32)
        sampler.epoch = epoch
        train_data = loader(train_ds, args.batch, args.workers, True, sampler)
        loss_sum, count, wait, seen = torch.zeros((), device=device), 0, 0.0, 0
        t0 = tick = time.perf_counter()
        for x, y in train_data:
            wait += time.perf_counter() - tick
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mixup is not None:
                x, y = mixup(x, y)
            else:
                y = F.one_hot(y, num_classes).float() * (1 - args.smoothing) + args.smoothing / num_classes
            lr = lr_at(step, total, warmup, args.lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = step_model(x)
            loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            loss_sum += loss.detach()
            count += 1
            seen += len(x)
            step += 1
            if count % args.log_every == 0:
                mean = (loss_sum / count).item()          # the only sync per log interval
                if not math.isfinite(mean):
                    status = "diverged"
                    break
                dt = time.perf_counter() - t0
                print(f"ep {epoch} step {count}/{steps_per_epoch} loss {mean:.4f} lr {lr:.2e} "
                      f"{seen / dt:.0f} img/s data-wait {100 * wait / dt:.0f}%", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
            tick = time.perf_counter()
        if status == "finished" and count and not math.isfinite((loss_sum / count).item()):
            status = "diverged"
        if status == "diverged":
            print(f"DIVERGED at epoch {epoch} (non-finite training loss)", flush=True)
            break
        dt = time.perf_counter() - t0
        record = {"epoch": epoch + 1, "train_loss": (loss_sum / max(1, count)).item(), "lr": lr,
                  "epoch_s": dt, "img_s": seen / dt, "data_wait": wait / dt}
        last = epoch + 1 == args.epochs or (args.max_steps and step >= args.max_steps)
        if (epoch + 1) % args.eval_every == 0 or last:
            record["val"] = evaluate(model, val_data, device)
        history.append(record)
        print(json.dumps(record), flush=True)
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                     "resumes": resumes, "history": history}, ckpt_path)
        if last:
            break

    result = {"status": status, "name": name, "model": args.model, "seed": args.seed, "lr": args.lr,
              "epochs_planned": args.epochs, "epochs_done": len(history), "resumes": resumes,
              "final_val": history[-1].get("val") if history and status == "finished" else None,
              "smoke": bool(args.classes or args.max_steps), "wall_h_this_session": (time.time() - run_start) / 3600,
              **info}
    if status == "finished":
        torch.save(model.state_dict(), os.path.join(out, "final.pt"))
    with open(os.path.join(out, "history.json"), "w") as fh:
        json.dump(history, fh, indent=1)
    with open(os.path.join(out, "result.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
