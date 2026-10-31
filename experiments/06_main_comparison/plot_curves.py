"""Test-accuracy curves for the paper: 30-epoch Cluttered CIFAR-10 (phase B + R) and
15-epoch Tiny ImageNet canvas (phase T + R). Mean over seeds, band = min..max.

usage: python plot_curves.py [results_dir] [out.pdf]
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "exp06_results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "paper", "figures", "curves.pdf")

# (config suffix, label, colour, line style). Anchor models solid, baselines dashed/dotted,
# so identity never depends on colour alone.
SERIES = [
    ("focus_A16", "focus-16", "#2a78d6", "-"),
    ("fixed_A16", "fixed-16", "#eb6834", "-"),
    ("vit256", "ViT 256$^2$", "#1baf7a", "--"),
    ("vit128", "ViT 128$^2$", "#eda100", "--"),
    ("pvt", "PVTv2-B0", "#e87ba4", ":"),
    ("resnet18_lr1e-3", "ResNet-18", "#008300", ":"),
]


def curves(prefix, key):
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, f"{prefix}_{key}_seed*.json"))):
        runs.append([h["test_acc"] for h in json.load(open(path))["history"]])
    return np.array(runs) if runs else None


plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
                     "xtick.color": "#555555", "ytick.color": "#555555"})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, prefix, title in [(axes[0], "B", "Cluttered CIFAR-10, 30 epochs"),
                          (axes[1], "T", "Tiny ImageNet canvas, 15 epochs")]:
    ends = []
    for key, label, colour, style in SERIES:
        runs = curves(prefix, key)
        if runs is None:
            continue
        x = np.arange(1, runs.shape[1] + 1)
        if "resnet" not in key:  # ResNet's per-epoch test accuracy swings by 30+ points
            ax.fill_between(x, runs.min(0), runs.max(0), color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, runs.mean(0), color=colour, linestyle=style, linewidth=1.5, label=label)
        ends.append([runs.mean(0)[-1], runs[:, -1].mean(), x[-1]])
    # End labels, pushed apart so close curves stay readable.
    ends.sort()
    gap = 0.045 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, value, last in ends:
        ax.annotate(f"{value:.1f}", (last, y), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color="#333333")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title(title, fontsize=8.5, color="#222222")
    ax.set_xlabel("epoch")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_xlim(1, ax.get_xlim()[1] + 1.5)
axes[0].set_ylabel("test accuracy (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=7,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("saved", OUT)
