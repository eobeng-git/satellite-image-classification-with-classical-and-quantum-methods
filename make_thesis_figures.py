#!/usr/bin/env python3
"""
make_thesis_figures.py
----------------------
Regenerates every thesis figure that depends on the final multiclass run.
No numbers are typed in by hand: everything is read from

    results_qnn_full/qnn_full_experiment_accuracy_42.csv   (per-fold results)
    job-progress.log                                       (per-epoch loss/accuracy)

Run it on the cluster from the folder that contains both:

    python make_thesis_figures.py

Output (PDF for LaTeX + PNG preview) goes to ./thesis_figures_v2/ :

    fig3_3_architecture          Figure 3.3  PennyQNN architecture
    fig4_1_learning_curve        Figure 4.1  loss / training accuracy per epoch
    fig4_2_test_accuracy_boxplot Figure 4.2  test accuracy across folds
    fig4_3_training_time         Figure 4.3  training time (linear + log)
    fig4_4_train_vs_test         Figure 4.4  train vs test accuracy
    fig4_5_per_fold_accuracy     Figure 4.5  per-fold test accuracy

Only needs numpy, pandas and matplotlib (already in the qnn-310 environment).
"""
import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless: no display on the cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------
# arguments / setup
# --------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results_qnn_full/qnn_full_experiment_accuracy_42.csv")
ap.add_argument("--log", default="job-progress.log")
ap.add_argument("--out", default="thesis_figures_v2")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

NAMES = {"knn": "KNN", "svc": "SVM", "dtree": "Decision Tree", "nb": "Naive Bayes", "penny_qnn": "PennyQNN"}
COL = {"knn": "#4C72B0", "svc": "#55A868", "dtree": "#DD8452", "penny_qnn": "#8172B2", "nb": "#CCB974"}
CV_TRAIN = {"knn", "svc", "dtree"}  # train score = cross-validated grid-search score


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.join(args.out, name) + ".pdf/.png")


def fmt_time(s):
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 60:
        return f"{s / 60:.1f} min"
    return f"{s:.2f} s" if s < 1 else f"{s:.1f} s"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
df = pd.read_csv(args.csv, index_col=0)
df["train_pct"] = df["train_score"] * 100
df["test_pct"] = df["test_score"] * 100
grp = df.groupby("cls_name")
mean_test, std_test = grp["test_pct"].mean(), grp["test_pct"].std()
mean_train, std_train = grp["train_pct"].mean(), grp["train_pct"].std()
mean_time = grp["time_seconds"].mean()
ORDER = [m for m in mean_test.sort_values(ascending=False).index]  # by test accuracy

# per-epoch loss / accuracy from the log
pat = re.compile(r"Epoch (\d+)/(\d+) \| Loss: ([\d.]+) \| Accuracy: ([\d.]+)")
epochs, fold = {}, 1
with open(args.log, encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        m = pat.search(line)
        if m:
            ep, tot = int(m.group(1)), int(m.group(2))
            epochs.setdefault(fold, []).append((ep, float(m.group(3)), float(m.group(4)) * 100))
            if ep == tot:
                fold += 1
for f, rows in epochs.items():
    print(f"fold {f}: {len(rows)} epochs read")
assert all(len(r) == 10 for r in epochs.values()) and len(epochs) == 3, "unexpected epoch log layout"

# --------------------------------------------------------------------------
# Figure 4.1 - learning curves
# --------------------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
for f in (1, 2, 3):
    e = np.array(epochs[f])
    a.plot(e[:, 0], e[:, 1], marker="o", ms=4, lw=1.4, label=f"Fold {f}")
    b.plot(e[:, 0], e[:, 2], marker="o", ms=4, lw=1.4, label=f"Fold {f}")
a.set_xlabel("Epoch"); a.set_ylabel("Training loss (cross-entropy)")
a.set_title(f"(a) Loss  (random guessing = ln 10 = {np.log(10):.3f})"); a.set_xticks(range(1, 11)); a.grid(alpha=0.3)
a.legend(loc="center right")
b.axhline(mean_test["penny_qnn"], ls="--", c="grey", lw=1,
          label=f"Mean test accuracy ({mean_test['penny_qnn']:.2f}%)")
b.set_xlabel("Epoch"); b.set_ylabel("Training accuracy (%)")
b.set_title("(b) Training accuracy"); b.set_xticks(range(1, 11)); b.grid(alpha=0.3); b.set_ylim(66.0, 70.2)
b.legend(loc="lower right")
fig.tight_layout()
save(fig, "fig4_1_learning_curve")

# --------------------------------------------------------------------------
# Figure 4.2 - test accuracy distribution across folds (box plot)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
data = [df[df.cls_name == m]["test_pct"].values for m in ORDER]
bp = ax.boxplot(data, positions=range(len(ORDER)), widths=0.5, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1.2))
for patch, m in zip(bp["boxes"], ORDER):
    patch.set_facecolor(COL[m]); patch.set_alpha(0.55)
rng = np.random.RandomState(1)
for i, d in enumerate(data):
    ax.scatter(i + rng.uniform(-0.08, 0.08, len(d)), d, c="black", s=14, zorder=3,
               label="Individual fold" if i == 0 else None)
    ax.text(i, 88.2, f"{d.mean():.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([NAMES[m] for m in ORDER])
ax.set_ylim(62, 90); ax.set_ylabel("Test accuracy (%)")
ax.set_title("Test accuracy across the 3 cross-validation folds", pad=18)
ax.grid(axis="y", alpha=0.3); ax.legend(loc="lower left")
fig.tight_layout()
save(fig, "fig4_2_test_accuracy_boxplot")

# --------------------------------------------------------------------------
# Figure 4.3 - training time (linear and log)
# --------------------------------------------------------------------------
t_order = list(mean_time.sort_values(ascending=False).index)
vals = [mean_time[m] for m in t_order]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
for ax, log in ((a, False), (b, True)):
    bars = ax.bar(range(len(t_order)), vals, color=[COL[m] for m in t_order], edgecolor="black", lw=0.6)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v * (1.15 if log else 1.0) + (0 if log else max(vals) * 0.012),
                fmt_time(v), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(t_order)))
    ax.set_xticklabels([NAMES[m] for m in t_order], rotation=30, ha="right")
    ax.set_ylabel("Training time (seconds)" + (", log scale" if log else ""))
    if log:
        ax.set_yscale("log"); ax.set_ylim(0.1, max(vals) * 8)
    ax.grid(axis="y", alpha=0.3)
a.set_title("Mean training time per fold (linear scale)")
b.set_title("Mean training time per fold (log scale)")
fig.tight_layout()
save(fig, "fig4_3_training_time")

# --------------------------------------------------------------------------
# Figure 4.4 - train vs test accuracy
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 4.4))
x, w = np.arange(len(ORDER)), 0.36
tr = [mean_train[m] for m in ORDER]; te = [mean_test[m] for m in ORDER]
ax.bar(x - w / 2, tr, w, yerr=[std_train[m] for m in ORDER], color="#4C72B0", label="Train accuracy", capsize=2)
ax.bar(x + w / 2, te, w, yerr=[std_test[m] for m in ORDER], color="#C44E52", label="Test accuracy", capsize=2)
for xi, m, a_, b_ in zip(x, ORDER, tr, te):
    ax.text(xi - w / 2, a_ + std_train[m] + 1.2, f"{a_:.2f}", ha="center", fontsize=7.5)
    ax.text(xi + w / 2, b_ + std_test[m] + 1.2, f"{b_:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels([NAMES[m] + ("*" if m in CV_TRAIN else "") for m in ORDER])
ax.set_ylim(0, 100); ax.set_ylabel("Accuracy (%)")
ax.set_title("Train vs. test accuracy by model")
ax.grid(axis="y", alpha=0.3); ax.legend(loc="lower right")
fig.text(0.01, -0.02,
         "* Train accuracy = cross-validated score of the best grid-search parameters (models fitted on half of the\n"
         "   training data). Naive Bayes and PennyQNN: accuracy on the full training set. Error bars: std over 3 folds.",
         fontsize=7.5, ha="left", va="top")
fig.tight_layout()
save(fig, "fig4_4_train_vs_test")

# --------------------------------------------------------------------------
# Figure 4.5 - per-fold test accuracy
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4))
mk = {1: "o", 2: "s", 3: "^"}
for fo in (1, 2, 3):
    ys = [df[(df.cls_name == m) & (df.fold == fo)]["test_pct"].values[0] for m in ORDER]
    ax.plot(range(len(ORDER)), ys, ls="", marker=mk[fo], ms=7, mfc="white", mec="black", label=f"Fold {fo}", zorder=3)
for i, m in enumerate(ORDER):
    ax.hlines(mean_test[m], i - 0.25, i + 0.25, colors=COL[m], lw=3, zorder=2)
ax.plot([], [], c="grey", lw=3, label="Fold mean")
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([f"{NAMES[m]}\n$\\sigma$={std_test[m] / 100:.4f}" for m in ORDER], fontsize=9)
ax.set_ylim(62, 86); ax.set_ylabel("Test accuracy (%)")
ax.set_title("Per-fold test accuracy of each model")
ax.grid(axis="y", alpha=0.3); ax.legend(loc="lower left", ncol=2)
fig.tight_layout()
save(fig, "fig4_5_per_fold_accuracy")


# --------------------------------------------------------------------------
# Figure 3.3 - PennyQNN architecture (multi-class readout)
# --------------------------------------------------------------------------
def architecture():
    PINK, BLUE, GREEN, YELLOW = "#F8B9C6", "#BFCDEE", "#CFE8CF", "#FBE9A6"
    fig, ax = plt.subplots(figsize=(13.8, 6.3))
    ax.set_xlim(0, 27.6); ax.set_ylim(0, 12.6); ax.set_aspect("equal"); ax.axis("off")
    ys = [8.6, 6.6, 4.6, 2.6]
    x_wire0, x_wire1 = 1.6, 20.6

    ax.text(13.8, 12.2, "PennyQNN: 4-qubit quantum neural network (multi-class readout)",
            ha="center", va="center", fontsize=13, fontweight="bold")

    blocks = [  # (x0, x1, colour, header, caption)
        (2.2, 4.4, PINK, "Data\nEncoding", r"$x_i$: band-group mean"),
        (5.0, 7.2, BLUE, "Variational\nLayer 1", r"$\theta_{1,i,0},\ \theta_{1,i,1}$"),
        (7.8, 11.0, GREEN, "Entangling\nLayer 1", "linear + cyclic CNOT"),
        (11.6, 13.8, BLUE, "Variational\nLayer 2", r"$\theta_{2,i,0},\ \theta_{2,i,1}$"),
        (14.4, 17.6, GREEN, "Entangling\nLayer 2", "linear + cyclic CNOT"),
        (17.95, 20.3, YELLOW, "Measurement", "all 4 qubits"),
    ]
    for x0, x1, c, head, cap in blocks:
        ax.add_patch(FancyBboxPatch((x0, 1.7), x1 - x0, 8.0, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    fc=c, ec="none", alpha=0.45, zorder=0))
        ax.add_patch(FancyBboxPatch((x0, 10.1), x1 - x0, 0.95, boxstyle="round,pad=0.02,rounding_size=0.2",
                                    fc=c, ec="#555", lw=0.8, zorder=2))
        ax.text((x0 + x1) / 2, 10.58, head, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text((x0 + x1) / 2, 1.1, cap, ha="center", va="center", fontsize=7.2)

    for i, y in enumerate(ys):  # wires, labels, |0>
        ax.plot([x_wire0, x_wire1], [y, y], c="#444", lw=1, zorder=1)
        ax.text(0.2, y, f"Q{i + 1}", ha="left", va="center", fontsize=10, fontweight="bold")
        ax.add_patch(Circle((1.1, y), 0.36, fc="white", ec="#444", lw=1, zorder=3))
        ax.text(1.1, y, r"$|0\rangle$", ha="center", va="center", fontsize=7.5, zorder=4)

    def gate(xc, y, label, fc="white", w=0.9, h=0.8, fs=7.5):
        ax.add_patch(Rectangle((xc - w / 2, y - h / 2), w, h, fc=fc, ec="#222", lw=1, zorder=3))
        ax.text(xc, y, label, ha="center", va="center", fontsize=fs, zorder=4)

    for i, y in enumerate(ys):
        gate(3.3, y, rf"$R_Y(x_{i + 1})$", fc="#FFFFFF", w=1.7, fs=7.5)
        for (x0, _, _, _, _) in (blocks[1], blocks[3]):
            gate(x0 + 0.6, y, r"$R_Y$"); gate(x0 + 1.6, y, r"$R_Z$")

    def cnot(xc, c, t):
        ax.plot([xc, xc], [ys[c], ys[t]], c="#222", lw=1.2, zorder=3)
        ax.add_patch(Circle((xc, ys[c]), 0.13, fc="#222", ec="#222", zorder=4))
        ax.add_patch(Circle((xc, ys[t]), 0.3, fc="white", ec="#222", lw=1.2, zorder=4))
        ax.plot([xc - 0.3, xc + 0.3], [ys[t], ys[t]], c="#222", lw=1.2, zorder=5)
        ax.plot([xc, xc], [ys[t] - 0.3, ys[t] + 0.3], c="#222", lw=1.2, zorder=5)

    for x0 in (blocks[2][0], blocks[4][0]):  # CNOT(0,1) CNOT(1,2) CNOT(2,3) CNOT(3,0)
        for k, (c, t) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
            cnot(x0 + 0.5 + 0.72 * k, c, t)

    mx = (blocks[5][0] + blocks[5][1]) / 2
    for y in ys:  # measurement meters
        ax.add_patch(Rectangle((mx - 0.45, y - 0.4), 0.9, 0.8, fc="white", ec="#222", lw=1, zorder=3))
        ax.text(mx, y - 0.03, "M", ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)

    # right panel ---------------------------------------------------------
    ax.annotate("", xy=(21.5, 5.6), xytext=(20.5, 5.6), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333"))
    # formulas
    ax.add_patch(FancyBboxPatch((21.5, 8.0), 5.6, 2.9, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="#EAF1FB", ec="#7A93B8", lw=0.9))
    ax.text(21.7, 10.45, r"$|\psi\rangle = U_2(\theta_2)\,U_1(\theta_1)\,\prod_{i}R_Y(x_i)\,|0\rangle$",
            fontsize=8.2, va="center")
    ax.text(21.7, 9.5, r"$p_k = P(\mathbf{z}_k)\,/\,\sum_{j=1}^{10}P(\mathbf{z}_j)$", fontsize=8.2, va="center")
    ax.text(21.7, 8.55, r"$\hat{y} = \mathrm{classes}\,[\,\arg\max_k\ p_k\,]$", fontsize=8.2, va="center")
    # probability bars
    rs = np.random.RandomState(3)
    h = rs.uniform(0.5, 2.1, 16)
    bx0, bw, base = 21.8, 0.25, 4.6
    for k in range(16):
        used = k < 10
        ax.add_patch(Rectangle((bx0 + k * 0.265, base), bw, h[k], fc="#4C72B0" if used else "#BBBBBB", ec="none"))
    ax.plot([bx0 - 0.1, bx0 + 16 * 0.265], [base, base], c="#333", lw=0.8)
    ax.text(21.7, 7.35, r"measured $P(\mathbf{z})$ (schematic): 16 outcomes", fontsize=7.5, va="center")
    ax.text(bx0 + 5 * 0.265, base - 0.35, "first 10: class scores", fontsize=7.2, ha="center", va="top", color="#2E4C85")
    ax.text(bx0 + 13 * 0.265, base - 0.35, "last 6:\nunused", fontsize=7.2, ha="center", va="top", color="#777")
    # statistics
    ax.add_patch(FancyBboxPatch((21.5, 0.2), 5.6, 3.0, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="white", ec="#555", lw=0.9))
    stats = ["Statistics", "Qubits: 4   |   Variational layers: 2", "Trainable parameters: 16",
             "Readout: qml.probs (all 4 qubits)", "Classes: 10 (of 16 basis states)",
             "Loss: categorical cross-entropy", "Optimizer: gradient descent (lr = 0.1)"]
    for k, s in enumerate(stats):
        ax.text(21.7, 2.9 - k * 0.4, s, fontsize=7.6, va="center", fontweight="bold" if k == 0 else "normal")
    return fig


save(architecture(), "fig3_3_architecture")
print("\nDone. Copy the thesis_figures_v2 folder to your computer with scp.")
