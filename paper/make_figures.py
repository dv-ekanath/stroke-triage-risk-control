"""Generates the paper's plots. Every value is copied from kgxai_paper.tex; nothing is recomputed.
Run: python paper/make_figures.py   (writes PDF + PNG into paper/figures/)
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Two hues that stay distinguishable in grayscale print, plus hatching as a second cue.
C_BASE, C_GRAPH, C_GREY, C_RED = "#7a7a7a", "#1f5fa8", "#bdbdbd", "#c0392b"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})
W = 3.45  # IEEE single-column width, inches


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# 1. Strict JDCR per fold/seed, B0 vs B2 (Table III of the full paper)
runs = ["f0/s0", "f0/s1", "f0/s2", "f1/s0", "f1/s1", "f1/s2", "f2/s0", "f2/s1", "f2/s2"]
b0 = [0.930, 0.000, 0.977, 1.000, 0.000, 0.973, 0.977, 0.953, 0.930]
b2 = [1.000, 0.977, 0.953, 1.000, 0.973, 0.946, 0.953, 0.930, 0.953]
x = np.arange(len(runs)); w = 0.38
fig, ax = plt.subplots(figsize=(W, 2.2))
ax.bar(x - w/2, b0, w, color=C_BASE, hatch="///", edgecolor="white", lw=0.4, label="B0 (no graph)")
ax.bar(x + w/2, b2, w, color=C_GRAPH, edgecolor="white", lw=0.4, label="B2 (graph + consistency)")
for i in (1, 4):
    ax.annotate("collapsed", (x[i] - w/2, 0.02), xytext=(x[i] - w/2, 0.3), ha="center", fontsize=6.5,
                color=C_RED, rotation=90, va="bottom")
ax.set_xticks(x); ax.set_xticklabels(runs, fontsize=6.5)
ax.set_xlabel("fold / seed"); ax.set_ylabel("Strict JDCR"); ax.set_ylim(0, 1.12)
ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.grid(axis="y", lw=0.3, color=C_GREY); ax.set_axisbelow(True)
save(fig, "fig_jdcr_grid")

# 2. Final Dice, mean +/- SD (Table II)
labels = ["B0\nall (n=9)", "B0\nexcl. collapse\n(n=7)", "B1\n(n=3)", "B2\n(n=9)"]
mean = [0.191, 0.240, 0.225, 0.242]
sd = [0.092, 0.010, 0.002, 0.012]
cols = [C_BASE, C_BASE, C_GRAPH, C_GRAPH]
fig, ax = plt.subplots(figsize=(W, 2.2))
bars = ax.bar(range(4), mean, yerr=sd, color=cols, capsize=2.5, error_kw={"lw": 0.7},
              edgecolor="white", lw=0.4)
bars[0].set_hatch("///")
for i, m in enumerate(mean):
    ax.text(i, 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=7, color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylabel("Final Dice"); ax.set_ylim(0, 0.34)
ax.grid(axis="y", lw=0.3, color=C_GREY); ax.set_axisbelow(True)
save(fig, "fig_dice_summary")

# 3. Failure detection: AUROC bars + who catches the 156 severe failures
fig, (a, b) = plt.subplots(1, 2, figsize=(W * 2 + 0.2, 2.2), gridspec_kw={"width_ratios": [1, 1]})
names = ["Knowledge\nflag", "Ensemble\ndisagreement", "Volume\ncoeff. of var."]
auc = [0.836, 0.583, 0.575]
br = a.bar(range(3), auc, color=[C_GRAPH, C_BASE, C_BASE], edgecolor="white", lw=0.4)
br[1].set_hatch("///"); br[2].set_hatch("///")
a.axhline(0.5, color="black", lw=0.6, ls="--"); a.text(-0.45, 0.505, "chance", fontsize=6.5, ha="left", va="bottom")
for i, v in enumerate(auc):
    a.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
a.set_xticks(range(3)); a.set_xticklabels(names, fontsize=6.5)
a.set_ylabel("AUROC (severe failure)"); a.set_ylim(0.4, 0.92)
a.set_title("(a) Detecting Dice < 0.05, n = 297", fontsize=7.5)

parts = [("Flag only", 76, C_GRAPH, None), ("Both", 32, "#6d97c9", None),
         ("Disagreement only", 19, C_BASE, "///"), ("Missed by both", 29, C_GREY, None)]
left = 0
for lab, n, c, h in parts:
    b.barh(0, n, left=left, color=c, hatch=h, edgecolor="white", lw=1.2, label=f"{lab} ({n})")
    left += n
b.set_xlim(0, 156); b.set_yticks([]); b.set_xlabel("Severe failures (of 156)")
b.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
b.spines["left"].set_visible(False)
b.set_title("(b) Who catches the failures", fontsize=7.5)
save(fig, "fig_failure_detection")
print("wrote", sorted(p.name for p in OUT.iterdir()))
