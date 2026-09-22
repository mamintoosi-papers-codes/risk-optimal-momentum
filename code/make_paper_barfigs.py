"""Bar charts for paper tables: real regression gaps and GNN accuracy (PDF)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent.parent / "paper"
PAPER.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "figure.dpi": 200, "savefig.bbox": "tight",
})

# --- real regression: paired gap AdamW - SR-Adam (positive = SR better)
labels = [
    "diabetes lin", "diabetes MLP", "CA housing lin", "wine lin",
    "linnerud lin", "linnerud MLP", "protein lin", "protein MLP",
    "BlogFeedback lin", "MNIST CNN",
]
gaps = [0.00, 0.24, 0.07, 0.01, 0.24, 0.65, 0.04, 0.03, 1.44, -0.96]
# MNIST is accuracy gap in points; scale note in caption — use as-is with color

fig, ax = plt.subplots(figsize=(6.8, 3.2))
cols = ["#2a9d8f" if g > 0 else "#e76f51" for g in gaps]
y = np.arange(len(labels))
ax.barh(y, gaps, color=cols)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel(r"AdamW $-$ SR-Adam (positive $=$ SR-Adam better)")
ax.set_title("Real problems: paired gaps (MSE, or accuracy points for MNIST CNN)")
fig.tight_layout()
fig.savefig(PAPER / "real_reg_gaps.pdf")
plt.close(fig)

# --- GNN: grouped bars accuracy
datasets = ["Cora", "CiteSeer", "PubMed"]
adamw_c = [77.9, 63.4, 76.1]
sradam_c = [78.1, 64.1, 76.3]
adamw_n = [62.4, 48.9, 62.9]
sradam_n = [63.2, 49.4, 62.9]

x = np.arange(len(datasets))
w = 0.2
fig, ax = plt.subplots(figsize=(5.8, 3.0))
ax.bar(x - 1.5 * w, adamw_c, w, label="AdamW clean", color="#8ecae6")
ax.bar(x - 0.5 * w, sradam_c, w, label="SR-Adam clean", color="#219ebc")
ax.bar(x + 0.5 * w, adamw_n, w, label="AdamW noisy", color="#ffb703")
ax.bar(x + 1.5 * w, sradam_n, w, label="SR-Adam noisy", color="#fb8500")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("test accuracy (%)")
ax.set_ylim(40, 85)
ax.legend(loc="lower right", ncol=2, fontsize=7)
ax.set_title("Graph node classification (GCN)")
fig.tight_layout()
fig.savefig(PAPER / "gnn_acc.pdf")
plt.close(fig)

# --- win-rate summary strip (method advantage across families)
families = [
    "Dim. sweep\n(noise, $p$ up to 500)",
    "Real linear\nregression",
    "Label noise\n(real / synthetic)",
    "Graphs\n(GCN)",
    "Digits logistic\n+ noise",
    "MNIST CNN",
]
wins = [23 / 24, 12 / 13, 8 / 9, 6 / 6, 1 / 1, 0 / 2]
colors = ["#2a9d8f" if v >= 0.5 else "#e76f51" for v in wins]
fig, ax = plt.subplots(figsize=(6.8, 2.6))
ax.barh(np.arange(len(families)), wins, color=colors)
ax.axvline(0.5, color="k", ls="--", lw=0.8)
ax.set_yticks(np.arange(len(families)))
ax.set_yticklabels(families, fontsize=7)
ax.invert_yaxis()
ax.set_xlim(0, 1.05)
ax.set_xlabel("fraction of settings with SR-Adam $\\geq$ AdamW")
ax.set_title("Where the risk-optimal weight helps")
fig.tight_layout()
fig.savefig(PAPER / "regime_winrate.pdf")
plt.close(fig)

print("wrote", PAPER / "real_reg_gaps.pdf")
print("wrote", PAPER / "gnn_acc.pdf")
print("wrote", PAPER / "regime_winrate.pdf")
