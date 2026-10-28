"""
make_paper_barfigs.py — Phase A, rebuilt data-driven.

Old version: every number (gaps, accuracies, win rates) was hard-coded in this
file and the win-rate strip counted ties as wins.  New version reads only the
stored results in ../out/*.json and writes:

  ../paper/real_reg_gaps.pdf   paired gaps: real regression + clean GCN rows
  ../paper/gnn_acc.pdf         GCN accuracy, clean vs noisy labels
  ../paper/regime_winrate.pdf  win-rate strip, ties reported separately
  ../out/winrate_summary.json  same numbers machine-readable (used by stats_tests)

Tie rule: two methods tie when |SR-Adam - AdamW| <= tol with
tol = max(1e-9, 1e-6 * max(|AdamW|, |SR-Adam|)).  Ties are NOT counted as wins;
the bar shows wins / total and the annotation prints "W/T/L".

Populations (same five families as the manuscript caption):

  dim. sweep         out/path4_dim_sweep.json          rows = 6 grids x 7 p = 42
  real linear reg    out/paper_tables.json regression  model == "linear"
  label/target noise out/noise_sweep.json              all rows (19)
  graphs (GCN)       out/gnn_pilot_*.json              all rows (7)
  tabular clean clf  out/paper_tables.json classification, label_noise == 0,
                     non-GCN rows (5)

Run:

    python make_paper_barfigs.py
    python make_paper_barfigs.py --check    # numbers only, no PDFs (smoke)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
PAPER = HERE.parent / "paper"
FIGS = HERE.parent / "figs"
PAPER.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "figure.dpi": 200, "savefig.bbox": "tight",
})


def load(name):
    p = OUT / name
    if not p.exists():
        print(f"  MISSING {p.name} — skipped")
        return None
    return json.loads(p.read_text())


def tol_of(a, s):
    return max(1e-9, 1e-6 * max(abs(a), abs(s)))


def outcome(diff, a, s):
    """'win' / 'tie' / 'loss' for SR-Adam, positive diff favours SR-Adam."""
    t = tol_of(a, s)
    if abs(diff) <= t:
        return "tie"
    return "win" if diff > 0 else "loss"


def tally(cells):
    w = sum(1 for _, o in cells if o == "win")
    t = sum(1 for _, o in cells if o == "tie")
    l = sum(1 for _, o in cells if o == "loss")
    return w, t, l


# ----------------------------------------------------------------- families


def family_dim_sweep():
    d = load("path4_dim_sweep.json")
    cells = []
    if d:
        for r in d["rows"]:
            key = r["problem"] + (f" tau={r['tau']}" if "tau" in r else
                                  (f" flip={r['flip']}" if "flip" in r else ""))
            label = f"{key}, p={r['p']}"
            cells.append((label, outcome(r["paired_diff"], r["adamw"], r["sradam"])))
    return "Dimension sweep", cells


def family_real_linear():
    d = load("paper_tables.json")
    cells = []
    if d:
        for r in d.get("regression", []):
            if r.get("status") == "ok" and r.get("model") == "linear":
                cells.append((f"{r['dataset']} lin",
                              outcome(r["paired_diff"], r["adamw"], r["sradam"])))
    return "Real linear\nregression", cells


def family_noise():
    d = load("noise_sweep.json")
    cells = []
    if d:
        for r in d.get("rows", []):
            cells.append((f"{r['problem']} n={r['noise']}",
                          outcome(r["paired_diff"], r["adamw"], r["sradam"])))
    return "Label / target\nnoise", cells


def gnn_rows():
    rows = []
    for name in ("gnn_pilot_cora.json", "gnn_pilot_planetoid.json"):
        d = load(name)
        if d:
            rows.extend(d.get("rows", []))
    return rows


def family_graphs():
    cells = []
    for r in gnn_rows():
        cells.append((f"{r['dataset']} n={r['label_noise']}",
                      outcome(r["paired_diff"], r["adamw"], r["sradam"])))
    return "Graphs\n(GCN)", cells


def family_clf_clean():
    d = load("paper_tables.json")
    cells = []
    if d:
        for r in d.get("classification", []):
            if r.get("status") != "ok" or r.get("model") == "GCN":
                continue
            if (r.get("label_noise") or 0.0) != 0.0:
                continue
            diff = r["sradam"] - r["adamw"]     # accuracy: positive favours SR
            cells.append((f"{r['dataset']} {r['model']}",
                          outcome(diff, r["adamw"], r["sradam"])))
    return "Tabular\nclassification", cells


FAMILIES = [family_dim_sweep, family_real_linear, family_noise,
            family_graphs, family_clf_clean]


# ------------------------------------------------------------------ figures


def winrate_fig(summary, write_pdf=True):
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ys = np.arange(len(summary))
    fracs = [r["wins"] / r["total"] for r in summary]
    # Green when SR-Adam wins at least half of the settings (the rule the
    # pre-diagnostic figure used).  The old condition
    #     wins >= total - losses     (equivalently: ties == 0)
    # painted the Graphs bar red merely because PubMed tied once, while the
    # dimension sweep (7 wins, 21 losses) stayed green — the opposite of
    # what the bar length says.
    colors = ["#2a9d8f" if f >= 0.5 else "#e76f51" for f in fracs]
    ax.barh(ys, fracs, color=colors)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["name"] for r in summary], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("fraction of settings with SR-Adam $>$ AdamW "
                  "(ties excluded from wins)")
    ax.set_title("Where the risk-optimal weight helps")
    for y, r, f in zip(ys, summary, fracs):
        ax.text(1.02, y, f"{r['wins']}W/{r['ties']}T/{r['losses']}L",
                va="center", fontsize=7)
    fig.tight_layout()
    if write_pdf:
        fig.savefig(PAPER / "regime_winrate.pdf")
        print("wrote", PAPER / "regime_winrate.pdf")
    plt.close(fig)


def real_reg_gaps_fig(write_pdf=True):
    labels, gaps, sources = [], [], []
    d = load("paper_tables.json")
    if d:
        for r in d.get("regression", []):
            if r.get("status") == "ok":
                labels.append(f"{r['dataset'][:14]} {r['model']}")
                gaps.append(r["paired_diff"] / 1.0)
                sources.append("reg")
    for r in gnn_rows():
        if (r.get("label_noise") or 0.0) == 0.0:
            labels.append(f"{r['dataset']} GCN")
            gaps.append(100.0 * r["paired_diff"])   # accuracy points
            sources.append("acc")
    if not labels:
        print("  no rows for real_reg_gaps — skipped")
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    cols = ["#2a9d8f" if g > 0 else "#e76f51" for g in gaps]
    y = np.arange(len(labels))
    ax.barh(y, gaps, color=cols)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{l}{'  [pp]' if s == 'acc' else ''}"
         for l, s in zip(labels, sources)], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"AdamW $-$ SR-Adam (positive $=$ SR-Adam better; "
                  "\n[pp] = accuracy points)")
    ax.set_title("Real problems: paired gaps (MSE for regression; "
                 "accuracy points for graphs)")
    fig.tight_layout()
    if write_pdf:
        fig.savefig(PAPER / "real_reg_gaps.pdf")
        print("wrote", PAPER / "real_reg_gaps.pdf")
    plt.close(fig)


def gnn_acc_fig(write_pdf=True):
    rows = gnn_rows()
    if not rows:
        print("  no GNN rows — skipped gnn_acc.pdf")
        return
    datasets = sorted({r["dataset"] for r in rows})
    x = np.arange(len(datasets))
    w = 0.2
    clean_a, clean_s, noise_a, noise_s = [], [], [], []
    for ds in datasets:
        rs = [r for r in rows if r["dataset"] == ds]
        cl = [r for r in rs if (r.get("label_noise") or 0.0) == 0.0]
        nz = [r for r in rs if (r.get("label_noise") or 0.0) != 0.0]
        nz = sorted(nz, key=lambda r: r["label_noise"])[0] if nz else None
        cl = cl[0] if cl else None
        clean_a.append(100 * cl["adamw"] if cl else np.nan)
        clean_s.append(100 * cl["sradam"] if cl else np.nan)
        noise_a.append(100 * nz["adamw"] if nz else np.nan)
        noise_s.append(100 * nz["sradam"] if nz else np.nan)
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.bar(x - 1.5 * w, clean_a, w, label="AdamW clean", color="#8ecae6")
    ax.bar(x - 0.5 * w, clean_s, w, label="SR-Adam clean", color="#219ebc")
    ax.bar(x + 0.5 * w, noise_a, w, label="AdamW noisy", color="#ffb703")
    ax.bar(x + 1.5 * w, noise_s, w, label="SR-Adam noisy", color="#fb8500")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("test accuracy (%)")
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.set_title("Graph node classification (GCN)")
    fig.tight_layout()
    if write_pdf:
        fig.savefig(PAPER / "gnn_acc.pdf")
        print("wrote", PAPER / "gnn_acc.pdf")
    plt.close(fig)


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="print the numbers only, do not write PDFs (smoke test)")
    args = ap.parse_args()
    write = not args.check

    summary = []
    print("family                      wins  ties  losses  total  fraction(wins)")
    for fn in FAMILIES:
        name, cells = fn()
        if not cells:
            print(f"{name[:28]:28s}  (no data skipped)")
            continue
        w, t, l = tally(cells)
        rec = dict(name=name, wins=w, ties=t, losses=l, total=len(cells),
                   fraction=w / len(cells),
                   cells=[dict(label=k, outcome=o) for k, o in cells])
        summary.append(rec)
        print(f"{name.replace(chr(10), ' ')[:28]:28s}  {w:4d}  {t:4d}  {l:5d}  "
              f"{len(cells):5d}  {w / len(cells):.3f}")

    (OUT / "winrate_summary.json").write_text(
        json.dumps(dict(summary=summary), indent=2))
    print("wrote", OUT / "winrate_summary.json")

    if write:
        winrate_fig(summary)
        real_reg_gaps_fig()
        gnn_acc_fig()
    else:
        print("--check: PDFs not written")


if __name__ == "__main__":
    main()