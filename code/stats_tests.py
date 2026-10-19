"""
stats_tests.py — Phase A statistics for the SR-Adam paper.

Diagnostic items covered: b3 (seeds/tests/ties/"distinguishable") and a8 (Table 5 vs Table 8 counts).

Reads the stored results in ../out/*.json (NO new simulation) and writes:

  ../out/stats_tests.md     printable Markdown report
  ../out/stats_tests.json   same content, machine-readable

What it computes, per family of cells:

  * oriented paired difference d = SR-Adam - AdamW after orientation, so that
    d > 0 always favours SR-Adam (regression/loss cells use AdamW - SR-Adam,
    accuracy cells use SR-Adam - AdamW; the stored paired_diff already has this
    sign convention in every source file, see orientation checks below);
  * mean d, standard error of d, two-sided paired t test against 0 at 5%;
  * Wilcoxon signed-rank test on the per-seed differences where they are stored;
  * Holm correction across the cells of each family (applied to the p values
    that are available: Wilcoxon when raw seeds exist, otherwise the t test);
  * win / tie / loss counts with an explicit tie tolerance instead of counting
    ties as wins (the current win-rate figure counts ties as wins).

Two special blocks requested by the diagnostic:

  A3  tab:batch — recompute which batch-size cells are actually significant at
      the 5% two-sided critical value of t with 4 degrees of freedom (2.7764);
      the manuscript calls B=8 (t=2.7) and B=32 (t=1.9) "distinguishable".

  A4  Table 5 vs Table 8 reconciliation — count how many settings
      path4_dim_sweep.json actually holds (4 grids x 6 p = 24), how many of
      them are shown in the printed tab:dim_sweep (3 columns x 6 p = 18), how
      many are wins, and which rows the "23 of 24" claim can refer to.

Known limitation (by design): a few driver rows do not store per-seed values
(paper_tables classification rows, noise_sweep classification rows,
simulation_noise rows before the Phase C edit).  For those cells only the
columns that the stored mean/sd/t allow are reported; the rest is printed as
"NA (seeds not stored)" and will become available after the Phase D re-run with
the edited drivers, which persist *\_all arrays.

Run (no arguments; reads stored JSON only):

    python stats_tests.py
    python stats_tests.py --quick      # skip the Holm family plots, same numbers
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as sps

OUT = Path(__file__).resolve().parent.parent / "out"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------- helpers


def load(name: str):
    p = OUT / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def tie_tol(a: float, s: float) -> float:
    """Tie tolerance: relative to the scale of the two means."""
    return max(1e-9, 1e-6 * max(abs(a), abs(s)))


def holm(pvals: dict) -> dict:
    """Holm step-down correction.  pvals: label -> p (nan allowed, skipped)."""
    items = [(k, v) for k, v in pvals.items() if v is not None and not math.isnan(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running = 0.0
    for i, (k, v) in enumerate(items):
        adj = min(1.0, (m - i) * v)
        running = max(running, adj)      # Holm is monotone
        out[k] = running
    return out


class Cell:
    """One comparison cell of a table."""

    def __init__(self, family, label, adamw, sradam, diffs=None, n_seeds=None,
                 t_stored=None, higher_better=False, source=""):
        self.family = family
        self.label = label
        self.adamw = float(adamw)
        self.sradam = float(sradam)
        self.higher_better = higher_better
        if diffs is not None:
            # diffs must already be oriented so that > 0 favours SR-Adam; the
            # orientation is verified against the stored gap in stats() below
            self.diffs = np.asarray(diffs, dtype=float)
        else:
            self.diffs = None
        self.n = int(n_seeds or (len(self.diffs) if self.diffs is not None else 0))
        self.t_stored = t_stored
        self.source = source
        # orientation check: mean of stored diffs must equal the oriented mean diff
        self.mean_d = (self.sradam - self.adamw) if higher_better else \
                      (self.adamw - self.sradam)

    # derived statistics -------------------------------------------------
    def stats(self):
        res = dict(label=self.label, family=self.family, source=self.source,
                   adamw=self.adamw, sradam=self.sradam,
                   mean_diff=self.mean_d, n_seeds=self.n)
        tol = tie_tol(self.adamw, self.sradam)
        if abs(self.mean_d) <= tol:
            res["outcome"] = "tie"
        else:
            res["outcome"] = "win" if self.mean_d > 0 else "loss"

        raw = None
        if self.diffs is not None and len(self.diffs) >= 3:
            raw = np.asarray(self.diffs, dtype=float)
            # guard: per-seed differences must average to the stored gap
            if not np.isclose(raw.mean(), self.mean_d, atol=max(1e-6, 10 * tol)):
                raw = None
                res["raw_note"] = "per-seed array inconsistent with stored gap; ignored"

        if raw is not None:
            n = len(raw)
            sd = raw.std(ddof=1)
            se = sd / math.sqrt(n)
            t = raw.mean() / se if se > 0 else float("nan")
            p_t = 2 * sps.t.sf(abs(t), n - 1) if se > 0 else float("nan")
            if sd > 0:
                try:
                    w = sps.wilcoxon(raw, zero_method="wilcox", alternative="two-sided")
                    p_w = float(w.pvalue)
                except ValueError:
                    p_w = float("nan")
            else:
                p_w = 1.0 if raw.mean() == 0 else float("nan")
            res.update(se=se, t=t, df=n - 1, t_p=p_t, wilcoxon_p=p_w, raw=True)
        else:
            res["raw"] = False
            if self.t_stored is not None and not math.isnan(self.t_stored):
                t = float(self.t_stored)
                df = max(1, self.n - 1)
                res.update(t=t, df=df, se=(self.mean_d / t if t else None),
                           t_p=2 * sps.t.sf(abs(t), df), wilcoxon_p=None)
            else:
                res.update(t=None, se=None, t_p=None, wilcoxon_p=None,
                           note="seeds not stored; rerun driver (Phase D) for tests")
        return res

    def p_for_holm(self):
        s = self.stats()
        return s.get("wilcoxon_p") if s.get("wilcoxon_p") is not None \
            else s.get("t_p")


# --------------------------------------------------------------------- loaders


def load_families():
    fam: dict = {}

    def add(family, cell):
        fam.setdefault(family, []).append(cell)

    # ---- dimension sweep (Table 8 inline): raw _all arrays present after the
    # ---- Phase C re-run; fall back to the stored t statistic before that
    d = load("path4_dim_sweep.json")
    if d:
        n_meta = len(d.get("meta", {}).get("seeds", []) or [])
        for r in d.get("rows", []):
            diffs, n = None, n_meta
            if "adamw_all" in r and "sradam_all" in r:
                diffs = [a - s for a, s in zip(r["adamw_all"], r["sradam_all"])]
                n = len(diffs)
            elif diffs is None and r.get("t_stat") is None:
                continue            # nothing testable in this row
            p, prob = r.get("p"), r.get("problem", "?")
            key = prob + (f" tau={r['tau']}" if "tau" in r else
                          (f" flip={r['flip']}" if "flip" in r else ""))
            add("dim_sweep", Cell("dim_sweep", f"{key}, p={p}",
                                  r["adamw"], r["sradam"], diffs=diffs,
                                  n_seeds=n, t_stored=r.get("t_stat"),
                                  higher_better=False,
                                  source="path4_dim_sweep.json"))

    # ---- real regression / classification tables
    pt = load("paper_tables.json")
    if pt:
        meta_n = len(pt.get("meta", {}).get("seeds", []) or [])
        for r in pt.get("regression", []):
            if r.get("status") != "ok":
                continue
            diffs = None
            if "adamw_all" in r and "sradam_all" in r:
                diffs = [a - s for a, s in zip(r["adamw_all"], r["sradam_all"])]
            add(f"reg_{r['model']}", Cell(f"reg_{r['model']}",
                                          f"{r['dataset']} ({r['model']})",
                                          r["adamw"], r["sradam"], diffs=diffs,
                                          n_seeds=meta_n, higher_better=False,
                                          source="paper_tables.json"))
        for r in pt.get("classification", []):
            if r.get("status") != "ok" or r.get("model") == "GCN":
                continue
            noise = r.get("label_noise") or 0.0
            fam_lbl = "clf_clean" if noise == 0 else "noise_clf"
            add(fam_lbl, Cell(fam_lbl,
                              f"{r['dataset']} {r['model']} noise={noise}",
                              r["adamw"], r["sradam"], diffs=None,
                              n_seeds=meta_n, higher_better=True,
                              source="paper_tables.json"))

    # ---- label / target noise sweep (Table 10)
    ns = load("noise_sweep.json")
    if ns:
        meta_n = len(ns.get("meta", {}).get("seeds", []) or [])
        for r in ns.get("rows", []):
            metric = r.get("metric", "acc_frac")
            higher = metric != "mse"
            fam_lbl = "noise_clf" if higher else "noise_reg"
            diffs = None
            if "adamw_all" in r and "sradam_all" in r:
                raw_a, raw_s = r["adamw_all"], r["sradam_all"]
                diffs = ([s - a for a, s in zip(raw_a, raw_s)] if higher
                         else [a - s for a, s in zip(raw_a, raw_s)])
            else:
                # noise_sweep stores a paired list under 'paired_all' after Phase C
                if "paired_all" in r:
                    diffs = r["paired_all"]
            add(fam_lbl, Cell(fam_lbl, f"{r['problem']} noise={r['noise']}",
                              r["adamw"], r["sradam"], diffs=diffs,
                              n_seeds=meta_n, higher_better=higher,
                              source="noise_sweep.json"))

    # ---- Monte Carlo optimisation under controlled noise (Table 5)
    sn = load("simulation_noise.json")
    if sn:
        for r in sn.get("quadratic", []):
            add("sim_quad", Cell("sim_quad", f"tau={r['tau']}",
                                 r["adamw"], r["sradam"],
                                 diffs=r.get("paired_all"),
                                 n_seeds=5, t_stored=r.get("t_stat"),
                                 higher_better=False,
                                 source="simulation_noise.json"))
        for r in sn.get("logistic", []):
            add("sim_logistic", Cell("sim_logistic", f"flip={r['flip']}",
                                     r["adamw"], r["sradam"],
                                     diffs=r.get("paired_all"),
                                     n_seeds=3, t_stored=r.get("t_stat"),
                                     higher_better=False,
                                     source="simulation_noise.json"))

    # ---- graphs
    gnn_cells = []
    for name in ("gnn_pilot_cora.json", "gnn_pilot_planetoid.json"):
        g = load(name)
        if not g:
            continue
        for r in g.get("rows", []):
            gnn_cells.append(Cell("gnn", f"{r['dataset']} noise={r['label_noise']}",
                                  r["adamw"], r["sradam"], diffs=None,
                                  n_seeds=None, higher_better=True, source=name))
    for c in gnn_cells:
        add("gnn", c)

    return fam


# ------------------------------------------------------------------- A3 block


def batch_claim_check():
    d = load("simulation_batch.json")
    if not d:
        return ["### A3 — tab:batch significance: simulation_batch.json missing", ""]
    n = 5
    df = n - 1
    tcrit = float(sps.t.ppf(0.975, df))
    lines = [f"### A3 — tab:batch: which cells are 'statistically distinguishable'",
             "",
             f"Paired t over {n} seeds, df={df}, two-sided 5% critical value "
             f"t = {tcrit:.3f}.",
             "",
             "| B | paired diff | t | significant at 5% ? |",
             "|---|---|---|---|"]
    flagged = []
    for r in d:
        t = r.get("t_stat")
        sig = "yes" if (t is not None and abs(t) >= tcrit) else "**no**"
        if sig != "yes":
            flagged.append((r.get("B"), t))
        lines.append(f"| {r['B']} | {r['paired_diff']:+.4f} | {t:+.2f} | {sig} |")
    lines.append("")
    if flagged:
        lines.append("Cells the manuscript must NOT call distinguishable: " +
                     ", ".join(f"B={b} (t={t:+.2f})" for b, t in flagged) + ".")
        lines.append("")
    return lines


# ------------------------------------------------------------------- A4 block


def reconcile_counts():
    d = load("path4_dim_sweep.json")
    if not d:
        return ["### A4 — Table 5 vs Table 8: path4_dim_sweep.json missing", ""]
    rows = d["rows"]
    printed = [r for r in rows
               if (r.get("problem") == "quadratic_gaussian" and r.get("tau") == 1.0)
               or (r.get("problem") == "quadratic_student" and r.get("tau") == 1.0)
               or (r.get("problem") == "logistic_labelnoise" and r.get("flip") == 0.2)]
    excluded = [r for r in rows if r not in printed]

    def desc(r):
        extra = (f"tau={r['tau']}" if "tau" in r else
                 (f"flip={r['flip']}" if "flip" in r else ""))
        return f"{r['problem']} {extra} p={r['p']}"

    def win(r):
        tol = tie_tol(r.get("adamw", 0.0), r.get("sradam", 0.0))
        return "W" if r.get("paired_diff", 0) > tol else (
            "T" if abs(r.get("paired_diff", 0)) <= tol else "L")

    w_all = sum(1 for r in rows if win(r) == "W")
    w_pr = sum(1 for r in printed if win(r) == "W")
    lines = ["### A4 — cell counts: '23 of 24' vs the printed table", "",
             f"| population | cells | wins | ties | losses |",
             f"|---|---|---|---|---|",
             f"| all settings in path4_dim_sweep.json | {len(rows)} | {w_all} | "
             f"{sum(1 for r in rows if win(r)=='T')} | "
             f"{sum(1 for r in rows if win(r)=='L')} |",
             f"| cells printed in tab:dim_sweep (3 columns x 6 p) | {len(printed)} | "
             f"{w_pr} | {sum(1 for r in printed if win(r)=='T')} | "
             f"{sum(1 for r in printed if win(r)=='L')} |", "",
             "Rows NOT shown in the printed table:", ""]
    lines += [f"  - {desc(r)} ({win(r)})" for r in excluded]
    per_p = {}
    for r in printed:
        per_p.setdefault(r["p"], []).append(r)
    lines += ["", "Per-p gaps with the p-normalised gap (diagnostic a8):", "",
              "| p | median gap | gap / p |", "|---|---|---|"]
    for p in sorted(per_p):
        gaps = sorted(x["paired_diff"] for x in per_p[p])
        med = gaps[len(gaps) // 2]
        lines.append(f"| {p} | {med:+.3f} | {med / p:+.5f} |")
    return lines + [""]


# ---------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="same numbers, fewer banners")
    args = ap.parse_args()

    fam = load_families()
    md = ["# SR-Adam — statistical analysis of the stored results (stats_tests.py)",
          "",
          "Positive difference favours SR-Adam in every row. "
          "Ties use tol = max(1e-9, 1e-6 * max|mean|).",
          ""]
    js = dict(families={}, batch_check=[], reconciliation=[])

    for family, cells in fam.items():
        tested = [c for c in cells]
        pvals = {c.label: c.p_for_holm() for c in tested}
        adj = holm(pvals)
        md += [f"## {family} ({len(cells)} cells)", "",
               "| cell | AdamW | SR-Adam | diff | SE | t | p(t) | "
               "p(Wilcoxon) | Holm | outcome |", 
               "|---|---|---|---|---|---|---|---|---|---|"]
        for c in cells:
            s = c.stats()
            fmt = lambda v, k=".4g": ("NA" if v is None or (
                isinstance(v, float) and math.isnan(v)) else format(v, k))
            h = adj.get(c.label)
            md.append("| {lab} | {a} | {sr} | {d} | {se} | {t} | {pt} | {pw} | "
                      "{h} | {o} |".format(
                          lab=s["label"], a=fmt(s["adamw"]), sr=fmt(s["sradam"]),
                          d=fmt(s["mean_diff"], "+.4g"),
                          se=fmt(s.get("se"), ".3g"),
                          t=fmt(s.get("t"), "+.3f"), pt=fmt(s.get("t_p"), ".3g"),
                          pw=fmt(s.get("wilcoxon_p"), ".3g"),
                          h=fmt(h, ".3g"), o=s["outcome"]))
            js["families"].setdefault(family, []).append(s)
        w = sum(1 for c in cells if c.stats()["outcome"] == "win")
        t_ = sum(1 for c in cells if c.stats()["outcome"] == "tie")
        l = sum(1 for c in cells if c.stats()["outcome"] == "loss")
        md += ["", f"**win / tie / loss = {w} / {t_} / {l}** "
                   f"(fraction with wins-only denominator = "
                   f"{w / max(1, w + t_ + l):.2%}; ties excluded from wins).", ""]

    bc = batch_claim_check()
    md += bc
    js["batch_check"] = bc
    rc = reconcile_counts()
    md += rc
    js["reconciliation"] = rc

    text = "\n".join(md)
    (OUT / "stats_tests.md").write_text(text + "\n")
    (OUT / "stats_tests.json").write_text(json.dumps(js, indent=2, default=str))
    print(text)
    print("\nwrote", OUT / "stats_tests.md")
    print("wrote", OUT / "stats_tests.json")


if __name__ == "__main__":
    main()