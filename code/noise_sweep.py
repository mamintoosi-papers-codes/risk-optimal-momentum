"""
Label-noise sweep for the paper: how the SR-Adam gap changes with noise.

Short CPU-friendly grid (override with --steps/--seeds for the GPU box).

Problems
--------
  digits + logistic, label noise {0, 0.1, 0.2, 0.3, 0.5}
  breast_cancer + logistic, same noise grid
  wine quality regression + linear, target noise {0, 0.5, 1.0}

Outputs
-------
  ../out/noise_sweep.json
  ../paper/tab_noise.tex
  ../paper/fig_noise_sweep.pdf

Run:  python -u noise_sweep.py
      python -u noise_sweep.py --seeds 5 --steps 800
"""

from __future__ import annotations

import sys

# Windows redirected stdio uses the ANSI codepage (cp1252 here), which cannot
# encode printed Unicode (Δ, k̄, …).  Keep non-interactive log output working.
if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure") and not sys.stderr.isatty():
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from paper_tables import load_clf, load_reg, split_clf, run_logistic_clf, run_linear
from path3_uci import OPTS, paired, split_scale
from dataset_names import pretty
from tune_helper import add_tuning_args, run_tuning

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
PAPER = HERE.parent / "paper"
for p in (OUT, PAPER):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    add_tuning_args(ap)
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.steps = 2, 250
    seeds = list(range(1, args.seeds + 1))
    noise_grid = [0.0, 0.1, 0.2, 0.3, 0.5]
    t0 = time.time()
    rows = []

    # ---- Phase C helpers (fair baselines, item a7) ----------------------
    def _carve(Xtr, ytr, frac=0.2, seed=0):
        n = int(Xtr.shape[0])
        m = max(1, int(n * frac))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        return Xtr[perm[m:]], ytr[perm[m:]], Xtr[perm[:m]], ytr[perm[:m]]

    def _ow(cfg, o, a):
        kw = {}
        if cfg.get("wd"):
            kw["wd"] = cfg["wd"]
        if cfg.get("beta1") is not None:
            kw["beta1"] = cfg["beta1"]
        if getattr(a, "gate_log", False) and o == "sradam":
            kw["track_stats"] = True
        return kw

    def _sel(cfgs, o, default):
        c = cfgs.get(o) or {}
        return c.get("lr") or default

    def _rec(mss, gfs, st):
        if st.get("ms_per_step"):
            mss.append(st["ms_per_step"])
        if st.get("gate_open_frac") is not None:
            gfs.append(st["gate_open_frac"])

    def _attach(rec, mss, gfs):
        if mss:
            rec["ms_per_step"] = float(np.mean(mss))
        if gfs:
            rec["gate_open_frac"] = float(np.mean(gfs))
        return rec

    cfg_clf, cfg_reg = {}, {}
    if args.tune:
        print("=== TUNING (a7): identical grid across optimisers ===", flush=True)
        CLF_GRID = {"sgdm": [0.01, 0.03, 0.1],
                    "adamw": [1e-2, 3e-2, 5e-2, 1e-1],
                    "amsgrad": [1e-2, 3e-2, 5e-2, 1e-1],
                    "sradam": [1e-2, 3e-2, 5e-2, 1e-1]}
        REG_GRID = {"sgdm": [0.03, 0.1, 0.3],
                    "adamw": [1e-2, 3e-2, 1e-1],
                    "amsgrad": [1e-2, 3e-2, 1e-1],
                    "sradam": [1e-2, 3e-2, 1e-1]}
        tune_steps = 200 if args.quick else min(args.steps, 400)
        try:
            Xd, yd = load_clf("digits")
        except Exception:
            Xd = yd = None
        if Xd is not None:
            def pilot_clf(cfg, seed, Xd=Xd, yd=yd):
                Xtr, ytr, Xte, yte = split_clf(Xd, yd, 0.3, seed)
                K0 = int(ytr.max()) + 1
                return -run_logistic_clf(Xtr, ytr, Xte, yte, cfg["opt"],
                                          tune_steps, cfg["lr"] or 5e-2,
                                          64, seed, K0,
                                          **_ow(cfg, cfg["opt"], args))
            _, cfg_clf = run_tuning(args, OPTS, pilot_clf, "noise_sweep_clf",
                                    lower_is_better=False,
                                    default_grids=CLF_GRID)
        try:
            Xr, yr = load_reg("wine_quality")
        except Exception:
            Xr = yr = None
        if Xr is not None:
            def pilot_reg(cfg, seed, Xr=Xr, yr=yr):
                Xtr, ytr, Xte, yte, _ = split_scale(Xr, yr, 0.3, seed)
                return run_linear(Xtr, ytr, Xte, yte, cfg["opt"],
                                  tune_steps, cfg["lr"] or 3e-2, 64, seed,
                                  **_ow(cfg, cfg["opt"], args))
            _, cfg_reg = run_tuning(args, OPTS, pilot_reg, "noise_sweep_reg",
                                    default_grids=REG_GRID)
    need_st = bool(args.timing or args.early_stop or args.gate_log)

    print("=== digits logistic ===", flush=True)
    X, y = load_clf("digits")
    K = int(y.max()) + 1
    for noise in noise_grid:
        per = {o: [] for o in OPTS}
        mss, gfs = [], []
        for seed in seeds:
            Xtr, ytr, Xte, yte = split_clf(X, y, 0.3, seed)
            val = None
            if args.early_stop > 0 and Xtr.shape[0] > 20:
                Xtr, ytr, Xv, yv = _carve(Xtr, ytr, 0.2, seed)
                val = (Xv, yv)
            if noise > 0:
                gen = torch.Generator().manual_seed(900 + seed)
                m = torch.rand(ytr.shape[0]) < noise
                ytr = ytr.clone()
                ytr[m] = torch.randint(0, K, (int(m.sum()),), generator=gen)
            for o in OPTS:
                st = {} if need_st else None
                per[o].append(run_logistic_clf(
                    Xtr, ytr, Xte, yte, o, args.steps, _sel(cfg_clf, o, 5e-2),
                    64, seed, K, val=val, patience=args.early_stop,
                    timing=args.timing, stats_out=st,
                    **_ow(cfg_clf.get(o) or {}, o, args)))
                if need_st:
                    _rec(mss, gfs, st)
        a = np.asarray(per["adamw"])
        s = np.asarray(per["sradam"])
        rec = dict(problem="digits_logistic", noise=float(noise), metric="acc_frac",
                   adamw=float(a.mean()), sradam=float(s.mean()),
                   paired_diff=float(s.mean() - a.mean()),
                   adamw_all=[float(x) for x in per["adamw"]],
                   sradam_all=[float(x) for x in per["sradam"]],
                   paired_all=[float(x) for x in (s - a)],
                   win_sradam=bool(s.mean() > a.mean()),
                   tuned=(cfg_clf if args.tune else None))
        _attach(rec, mss, gfs)
        rows.append(rec)
        print(f"  noise={noise:.1f}  AdamW={100*a.mean():.1f}  SR={100*s.mean():.1f}  "
              f"Δ={100*rec['paired_diff']:+.1f}pp", flush=True)

    print("=== Breast Cancer logistic ===", flush=True)
    X, y = load_clf("breast_cancer")
    K = int(y.max()) + 1
    for noise in noise_grid:
        per = {o: [] for o in OPTS}
        mss, gfs = [], []
        for seed in seeds:
            Xtr, ytr, Xte, yte = split_clf(X, y, 0.3, seed)
            val = None
            if args.early_stop > 0 and Xtr.shape[0] > 20:
                Xtr, ytr, Xv, yv = _carve(Xtr, ytr, 0.2, seed)
                val = (Xv, yv)
            if noise > 0:
                gen = torch.Generator().manual_seed(910 + seed)
                m = torch.rand(ytr.shape[0]) < noise
                ytr = ytr.clone()
                ytr[m] = torch.randint(0, K, (int(m.sum()),), generator=gen)
            for o in OPTS:
                st = {} if need_st else None
                per[o].append(run_logistic_clf(
                    Xtr, ytr, Xte, yte, o, args.steps, _sel(cfg_clf, o, 5e-2),
                    32, seed, K, val=val, patience=args.early_stop,
                    timing=args.timing, stats_out=st,
                    **_ow(cfg_clf.get(o) or {}, o, args)))
                if need_st:
                    _rec(mss, gfs, st)
        a = np.asarray(per["adamw"])
        s = np.asarray(per["sradam"])
        rec = dict(problem="breast_cancer_logistic", noise=float(noise), metric="acc_frac",
                   adamw=float(a.mean()), sradam=float(s.mean()),
                   paired_diff=float(s.mean() - a.mean()),
                   adamw_all=[float(x) for x in per["adamw"]],
                   sradam_all=[float(x) for x in per["sradam"]],
                   paired_all=[float(x) for x in (s - a)],
                   win_sradam=bool(s.mean() > a.mean()),
                   tuned=(cfg_clf if args.tune else None))
        _attach(rec, mss, gfs)
        rows.append(rec)
        print(f"  noise={noise:.1f}  AdamW={100*a.mean():.1f}  SR={100*s.mean():.1f}  "
              f"Δ={100*rec['paired_diff']:+.1f}pp", flush=True)

    print("=== regression (target noise) ===", flush=True)
    for rname in ("wine_quality", "diabetes", "california_housing"):
        X, y = load_reg(rname)
        for noise in [0.0, 0.5, 1.0]:
            per = {o: [] for o in OPTS}
            mss, gfs = [], []
            for seed in seeds:
                Xtr, ytr, Xte, yte, ysd = split_scale(X, y, 0.3, seed)
                val = None
                if args.early_stop > 0 and Xtr.shape[0] > 20:
                    Xtr, ytr, Xv, yv = _carve(Xtr, ytr, 0.2, seed)
                    val = (Xv, yv)
                if noise > 0:
                    gen = torch.Generator().manual_seed(920 + seed)
                    ytr = ytr + noise * torch.randn(ytr.shape, generator=gen)
                for o in OPTS:
                    st = {} if need_st else None
                    per[o].append(run_linear(
                        Xtr, ytr, Xte, yte, o, args.steps, _sel(cfg_reg, o, 3e-2), 64, seed,
                        val=val, patience=args.early_stop,
                        timing=args.timing, stats_out=st,
                        **_ow(cfg_reg.get(o) or {}, o, args)))
                    if need_st:
                        _rec(mss, gfs, st)
            rec = paired(per)
            rec.update(problem=f"{rname}_linear", noise=float(noise), metric="mse",
                       paired_all=[float(x) for x in
                                   (np.asarray(per["adamw"]) -
                                    np.asarray(per["sradam"]))],
                       tuned=(cfg_reg if args.tune else None))
            _attach(rec, mss, gfs)
            rows.append(rec)
            print(f"  {rname} sigma={noise:.1f}  Δ_MSE={rec['paired_diff']:+.4f}  "
                  f"win={rec['win_sradam']}", flush=True)

    # --- figure: two panels (regression first, then classification)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax2 = axes[0]
    for rname, lab, st in (
        ("wine_quality_linear", "Wine Quality", "o-"),
        ("diabetes_linear", "Diabetes", "s--"),
        ("california_housing_linear", "California Housing", "^-."),
    ):
        xs = [r["noise"] for r in rows if r["problem"] == rname]
        ys = [r["paired_diff"] for r in rows if r["problem"] == rname]
        ax2.plot(xs, ys, st, ms=4, label=lab)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("target-noise scale $\\sigma$")
    ax2.set_ylabel("MSE gap: AdamW $-$ SR-Adam")
    ax2.set_title("Regression (linear)")
    ax2.legend(fontsize=7)

    ax = axes[1]
    for key, lab, st in (
        ("digits_logistic", "Digits (logistic)", "o-"),
        ("breast_cancer_logistic", "Breast Cancer (logistic)", "s--"),
    ):
        xs = [r["noise"] for r in rows if r["problem"] == key]
        ys = [100 * r["paired_diff"] for r in rows if r["problem"] == key]
        ax.plot(xs, ys, st, ms=4, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("training label-noise rate")
    ax.set_ylabel("acc. gap: SR-Adam $-$ AdamW (pp)")
    ax.set_title("Classification")
    ax.legend(fontsize=7)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        try:
            fig.savefig(PAPER / f"fig_noise_sweep.{ext}")
        except PermissionError:
            pass
    plt.close(fig)

    # --- LaTeX table: regression block first, then classification
    lines = [
        r"% Auto-generated by noise_sweep.py",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Problem & Noise & AdamW & SR-Adam & Gap \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Regression (linear), target noise on $y$}} \\",
    ]
    reg = [r for r in rows if r.get("metric") == "mse"]
    clf = [r for r in rows if r.get("metric") != "mse"]

    def fmt_reg(r):
        gap = r["paired_diff"]
        gap_s = f"${gap:+.4f}$"
        if gap > 0:
            gap_s = r"$\bm{" + f"{gap:+.4f}" + r"}$"
        return (
            f"{pretty(r['problem'])} & {r['noise']:.1f} & "
            f"{r['adamw']:.4f} & {r['sradam']:.4f} & {gap_s} \\\\"
        )

    def fmt_clf(r):
        gap_pp = 100 * r["paired_diff"]
        gap_s = f"${gap_pp:+.1f}$pp"
        if gap_pp > 0:
            gap_s = r"$\bm{" + f"{gap_pp:+.1f}" + r"}$pp"
        return (
            f"{pretty(r['problem'])} & {r['noise']:.1f} & "
            f"{100*r['adamw']:.1f} & {100*r['sradam']:.1f} & {gap_s} \\\\"
        )

    for r in reg:
        lines.append(fmt_reg(r))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\emph{Classification, training-label noise}} \\")
    for r in clf:
        lines.append(fmt_clf(r))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tab_noise.tex").write_text("\n".join(lines) + "\n")

    payload = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             seconds=time.time() - t0, seeds=seeds, steps=args.steps),
                   rows=rows)
    (OUT / "noise_sweep.json").write_text(json.dumps(payload, indent=2))
    print("wrote", OUT / "noise_sweep.json", flush=True)
    print("wrote", PAPER / "tab_noise.tex", flush=True)
    print("wrote", PAPER / "fig_noise_sweep.pdf", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
