"""
Dimension sweep (Path 4): how the SR-Adam advantage changes with block size p.

Directly addresses the title question «High Dimensions»: if the method's edge is
a Stein / high-dimensional effect, the paired gap AdamW - SR-Adam should grow
(or at least stay positive) as p increases, in the noise-dominated regime.

Problems (synthetic, controlled — same family as Path 2):
  quadratic_gaussian   exact gradient + N(0, tau^2 I)
  quadratic_student    same, Student-t (nu=5, variance-matched)
  logistic_labelnoise  multinomial logistic + label flips

Grid: p x problem x noise level.  3 seeds.  Lower objective is better;
positive paired_diff = SR-Adam better.

Run:  python -u path4_dim_sweep.py
      python -u path4_dim_sweep.py --smoke
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
import math
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sr_adam import SRAdam
from tune_helper import (add_tuning_args, run_tuning, Stopwatch, gate_stats,
                         EarlyStopper)

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs"
OUT = HERE.parent / "out"
REPORT_DIR = HERE.parent / "reports" / "REPORT_for_Arashi"
for p in (FIGS, OUT, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)

OPTS = ["adamw", "amsgrad", "sradam"]


def build(name, params, lr, wd=0.0, beta1=None, track_stats=False):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "amsgrad":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, amsgrad=True)
    kw = dict(warmup=5, weight_decay=wd, track_stats=track_stats)
    if beta1 is not None:
        kw["betas"] = (beta1, 0.999)
    return SRAdam(params, lr=lr, **kw)


def pilot_quad(cfg, seed, p, tau, steps):
    """Single-optimiser pilot run for --tune (item a7): terminal objective."""
    H = torch.linspace(0.05, 2.0, p)
    torch.manual_seed(seed)
    x_star = torch.randn(p)
    x = torch.zeros(p, requires_grad=True)
    opt = build(cfg["opt"], [x], cfg["lr"] or 2e-2,
                wd=cfg.get("wd") or 0.0, beta1=cfg.get("beta1"))
    gen = torch.Generator().manual_seed(500 + seed)
    for _ in range(steps):
        x.grad = H * (x.detach() - x_star) + tau * torch.randn(p, generator=gen)
        opt.step()
    return float(0.5 * (H * (x.detach() - x_star) ** 2).sum())


def pilot_log(cfg, seed, p, flip, steps, N=4000, K=10, batch=64):
    """Single-optimiser pilot for --tune: validation log-loss (clean val)."""
    lossf = torch.nn.CrossEntropyLoss()
    torch.manual_seed(seed)
    Wt = torch.randn(p, K) / math.sqrt(p)
    X = torch.randn(N, p)
    y = (X @ Wt).argmax(1)
    if flip > 0:
        mask = torch.rand(N) < flip
        y = torch.where(mask, torch.randint(0, K, (N,)), y)
    Xv = torch.randn(2000, p)
    yv = (Xv @ Wt).argmax(1)
    W = torch.zeros(p, K, requires_grad=True)
    opt = build(cfg["opt"], [W], cfg["lr"] or 3e-2,
                wd=cfg.get("wd") or 0.0, beta1=cfg.get("beta1"))
    gen = torch.Generator().manual_seed(600 + seed)
    for _ in range(steps):
        idx = torch.randint(0, N, (batch,), generator=gen)
        opt.zero_grad()
        lossf(X[idx] @ W, y[idx]).backward()
        opt.step()
    with torch.no_grad():
        return float(lossf(Xv @ W, yv))


def student_noise(shape, tau, gen):
    nu = 5.0
    z = torch.randn(shape, generator=gen)
    scale = tau * math.sqrt((nu - 2.0) / nu)
    if len(shape) == 1:
        chi = torch.distributions.Chi2(torch.tensor([nu])).sample((shape[0],)).view(-1)
    else:
        chi = torch.distributions.Chi2(torch.tensor([nu])).sample((shape[0],)).squeeze(-1)
        chi = chi.view(shape[0], *([1] * (len(shape) - 1)))
    return scale * z / (chi.sqrt() + 1e-8)


def paired(per, seeds):
    rec = {}
    for o in OPTS:
        a = np.asarray(per[o], dtype=float)
        rec[o] = float(a.mean())
        rec[o + "_sd"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    d = np.asarray(per["adamw"]) - np.asarray(per["sradam"])
    rec["paired_diff"] = float(d.mean())
    if len(d) > 1 and d.std(ddof=1) > 0:
        rec["t_stat"] = float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))
    else:
        rec["t_stat"] = 0.0
    rec["win_sradam"] = bool(d.mean() > 0)
    return rec


def run_quadratic(p, tau, law, steps, seeds, cfgs=None, timing=False, gate=False):
    cfgs = cfgs or {}
    H = torch.linspace(0.05, 2.0, p)
    per = {o: [] for o in OPTS}
    k_mean, mss, gfs = [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        x_star = torch.randn(p)
        gen = torch.Generator().manual_seed(500 + seed)
        for o in OPTS:
            torch.manual_seed(seed)
            x = torch.zeros(p, requires_grad=True)
            c = cfgs.get(o) or {}
            opt = build(o, [x], c.get("lr") or 2e-2, wd=c.get("wd") or 0.0,
                        beta1=c.get("beta1"),
                        track_stats=(gate and o == "sradam"))
            ks = []
            sw = Stopwatch(timing)
            for _ in range(steps):
                eps = (tau * torch.randn(p, generator=gen) if law == "gaussian"
                       else student_noise((p,), tau, gen))
                x.grad = H * (x.detach() - x_star) + eps
                opt.step()
                sw.tick()
                if o == "sradam" and getattr(opt, "stats", None):
                    ks.append(opt.stats[0]["shrink_weight"])
            per[o].append(float(0.5 * (H * (x.detach() - x_star) ** 2).sum()))
            if timing and sw.ms_per_step:
                mss.append(sw.ms_per_step)
            if gate and ks:
                gs = gate_stats(opt, o)
                if gs.get("gate_open_frac") is not None:
                    gfs.append(gs["gate_open_frac"])
            if ks:
                k_mean.append(float(np.mean(ks)))
    rec = paired(per, seeds)
    rec.update(problem=f"quadratic_{law}", p=p, tau=tau,
               mean_k=float(np.mean(k_mean)) if k_mean else None,
               paired_all=[float(x) for x in
                           (np.asarray(per["adamw"]) -
                            np.asarray(per["sradam"]))])
    if mss:
        rec["ms_per_step"] = float(np.mean(mss))
    if gfs:
        rec["gate_open_frac"] = float(np.mean(gfs))
    return rec


def run_logistic(p, flip, steps, seeds, N=4000, K=10, batch=64,
                 cfgs=None, timing=False, gate=False, patience=0):
    cfgs = cfgs or {}
    lossf = torch.nn.CrossEntropyLoss()
    per = {o: [] for o in OPTS}
    mss, gfs, stops = [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        Wt = torch.randn(p, K) / math.sqrt(p)
        X = torch.randn(N, p)
        y = (X @ Wt).argmax(1)
        if flip > 0:
            mask = torch.rand(N) < flip
            y = torch.where(mask, torch.randint(0, K, (N,)), y)
        Xte = torch.randn(3000, p)
        yte = (Xte @ Wt).argmax(1)
        Xv = torch.randn(2000, p)              # clean validation for early stop
        yv = (Xv @ Wt).argmax(1)
        for o in OPTS:
            torch.manual_seed(seed)
            W = torch.zeros(p, K, requires_grad=True)
            c = cfgs.get(o) or {}
            opt = build(o, [W], c.get("lr") or 3e-2, wd=c.get("wd") or 0.0,
                        beta1=c.get("beta1"),
                        track_stats=(gate and o == "sradam"))
            gen = torch.Generator().manual_seed(600 + seed)
            sw = Stopwatch(timing)
            es = (EarlyStopper(patience, mode="min")
                  if patience else None)
            best_w, stopped = None, None
            for step in range(steps):
                idx = torch.randint(0, N, (batch,), generator=gen)
                opt.zero_grad()
                lossf(X[idx] @ W, y[idx]).backward()
                opt.step()
                sw.tick()
                if es is not None and es.due(step):
                    with torch.no_grad():
                        v = float(lossf(Xv @ W, yv))
                    if es.update(v):
                        best_w = W.detach().clone()
                    if es.should_stop:
                        stopped = step + 1
                        break
            with torch.no_grad():
                if best_w is not None:
                    W.data.copy_(best_w)
                per[o].append(float(lossf(Xte @ W, yte)))
                if gate and getattr(opt, "stats", None):
                    gs = gate_stats(opt, o)
                    if gs.get("gate_open_frac") is not None:
                        gfs.append(gs["gate_open_frac"])
            if timing and sw.ms_per_step:
                mss.append(sw.ms_per_step)
            if stopped:
                stops.append(stopped)
    rec = paired(per, seeds)
    rec.update(problem="logistic_labelnoise", p=p, flip=flip, mean_k=None,
               paired_all=[float(x) for x in
                           (np.asarray(per["adamw"]) -
                            np.asarray(per["sradam"]))])
    if mss:
        rec["ms_per_step"] = float(np.mean(mss))
    if gfs:
        rec["gate_open_frac"] = float(np.mean(gfs))
    if stops:
        rec["early_stop_step"] = float(np.mean(stops))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=None,
                    help="override steps per run (config unification, item a8)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="override the number of seeds (item a8)")
    ap.add_argument("--ps", default=None,
                    help="comma list of p values (item a8)")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="noise scale of the main quadratic grid (item a8)")
    ap.add_argument("--flip", type=float, default=0.2,
                    help="label-flip rate of the logistic grid (item a8)")
    add_tuning_args(ap)
    args = ap.parse_args()
    if args.smoke:
        ps = [10, 50, 200]
        seeds = (1, 2)
        steps = 400
    else:
        ps = [10, 20, 50, 100, 200, 500]
        seeds = (1, 2, 3)
        steps = 800
    if args.steps:
        steps = args.steps
    if args.seeds:
        seeds = tuple(range(1, args.seeds + 1))
    if args.ps:
        ps = [int(x) for x in str(args.ps).split(",")]
    tau_hi, tau_lo, flip = args.tau, 0.1 * args.tau, args.flip

    # ---- Phase C tuning, fair baselines (item a7) ----------------------
    cfgs_q, cfgs_l = {}, {}
    if args.tune:
        QUAD_GRID = {"adamw": [5e-3, 1e-2, 2e-2, 5e-2],
                     "amsgrad": [5e-3, 1e-2, 2e-2, 5e-2],
                     "sradam": [5e-3, 1e-2, 2e-2, 5e-2]}
        LG_GRID = {"adamw": [3e-3, 1e-2, 3e-2, 1e-1],
                   "amsgrad": [3e-3, 1e-2, 3e-2, 1e-1],
                   "sradam": [3e-3, 1e-2, 3e-2, 1e-1]}
        print("=== TUNING (a7): identical grid across optimisers ===", flush=True)
        p_tune = min(ps)
        ts = 400 if args.smoke else 600
        _, cfgs_q = run_tuning(
            args, OPTS,
            lambda cfg, seed: pilot_quad(cfg, seed, p_tune, tau_hi, ts),
            f"path4_quad_t{tau_hi}", default_grids=QUAD_GRID)
        _, cfgs_l = run_tuning(
            args, OPTS,
            lambda cfg, seed: pilot_log(cfg, seed, p_tune, flip, ts),
            f"path4_log_f{flip}", default_grids=LG_GRID)

    rows = []
    t0 = time.time()

    print(f"=== quadratic gaussian, tau={tau_hi} (noise-dominated-ish) ===",
          flush=True)
    for p in ps:
        r = run_quadratic(p, tau_hi, "gaussian", steps, seeds, cfgs=cfgs_q,
                          timing=args.timing, gate=args.gate_log)
        rows.append(r)
        print(f"  p={p:4d}  AdamW={r['adamw']:.4f}  SR={r['sradam']:.4f}  "
              f"Δ={r['paired_diff']:+.4f} (t={r['t_stat']:+.2f})  win={r['win_sradam']}  "
              f"k̄={r['mean_k']}", flush=True)

    print(f"=== quadratic gaussian, tau={tau_lo} (more signal) ===", flush=True)
    for p in ps:
        r = run_quadratic(p, tau_lo, "gaussian", steps, seeds, cfgs=cfgs_q,
                          timing=args.timing, gate=args.gate_log)
        rows.append(r)
        print(f"  p={p:4d}  AdamW={r['adamw']:.4f}  SR={r['sradam']:.4f}  "
              f"Δ={r['paired_diff']:+.4f}  win={r['win_sradam']}", flush=True)

    print(f"=== quadratic student-t, tau={tau_hi} ===", flush=True)
    for p in ps:
        r = run_quadratic(p, tau_hi, "student", steps, seeds, cfgs=cfgs_q,
                          timing=args.timing, gate=args.gate_log)
        rows.append(r)
        print(f"  p={p:4d}  AdamW={r['adamw']:.4f}  SR={r['sradam']:.4f}  "
              f"Δ={r['paired_diff']:+.4f}  win={r['win_sradam']}", flush=True)

    print(f"=== logistic label flip={flip} ===", flush=True)
    for p in ps:
        r = run_logistic(p, flip, steps, seeds, cfgs=cfgs_l,
                         timing=args.timing, gate=args.gate_log,
                         patience=args.early_stop)
        rows.append(r)
        print(f"  p={p:4d}  AdamW={r['adamw']:.4f}  SR={r['sradam']:.4f}  "
              f"Δ={r['paired_diff']:+.4f}  win={r['win_sradam']}", flush=True)

    payload = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  seconds=time.time() - t0, smoke=args.smoke,
                  ps=ps, seeds=list(seeds), steps=steps,
                  tau_hi=tau_hi, tau_lo=tau_lo, flip=flip,
                  tuned_quadratic=(cfgs_q or None),
                  tuned_logistic=(cfgs_l or None),
                  timing=args.timing, gate_log=args.gate_log,
                  early_stop=args.early_stop,
                  note="positive paired_diff = SR-Adam better (lower loss)"),
        rows=rows,
    )
    (OUT / "path4_dim_sweep.json").write_text(json.dumps(payload, indent=2))
    (REPORT_DIR / "path4_dim_sweep.json").write_text(json.dumps(payload, indent=2))

    # figure: Δ vs p for each problem/noise
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    axl = [
        ("quadratic_gaussian", tau_hi, f"Quadratic Gaussian τ={tau_hi:g}",
         axes[0, 0]),
        ("quadratic_gaussian", tau_lo, f"Quadratic Gaussian τ={tau_lo:g}",
         axes[0, 1]),
        ("quadratic_student", tau_hi, f"Quadratic Student-t τ={tau_hi:g}",
         axes[1, 0]),
        ("logistic_labelnoise", flip, f"Logistic flip={flip:g}", axes[1, 1]),
    ]
    for problem, noise, title, ax in axl:
        series = []
        for r in rows:
            if r.get("problem") != problem:
                continue
            key = "tau" if "tau" in r else "flip"
            if abs(r.get(key, -1) - noise) < 1e-9:
                series.append((r["p"], r["paired_diff"]))
        series.sort()
        if series:
            ax.plot([p for p, _ in series], [d for _, d in series], "o-", ms=4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("p (block / feature dimension)")
        ax.set_ylabel("AdamW − SR-Adam")
    fig.suptitle("Dimension sweep: positive = SR-Adam better", fontsize=10)
    fig.tight_layout()
    for dest in (FIGS / "path4_dim_sweep.png", REPORT_DIR / "path4_dim_sweep.png",
                 FIGS / "path4_dim_sweep.pdf", REPORT_DIR / "path4_dim_sweep.pdf"):
        try:
            fig.savefig(dest)
        except PermissionError:
            pass
    plt.close(fig)

    wins = sum(1 for r in rows if r["win_sradam"])
    print(f"\n{wins}/{len(rows)} settings: SR-Adam better", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
