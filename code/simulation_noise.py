"""
Optimisation under controlled gradient-noise levels  (Table 5 / fig:noise).

The theory says the Stein weight buys most where the mini-batch gradient is
noise-dominated, that is where the squared distance between the current gradient
and the historical average is small relative to the total noise variance of the
block.  The studies below vary that ratio directly rather than inferring it.

Problem 1, a noisy quadratic.  The objective is (1/2)(theta - theta*)' H (theta - theta*)
with H diagonal and a spread of curvatures, and the observed gradient is the exact
gradient plus isotropic noise of standard deviation tau.  Sweeping tau traces the
whole range from the signal-dominated to the noise-dominated regime while holding
everything else fixed.

Problem 2, logistic regression with label noise.  A convex problem with genuine mini-batch
noise, where the noise level is controlled by the label-flip rate rather than by an
artificial additive term.

Reported for each setting: terminal objective (problem 1) or test log-loss (problem 2),
averaged over seeds, together with a paired comparison against AdamW.

Phase C additions (diagnostic item a7 and friends), all OFF by default so the
original behaviour is byte-for-byte preserved:

  --tune / --lr-grid / --beta1-grid / --wd-grid
        per-noise-step grid search over an identical grid for every optimiser
        (the diagnostic's complaint is exactly the missing lr and beta1 grids);
        the chosen settings are stored in the rows and in
        ../out/simulation_noise_tune*.json.
  --early-stop PATIENCE   validation-early-stopping control arm; a clean
        validation set of 2000 synthetic points is held out for that purpose.
  --timing                 per-step wall-clock time column (complexity claim).
  --gate-log               SR-Adam gate-open statistics per cell.
  --steps / --seeds / --taus / --flips
        configuration overrides so this script and path4_dim_sweep.py can be
        run with ONE configuration (item a8 reconciliation).

Every row now also stores *\_all per-seed values (paired_all) so Wilcoxon/Holm
tests in stats_tests.py become possible.

Run:
  python -u simulation_noise.py                  # original behaviour
  python -u simulation_noise.py --tune --timing  # Phase C experiments
  python -u simulation_noise.py --smoke          # short pilot
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sr_adam import SRAdam
from tune_helper import (add_tuning_args, run_tuning, Stopwatch, EarlyStopper,
                         gate_stats)

FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")
OUT = os.path.join(os.path.dirname(__file__), "..", "out")

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "lines.linewidth": 1.3,
})

OPTS = ["sgdm", "adamw", "amsgrad", "sradam"]
LABEL = {"sgdm": "SGD with momentum", "adamw": "AdamW",
         "amsgrad": "AdamW (AMSGrad)", "sradam": "SR-Adam"}

DEFAULT_LR = {  # step sizes used originally; used whenever --tune is off
    "quad": {o: (5e-2 if o == "sgdm" else 2e-2) for o in OPTS},
    "log": {o: (1e-1 if o == "sgdm" else 3e-2) for o in OPTS},
}
QUAD_GRID = {o: ([0.02, 0.05, 0.1, 0.2] if o == "sgdm"
                 else [5e-3, 1e-2, 2e-2, 5e-2]) for o in OPTS}
LG_GRID = {o: ([0.03, 0.1, 0.3] if o == "sgdm"
               else [3e-3, 1e-2, 3e-2, 1e-1]) for o in OPTS}

VAL_N = 2000          # clean synthetic validation set for early stopping


def build(name, params, lr, wd=0.0, beta1=None, track_stats=False):
    """wd default 0 and beta1 default reproduce the original optimisers."""
    if name == "sgdm":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "amsgrad":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, amsgrad=True)
    kw = dict(warmup=5, weight_decay=wd, track_stats=track_stats)
    if beta1 is not None:
        kw["betas"] = (beta1, 0.999)
    return SRAdam(params, lr=lr, **kw)


def _lr(chosen, o, defaults):
    c = (chosen or {}).get(o) or {}
    return c.get("lr") or defaults[o]


def _ow(chosen, o, gate=False):
    c = (chosen or {}).get(o) or {}
    kw = {}
    if c.get("wd"):
        kw["wd"] = c["wd"]
    if c.get("beta1") is not None:
        kw["beta1"] = c["beta1"]
    if gate and o == "sradam":
        kw["track_stats"] = True
    return kw


# --------------------------------------------------------- tuning pilots (a7)
def pilot_quad(cfg, seed, tau, steps, p=400):
    H = torch.linspace(0.05, 2.0, p)
    torch.manual_seed(seed)
    x_star = torch.randn(p)
    x = torch.zeros(p, requires_grad=True)
    opt = build(cfg["opt"], [x], cfg["lr"] or 2e-2,
                wd=cfg.get("wd") or 0.0, beta1=cfg.get("beta1"))
    gen = torch.Generator().manual_seed(777 + seed)
    for _ in range(steps):
        x.grad = H * (x.detach() - x_star) + tau * torch.randn(p, generator=gen)
        opt.step()
    return float(0.5 * (H * (x.detach() - x_star) ** 2).sum())


def pilot_log(cfg, seed, flip, steps, d=200, K=10, N=20_000, B=64):
    lossf = torch.nn.CrossEntropyLoss()
    torch.manual_seed(seed)
    Wt = torch.randn(d, K) / np.sqrt(d)
    X = torch.randn(N, d)
    y = (X @ Wt).argmax(1)
    if flip > 0:
        mask = torch.rand(N) < flip
        y = torch.where(mask, torch.randint(0, K, (N,)), y)
    Xv = torch.randn(VAL_N, d)               # clean validation log-loss
    yv = (Xv @ Wt).argmax(1)
    W = torch.zeros(d, K, requires_grad=True)
    opt = build(cfg["opt"], [W], cfg["lr"] or 3e-2,
                wd=cfg.get("wd") or 0.0, beta1=cfg.get("beta1"))
    gen = torch.Generator().manual_seed(31 + seed)
    for _ in range(steps):
        idx = torch.randint(0, N, (B,), generator=gen)
        opt.zero_grad()
        lossf(X[idx] @ W, y[idx]).backward()
        opt.step()
    with torch.no_grad():
        return float(lossf(Xv @ W, yv))


# -------------------------------------------------------------------- studies
def quadratic(taus=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), p=400, steps=3000,
              seeds=(1, 2, 3, 4, 5), args=None):
    H = torch.linspace(0.05, 2.0, p)
    timing = bool(getattr(args, "timing", False))
    gate = bool(getattr(args, "gate_log", False))
    do_tune = bool(getattr(args, "tune", False))
    pilot_steps = 400 if getattr(args, "smoke", False) else 600
    rows = []
    for tau in taus:
        chosen = {}
        if do_tune:
            _, chosen = run_tuning(
                args, OPTS,
                lambda cfg, seed, tau=tau: pilot_quad(cfg, seed, tau,
                                                      pilot_steps),
                f"simulation_noise_quad_t{tau:g}", default_grids=QUAD_GRID)
        rec = dict(tau=tau)
        per = {o: [] for o in OPTS}
        mss, gfs = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            x_star = torch.randn(p)
            for o in OPTS:
                torch.manual_seed(seed)
                x = torch.zeros(p, requires_grad=True)
                opt = build(o, [x], _lr(chosen, o, DEFAULT_LR["quad"]),
                            **_ow(chosen, o, gate=gate))
                gen = torch.Generator().manual_seed(777 + seed)
                sw = Stopwatch(timing)
                for _ in range(steps):
                    x.grad = H * (x.detach() - x_star) + tau * torch.randn(
                        p, generator=gen)
                    opt.step()
                    sw.tick()
                per[o].append(float(0.5 * (H * (x.detach() - x_star) ** 2).sum()))
                if timing and sw.ms_per_step:
                    mss.append(sw.ms_per_step)
                if gate and getattr(opt, "stats", None):
                    gs = gate_stats(opt, o)
                    if gs.get("gate_open_frac") is not None:
                        gfs.append(gs["gate_open_frac"])
        for o in OPTS:
            a = np.array(per[o])
            rec[o] = float(a.mean())
            rec[o + "_sd"] = float(a.std(ddof=1))
        d = np.array(per["adamw"]) - np.array(per["sradam"])
        rec["paired_diff"] = float(d.mean())
        rec["t_stat"] = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(seeds))))
        rec["paired_all"] = [float(x) for x in d]
        rec["tuned"] = chosen or None
        if mss:
            rec["opt_ms_per_step"] = float(np.mean(mss))
        if gfs:
            rec["gate_open_frac"] = float(np.mean(gfs))
        rows.append(rec)
        print(f"    tau={tau:4.2f}  " + "  ".join(
            f"{LABEL[o]}={rec[o]:8.3f}" for o in OPTS)
            + f"   paired AdamW-SR={rec['paired_diff']:+8.3f} "
              f"(t={rec['t_stat']:+6.2f})", flush=True)
    return rows


def logistic(flips=(0.0, 0.1, 0.2, 0.3, 0.4), d=200, K=10, N=20_000, B=64,
             steps=2000, seeds=(1, 2, 3), args=None):
    lossf = torch.nn.CrossEntropyLoss()
    timing = bool(getattr(args, "timing", False))
    gate = bool(getattr(args, "gate_log", False))
    do_tune = bool(getattr(args, "tune", False))
    patience = int(getattr(args, "early_stop", 0) or 0)
    pilot_steps = 400 if getattr(args, "smoke", False) else 600
    rows = []
    for fr in flips:
        chosen = {}
        if do_tune:
            _, chosen = run_tuning(
                args, OPTS,
                lambda cfg, seed, fr=fr: pilot_log(cfg, seed, fr, pilot_steps),
                f"simulation_noise_log_f{fr:g}", default_grids=LG_GRID)
        rec = dict(flip=fr)
        per = {o: [] for o in OPTS}
        mss, gfs, stops = [], [], []
        for seed in seeds:
            torch.manual_seed(seed)
            Wt = torch.randn(d, K) / np.sqrt(d)
            X = torch.randn(N, d)
            y = (X @ Wt).argmax(1)
            mask = torch.rand(N) < fr
            y = torch.where(mask, torch.randint(0, K, (N,)), y)
            Xte = torch.randn(4000, d)
            yte = (Xte @ Wt).argmax(1)
            Xv = torch.randn(VAL_N, d)           # clean validation for ES
            yv = (Xv @ Wt).argmax(1)
            for o in OPTS:
                torch.manual_seed(seed)
                W = torch.zeros(d, K, requires_grad=True)
                opt = build(o, [W], _lr(chosen, o, DEFAULT_LR["log"]),
                            **_ow(chosen, o, gate=gate))
                gen = torch.Generator().manual_seed(31 + seed)
                sw = Stopwatch(timing)
                es = EarlyStopper(patience, mode="min") if patience else None
                best_w, stopped = None, None
                for step in range(steps):
                    idx = torch.randint(0, N, (B,), generator=gen)
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
                if timing and sw.ms_per_step:
                    mss.append(sw.ms_per_step)
                if stopped:
                    stops.append(stopped)
                if gate and getattr(opt, "stats", None):
                    gs = gate_stats(opt, o)
                    if gs.get("gate_open_frac") is not None:
                        gfs.append(gs["gate_open_frac"])
        for o in OPTS:
            a = np.array(per[o])
            rec[o] = float(a.mean())
            rec[o + "_sd"] = float(a.std(ddof=1))
        dd = np.array(per["adamw"]) - np.array(per["sradam"])
        rec["paired_diff"] = float(dd.mean())
        rec["t_stat"] = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(seeds))))
        rec["paired_all"] = [float(x) for x in dd]
        rec["tuned"] = chosen or None
        if mss:
            rec["opt_ms_per_step"] = float(np.mean(mss))
        if gfs:
            rec["gate_open_frac"] = float(np.mean(gfs))
        if stops:
            rec["early_stop_step"] = float(np.mean(stops))
        rows.append(rec)
        print(f"    flip={fr:4.2f}  " + "  ".join(
            f"{LABEL[o]}={rec[o]:7.4f}" for o in OPTS)
            + f"   paired AdamW-SR={rec['paired_diff']:+7.4f} "
              f"(t={rec['t_stat']:+6.2f})", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="short runs and 2 seeds for pipeline checks")
    ap.add_argument("--steps", type=int, default=None,
                    help="override steps per run (item a8 config unification)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="override seed count (item a8)")
    ap.add_argument("--taus", default=None, help="comma list of tau values")
    ap.add_argument("--flips", default=None, help="comma list of flip rates")
    add_tuning_args(ap)
    args = ap.parse_args()

    taus = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    flips = (0.0, 0.1, 0.2, 0.3, 0.4)
    q_steps, q_seeds = 3000, (1, 2, 3, 4, 5)
    l_steps, l_seeds = 2000, (1, 2, 3)
    if args.smoke:
        q_steps, q_seeds = 300, (1, 2)
        l_steps, l_seeds = 300, (1, 2)
    if args.steps:
        q_steps = l_steps = args.steps
    if args.seeds:
        q_seeds = l_seeds = tuple(range(1, args.seeds + 1))
    if args.taus:
        taus = tuple(float(x) for x in str(args.taus).split(","))
    if args.flips:
        flips = tuple(float(x) for x in str(args.flips).split(","))

    print("  noisy quadratic")
    q = quadratic(taus=taus, steps=q_steps, seeds=q_seeds, args=args)
    print("  logistic regression with label noise")
    lg = logistic(flips=flips, steps=l_steps, seeds=l_seeds, args=args)

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ta = [r["tau"] for r in q]
    for o, st in zip(OPTS, ["v:", "s--", "^-.", "o-"]):
        ax[0].loglog(ta, [r[o] for r in q], st, ms=3, label=LABEL[o])
    ax[0].set_xlabel(r"gradient-noise scale $\tau$")
    ax[0].set_ylabel("terminal objective")
    ax[0].legend(loc="upper left")

    fr = [r["flip"] for r in lg]
    for o, st in zip(OPTS, ["v:", "s--", "^-.", "o-"]):
        ax[1].plot(fr, [r[o] for r in lg], st, ms=3, label=LABEL[o])
    ax[1].set_xlabel("label-flip rate")
    ax[1].set_ylabel("test log-loss")
    fig.savefig(os.path.join(FIGS, "sim_noise_levels.pdf"))
    plt.close(fig)

    payload = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  smoke=args.smoke, steps=q_steps, seeds=list(q_seeds),
                  taus=list(taus), flips=list(flips),
                  tuned=bool(args.tune), early_stop=args.early_stop,
                  timing=args.timing, gate_log=args.gate_log,
                  note="positive paired_diff = SR-Adam better (lower loss); "
                       "corrupted labels are TRAIN labels only, tests are clean"),
        quadratic=q, logistic=lg,
    )
    with open(os.path.join(OUT, "simulation_noise.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("\nwritten", os.path.join(OUT, "simulation_noise.json"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()