"""
log_variance_ratio.py — Phase B / diagnostic item a5.

Sections 8.1 and 8.6 of the manuscript never measure the quantity the
dominance window lives on: the ratio of the estimated gradient-noise variance
to the true one, sigma_hat^2 / sigma^2, over the course of an optimisation
run.  Diagnostic item a5 shows that with beta1 = 0.9 (about 10 steps of
memory) and beta2 = 0.999 (about 1000 steps) a drifting gradient mean inflates
the statistic far beyond the factor-two margin of Corollary 3.

This script runs two controlled-noise problems where the true conditional
gradient variance is known or estimable, logs the ratio every step, and flags
excursions outside [0.5, 2]:

  quadratic   g = H (x - x*) + tau_j z_j with heteroscedastic tau_j known
              exactly; the mean drifts as the optimiser converges.
  logistic    multinomial logistic regression, mini-batch size B; the true
              conditional variance is estimated empirically every `--probe`
              steps from `--replicas` independent batches at the same
              parameters (per-coordinate variance across replicas, pooled).

Both variance windows of SRAdam are available:

  --window beta1     the paper's Theorem 7 exactly as implemented
  --window matched   both moment terms on the beta2 horizon (a5 suggestion,
                     new `var_window="matched"` flag in sr_adam.py)

Outputs:
  ../figs/var_ratio_<problem>_<window>.pdf/.png
  ../out/log_variance_ratio.json      (series + summary per run)

Run:
    python log_variance_ratio.py                        # both problems, both windows
    python log_variance_ratio.py --problem quadratic --window matched
    python log_variance_ratio.py --smoke                # seconds, one seed
"""

from __future__ import annotations

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

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs"
OUT = HERE.parent / "out"
FIGS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25,
})

FLAGS = (0.5, 2.0)          # factor-two margin of Corollary 3


# ----------------------------------------------------------------- quadratic


def run_quadratic(window, steps=600, seeds=(1, 2), p=400, verbose=False):
    """Return list of dicts t, ratio (mean over seeds), lo, hi."""
    H = None
    tau = np.linspace(0.1, 1.0, p) ** 1.0          # heteroscedastic, known
    truth = float(np.mean(tau ** 2))
    tau_t = torch.from_numpy(tau).float()
    series = None
    for seed in seeds:
        torch.manual_seed(seed)
        H = torch.linspace(0.05, 2.0, p)
        x_star = torch.randn(p)
        x = torch.zeros(p, requires_grad=True)
        opt = SRAdam([x], lr=2e-2, warmup=5, track_stats=True, var_window=window)
        gen = torch.Generator().manual_seed(777 + seed)
        rs = []
        for t in range(1, steps + 1):
            x.grad = H * (x.detach() - x_star) + tau_t * torch.randn(
                p, generator=gen)
            opt.step()
            s2 = float(opt.stats[0]["sigma2"]) if opt.stats else float("nan")
            rs.append(s2 / truth)
        rs = np.asarray(rs)
        series = rs[:, None] if series is None else np.concatenate(
            [series, rs[:, None]], axis=1)
    mean = np.nanmean(series, axis=1)
    lo = np.nanpercentile(series, 25, axis=1)
    hi = np.nanpercentile(series, 75, axis=1)
    return [dict(t=i + 1, ratio=float(mean[i]), q25=float(lo[i]),
                 q75=float(hi[i])) for i in range(len(mean))]


# ------------------------------------------------------------------ logistic


def _empirical_variance(W, X, y, lossf, batches, gen, d, K, B):
    """Pooled per-coordinate gradient variance over `batches` mini-batches."""
    grads = []
    n = X.shape[0]
    for _ in range(batches):
        idx = torch.randint(0, n, (B,), generator=gen)
        g = torch.autograd.grad(lossf(X[idx] @ W, y[idx]), W)[0]
        grads.append(g.reshape(-1).numpy())
    G = np.asarray(grads)                                   # (R, d*K)
    return float(G.var(axis=0, ddof=1).mean())


def run_logistic(window, steps=800, seeds=(1,), d=200, K=10, N=20_000, B=64,
                  probe=50, replicas=12, verbose=False):
    """Ratio at probe steps only (truth estimated empirically at those steps)."""
    lossf = torch.nn.CrossEntropyLoss()
    points = None
    for seed in seeds:
        torch.manual_seed(seed)
        Wt = torch.randn(d, K) / math.sqrt(d)
        X = torch.randn(N, d)
        y = (X @ Wt).argmax(1)
        X = X.clone()
        W = torch.zeros(d, K, requires_grad=True)
        opt = SRAdam([W], lr=3e-2, warmup=5, track_stats=True, var_window=window)
        gen = torch.Generator().manual_seed(31 + seed)
        rs = {}
        for t in range(1, steps + 1):
            idx = torch.randint(0, N, (B,), generator=gen)
            opt.zero_grad()
            lossf(X[idx] @ W, y[idx]).backward()
            truth = _empirical_variance(W, X, y, lossf, replicas, gen, d, K, B) \
                if t % probe == 0 else None
            opt.step()
            if truth:
                s2 = float(opt.stats[0]["sigma2"]) if opt.stats else float("nan")
                rs[t] = s2 / truth
        points = rs if points is None else {
            t: (points[t] + rs[t]) / 2 for t in points if t in rs}
    return [dict(t=t, ratio=float(r)) for t, r in sorted(points.items())]


# ------------------------------------------------------------------- summary


def summarise(name, window, series):
    ts = np.array([r["t"] for r in series], float)
    rs = np.array([r["ratio"] for r in series], float)
    ok = np.isfinite(rs)
    ts, rs = ts[ok], rs[ok]
    if rs.size == 0:
        return dict(problem=name, window=window, note="no finite points")
    at = {}
    for tt in (20, 100, 300):
        m = np.abs(ts - tt) <= max(1, tt * 0.05) + 1
        at[str(tt)] = float(rs[m].mean()) if m.any() else None
    beyond = int(np.sum((rs > FLAGS[1]) | (rs < FLAGS[0])))
    return dict(problem=name, window=window, n_points=int(rs.size),
                ratio_at=at, max_ratio=float(rs.max()),
                min_ratio=float(rs.min()),
                frac_beyond_factor2=beyond / rs.size,
                beyond_factor2=bool(beyond > 0),
                median_ratio=float(np.median(rs)))


def plot(name, runs, write=True):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for window, series in runs.items():
        ts = [r["t"] for r in series]
        rs = [r["ratio"] for r in series]
        ax.plot(ts, rs, lw=1.4, label=f"var_window={window}")
    ax.axhline(1.0, color="k", lw=1.0)
    ax.axhline(FLAGS[1], color="r", ls="--", lw=0.9,
               label=f"upper margin {FLAGS[1]:g}")
    ax.axhline(FLAGS[0], color="r", ls=":", lw=0.9,
               label=f"lower margin {FLAGS[0]:g}")
    ax.set_xscale("log")
    ax.set_xlabel("step t")
    ax.set_ylabel(r"$\hat\sigma^2_t / \sigma^2_{\mathrm{true}}$")
    ax.set_title(f"Variance ratio along training — {name}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    if write:
        for ext in ("pdf", "png"):
            try:
                fig.savefig(FIGS / f"var_ratio_{name}.{ext}")
            except PermissionError:
                pass
        print("wrote", FIGS / f"var_ratio_{name}.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="both",
                    choices=["both", "quadratic", "logistic"])
    ap.add_argument("--window", default="both",
                    choices=["both", "beta1", "matched"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--logistic-steps", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--probe", type=int, default=50)
    ap.add_argument("--replicas", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    windows = (["beta1", "matched"] if args.window == "both"
               else [args.window])
    seeds = (1,) if args.smoke else tuple(range(1, args.seeds + 1))
    steps = 150 if args.smoke else args.steps
    lsteps = 200 if args.smoke else args.logistic_steps

    payload = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             smoke=args.smoke, seeds=list(seeds),
                             flags=list(FLAGS),
                             note="ratio = pooled sigma_hat^2 / true pooled "
                                  "variance; empirical truth for logistic "
                                  "estimated from replica batches at probes"),
                   runs=[], summary=[])
    t0 = time.time()

    problems = (["quadratic", "logistic"] if args.problem == "both"
                else [args.problem])
    for prob in problems:
        runs = {}
        for w in windows:
            if prob == "quadratic":
                ser = run_quadratic(w, steps=steps, seeds=seeds)
            else:
                ser = run_logistic(w, steps=lsteps, seeds=seeds,
                                   probe=max(10, args.probe // (5 if args.smoke else 1)),
                                   replicas=8 if args.smoke else args.replicas)
            runs[w] = ser
            s = summarise(prob, w, ser)
            payload["runs"].append(dict(problem=prob, window=w, series=ser))
            payload["summary"].append(s)
            print(f"  {prob:10s} window={w:8s}  median={s.get('median_ratio'):.3f}  "
                  f"max={s.get('max_ratio'):.3f}  "
                  f"t20={s.get('ratio_at', {}).get('20')}  "
                  f"t100={s.get('ratio_at', {}).get('100')}  "
                  f"t300={s.get('ratio_at', {}).get('300')}  "
                  f"beyond x2: {100 * s.get('frac_beyond_factor2', 0):.1f}%"
                  if "median_ratio" in s else f"  {prob} {w}: {s.get('note')}",
                  flush=True)
        if not args.no_plot:
            plot(prob, runs)
        for s in payload["summary"]:
            if s["problem"] == prob and s.get("beyond_factor2"):
                print(f"  !! {prob}/{s['window']}: ratio leaves the "
                      f"factor-two window of Corollary 3 "
                      f"({100 * s['frac_beyond_factor2']:.1f}% of logged steps)")

    (OUT / "log_variance_ratio.json").write_text(json.dumps(payload, indent=2))
    print(f"elapsed {time.time() - t0:.1f}s")
    print("wrote", OUT / "log_variance_ratio.json")


if __name__ == "__main__":
    main()