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
REPORT_DIR = HERE.parent / "reports" / "REPORT_for_Arashi"
for p in (FIGS, OUT, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)

OPTS = ["adamw", "amsgrad", "sradam"]


def build(name, params, lr):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if name == "amsgrad":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, amsgrad=True)
    return SRAdam(params, lr=lr, warmup=5)


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


def run_quadratic(p, tau, law, steps, seeds):
    H = torch.linspace(0.05, 2.0, p)
    per = {o: [] for o in OPTS}
    k_mean = []
    for seed in seeds:
        torch.manual_seed(seed)
        x_star = torch.randn(p)
        gen = torch.Generator().manual_seed(500 + seed)
        for o in OPTS:
            torch.manual_seed(seed)
            x = torch.zeros(p, requires_grad=True)
            opt = build(o, [x], 2e-2)
            ks = []
            for _ in range(steps):
                eps = (tau * torch.randn(p, generator=gen) if law == "gaussian"
                       else student_noise((p,), tau, gen))
                x.grad = H * (x.detach() - x_star) + eps
                opt.step()
                if o == "sradam" and getattr(opt, "stats", None):
                    ks.append(opt.stats[0]["shrink_weight"])
            per[o].append(float(0.5 * (H * (x.detach() - x_star) ** 2).sum()))
            if ks:
                k_mean.append(float(np.mean(ks)))
    rec = paired(per, seeds)
    rec.update(problem=f"quadratic_{law}", p=p, tau=tau,
               mean_k=float(np.mean(k_mean)) if k_mean else None)
    return rec


def run_logistic(p, flip, steps, seeds, N=4000, K=10, batch=64):
    lossf = torch.nn.CrossEntropyLoss()
    per = {o: [] for o in OPTS}
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
        for o in OPTS:
            torch.manual_seed(seed)
            W = torch.zeros(p, K, requires_grad=True)
            opt = build(o, [W], 3e-2)
            gen = torch.Generator().manual_seed(600 + seed)
            for _ in range(steps):
                idx = torch.randint(0, N, (batch,), generator=gen)
                opt.zero_grad()
                lossf(X[idx] @ W, y[idx]).backward()
                opt.step()
            with torch.no_grad():
                per[o].append(float(lossf(Xte @ W, yte)))
    rec = paired(per, seeds)
    rec.update(problem="logistic_labelnoise", p=p, flip=flip, mean_k=None)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        ps = [10, 50, 200]
        seeds = (1, 2)
        steps = 400
    else:
        ps = [10, 20, 50, 100, 200, 500]
        seeds = (1, 2, 3)
        steps = 800

    rows = []
    t0 = time.time()

    print("=== quadratic gaussian, tau=1.0 (noise-dominated-ish) ===", flush=True)
    for p in ps:
        r = run_quadratic(p, 1.0, "gaussian", steps, seeds)
        rows.append(r)
        print(f"  p={p:4d}  AdamW={r['adamw']:.4f}  SR={r['sradam']:.4f}  "
              f"Δ={r['paired_diff']:+.4f} (t={r['t_stat']:+.2f})  win={r['win_sradam']}  "
              f"k̄={r['mean_k']}", flush=True)

    print("=== quadratic gaussian, tau=0.1 (more signal) ===", flush=True)
    for p in ps:
        r = run_quadratic(p, 0.1, "gaussian", steps, seeds)
        rows.append(r)
        print(f"  p={p:4d}  AdamW={r['adamw']:.4f}  SR={r['sradam']:.4f}  "
              f"Δ={r['paired_diff']:+.4f}  win={r['win_sradam']}", flush=True)

    print("=== quadratic student-t, tau=1.0 ===", flush=True)
    for p in ps:
        r = run_quadratic(p, 1.0, "student", steps, seeds)
        rows.append(r)
        print(f"  p={p:4d}  AdamW={r['adamw']:.4f}  SR={r['sradam']:.4f}  "
              f"Δ={r['paired_diff']:+.4f}  win={r['win_sradam']}", flush=True)

    print("=== logistic label flip=0.2 ===", flush=True)
    for p in ps:
        r = run_logistic(p, 0.2, steps, seeds)
        rows.append(r)
        print(f"  p={p:4d}  AdamW={r['adamw']:.4f}  SR={r['sradam']:.4f}  "
              f"Δ={r['paired_diff']:+.4f}  win={r['win_sradam']}", flush=True)

    payload = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  seconds=time.time() - t0, smoke=args.smoke,
                  ps=ps, seeds=list(seeds), steps=steps,
                  note="positive paired_diff = SR-Adam better (lower loss)"),
        rows=rows,
    )
    (OUT / "path4_dim_sweep.json").write_text(json.dumps(payload, indent=2))
    (REPORT_DIR / "path4_dim_sweep.json").write_text(json.dumps(payload, indent=2))

    # figure: Δ vs p for each problem/noise
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    axl = [
        ("quadratic_gaussian", 1.0, "Quadratic Gaussian τ=1", axes[0, 0]),
        ("quadratic_gaussian", 0.1, "Quadratic Gaussian τ=0.1", axes[0, 1]),
        ("quadratic_student", 1.0, "Quadratic Student-t τ=1", axes[1, 0]),
        ("logistic_labelnoise", 0.2, "Logistic flip=0.2", axes[1, 1]),
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
