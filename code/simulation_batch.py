"""
Optimisation across mini-batch sizes on a synthetic non-convex problem.

A two-layer network with a fixed teacher is fitted by SR-Adam and by AdamW with the
AMSGrad maximum, under identical step sizes, identical initialisations and identical
mini-batch streams, so that the optimiser is the only thing that differs.  Each
configuration is repeated over several teacher and initialisation seeds.

The convergence theorem bounds the stationarity gap of SR-Adam by a term that vanishes
with the step count plus a floor proportional to c_t lambda_max(Sigmahat_t), and the
gradient-noise scale falls like 1/B in the batch size B.  The middle panel of the figure
records the realised noise trace against that prediction, and the right panel records the
realised direction weight against the fixed weight Adam would have used.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sr_adam import SRAdam

FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")
OUT = os.path.join(os.path.dirname(__file__), "..", "out")

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "lines.linewidth": 1.3,
})

D, H, K, N = 120, 48, 8, 40_000
BATCHES = [8, 16, 32, 64, 128, 256, 512]
STEPS = 1500


def make_problem(seed):
    torch.manual_seed(seed)
    W1 = torch.randn(D, 40) / np.sqrt(D)
    W2 = torch.randn(40, K) / np.sqrt(40)
    X = torch.randn(N, D)
    y = (torch.tanh(X @ W1) @ W2 + 0.5 * torch.randn(N, K)).argmax(1)
    return X, y


def run(name, X, y, B, seed):
    lossf = torch.nn.CrossEntropyLoss()
    torch.manual_seed(1000 + seed)
    net = torch.nn.Sequential(torch.nn.Linear(D, H), torch.nn.Tanh(),
                              torch.nn.Linear(H, K))
    opt = (SRAdam(net.parameters(), lr=3e-3, warmup=5, track_stats=True)
           if name == "sr" else
           torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.0, amsgrad=True))
    gen = torch.Generator().manual_seed(4242 + seed)
    floors, weights = [], []
    for i in range(STEPS):
        idx = torch.randint(0, N, (B,), generator=gen)
        opt.zero_grad()
        lossf(net(X[idx]), y[idx]).backward()
        opt.step()
        if name == "sr" and i > STEPS // 3 and opt.stats:
            floors.append(sum(s["sigma2"] * s["numel"] for s in opt.stats
                              if s["sigma2"] == s["sigma2"]))
            weights.append(float(np.mean([s["shrink_weight"] for s in opt.stats])))
    net.zero_grad()
    lossf(net(X), y).backward()
    gn = float(torch.sqrt(sum((q.grad ** 2).sum() for q in net.parameters())))
    with torch.no_grad():
        fl = float(lossf(net(X), y))
    return fl, gn, (float(np.mean(floors)) if floors else float("nan")), \
        (float(np.mean(weights)) if weights else float("nan"))


def main(seeds=(15, 16, 17, 18, 19)):
    acc = {B: dict(adam=[], sr=[], adam_gn=[], sr_gn=[], floor=[], weight=[])
           for B in BATCHES}
    for seed in seeds:
        X, y = make_problem(seed)
        for B in BATCHES:
            la, ga, _, _ = run("adam", X, y, B, seed)
            ls, gs, fl, wt = run("sr", X, y, B, seed)
            acc[B]["adam"].append(la)
            acc[B]["sr"].append(ls)
            acc[B]["adam_gn"].append(ga)
            acc[B]["sr_gn"].append(gs)
            acc[B]["floor"].append(fl)
            acc[B]["weight"].append(wt)
        print(f"    seed {seed} done")

    rows = []
    for B in BATCHES:
        a, s = np.array(acc[B]["adam"]), np.array(acc[B]["sr"])
        ga, gs = np.array(acc[B]["adam_gn"]), np.array(acc[B]["sr_gn"])
        rows.append(dict(
            B=B,
            adam_mean=float(a.mean()), adam_sd=float(a.std(ddof=1)),
            sr_mean=float(s.mean()), sr_sd=float(s.std(ddof=1)),
            adam_gn=float(ga.mean()), sr_gn=float(gs.mean()),
            floor=float(np.mean(acc[B]["floor"])),
            weight=float(np.mean(acc[B]["weight"])),
            paired_diff=float((a - s).mean()), paired_sd=float((a - s).std(ddof=1)),
        ))
        d, sd = rows[-1]["paired_diff"], rows[-1]["paired_sd"]
        tstat = d / (sd / np.sqrt(len(seeds))) if sd > 0 else float("nan")
        rows[-1]["t_stat"] = float(tstat)
        print(f"    B={B:4d}  AdamW {a.mean():.4f}+-{a.std(ddof=1):.4f}"
              f"   SR-Adam {s.mean():.4f}+-{s.std(ddof=1):.4f}"
              f"   paired diff {d:+.4f} (t={tstat:+.2f})"
              f"   tr(Sigmahat)={rows[-1]['floor']:.3e}  weight={rows[-1]['weight']:.3f}")

    Bs = np.array(BATCHES, dtype=float)
    fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.4))
    ax[0].errorbar(Bs, [r["adam_mean"] for r in rows], yerr=[r["adam_sd"] for r in rows],
                   fmt="s--", ms=3, capsize=2, label="AdamW (AMSGrad)")
    ax[0].errorbar(Bs, [r["sr_mean"] for r in rows], yerr=[r["sr_sd"] for r in rows],
                   fmt="o-", ms=3, capsize=2, label="SR-Adam")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("mini-batch size $B$")
    ax[0].set_ylabel(f"training loss after {STEPS} steps")
    ax[0].legend()

    fl = np.array([r["floor"] for r in rows])
    ax[1].loglog(Bs, fl, "o-", ms=3, label=r"$\mathrm{tr}\,\hat\Sigma_t$")
    ax[1].loglog(Bs, fl[0] * Bs[0] / Bs, "k:", label=r"$\propto B^{-1}$")
    ax[1].set_xlabel("mini-batch size $B$")
    ax[1].set_ylabel("estimated noise trace")
    ax[1].legend()

    ax[2].semilogx(Bs, [r["weight"] for r in rows], "o-", ms=3, base=2,
                   label="mean Stein weight")
    ax[2].axhline(0.1, ls="--", color="C1", label=r"fixed weight $1-\beta_1$")
    ax[2].set_xlabel("mini-batch size $B$")
    ax[2].set_ylabel("realised direction weight")
    ax[2].set_ylim(0, 1)
    ax[2].legend()
    fig.savefig(os.path.join(FIGS, "sim_batch_size.pdf"))
    plt.close(fig)

    with open(os.path.join(OUT, "simulation_batch.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwritten")


if __name__ == "__main__":
    main()
