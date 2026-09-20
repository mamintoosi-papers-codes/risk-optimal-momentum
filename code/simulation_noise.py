"""
Optimisation under controlled gradient-noise levels.

The theory says the Stein weight buys most where the mini-batch gradient is
noise-dominated, that is where the squared distance between the current gradient and the
historical average is small relative to the total noise variance of the block.  The
studies below vary that ratio directly rather than inferring it.

Problem 1, a noisy quadratic.  The objective is (1/2)(theta - theta*)' H (theta - theta*)
with H diagonal and a spread of curvatures, and the observed gradient is the exact
gradient plus isotropic noise of standard deviation tau.  Sweeping tau traces the whole
range from the signal-dominated to the noise-dominated regime while holding everything
else fixed.

Problem 2, logistic regression with label noise.  A convex problem with genuine mini-batch
noise, where the noise level is controlled by the label-flip rate rather than by an
artificial additive term.

Reported for each setting: terminal objective (problem 1) or test log-loss (problem 2),
averaged over seeds, together with a paired comparison against AdamW.
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

OPTS = ["sgdm", "adamw", "amsgrad", "sradam"]
LABEL = {"sgdm": "SGD with momentum", "adamw": "AdamW",
         "amsgrad": "AdamW (AMSGrad)", "sradam": "SR-Adam"}


def build(name, params, lr):
    if name == "sgdm":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if name == "amsgrad":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, amsgrad=True)
    return SRAdam(params, lr=lr, warmup=5)


# -------------------------------------------------------------------- noisy quadratic
def quadratic(taus=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), p=400, steps=3000, seeds=(1, 2, 3, 4, 5)):
    H = torch.linspace(0.05, 2.0, p)
    rows = []
    for tau in taus:
        rec = dict(tau=tau)
        per = {o: [] for o in OPTS}
        for seed in seeds:
            torch.manual_seed(seed)
            x_star = torch.randn(p)
            for o in OPTS:
                torch.manual_seed(seed)
                x = torch.zeros(p, requires_grad=True)
                opt = build(o, [x], lr=2e-2 if o != "sgdm" else 5e-2)
                gen = torch.Generator().manual_seed(777 + seed)
                for _ in range(steps):
                    x.grad = H * (x.detach() - x_star) + tau * torch.randn(
                        p, generator=gen)
                    opt.step()
                per[o].append(float(0.5 * (H * (x.detach() - x_star) ** 2).sum()))
        for o in OPTS:
            a = np.array(per[o])
            rec[o] = float(a.mean())
            rec[o + "_sd"] = float(a.std(ddof=1))
        d = np.array(per["adamw"]) - np.array(per["sradam"])
        rec["paired_diff"] = float(d.mean())
        rec["t_stat"] = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(seeds))))
        rows.append(rec)
        print(f"    tau={tau:4.2f}  " + "  ".join(
            f"{LABEL[o]}={rec[o]:8.3f}" for o in OPTS)
            + f"   paired AdamW-SR={rec['paired_diff']:+8.3f} (t={rec['t_stat']:+6.2f})")
    return rows


# ------------------------------------------------------------- logistic with label noise
def logistic(flips=(0.0, 0.1, 0.2, 0.3, 0.4), d=200, K=10, N=20_000, B=64,
             steps=2000, seeds=(1, 2, 3)):
    rows = []
    lossf = torch.nn.CrossEntropyLoss()
    for fr in flips:
        rec = dict(flip=fr)
        per = {o: [] for o in OPTS}
        for seed in seeds:
            torch.manual_seed(seed)
            Wt = torch.randn(d, K) / np.sqrt(d)
            X = torch.randn(N, d)
            y = (X @ Wt).argmax(1)
            mask = torch.rand(N) < fr
            y = torch.where(mask, torch.randint(0, K, (N,)), y)
            Xte = torch.randn(4000, d)
            yte = (Xte @ Wt).argmax(1)
            for o in OPTS:
                torch.manual_seed(seed)
                W = torch.zeros(d, K, requires_grad=True)
                opt = build(o, [W], lr=3e-2 if o != "sgdm" else 1e-1)
                gen = torch.Generator().manual_seed(31 + seed)
                for _ in range(steps):
                    idx = torch.randint(0, N, (B,), generator=gen)
                    opt.zero_grad()
                    lossf(X[idx] @ W, y[idx]).backward()
                    opt.step()
                with torch.no_grad():
                    per[o].append(float(lossf(Xte @ W, yte)))
        for o in OPTS:
            a = np.array(per[o])
            rec[o] = float(a.mean())
            rec[o + "_sd"] = float(a.std(ddof=1))
        dd = np.array(per["adamw"]) - np.array(per["sradam"])
        rec["paired_diff"] = float(dd.mean())
        rec["t_stat"] = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(seeds))))
        rows.append(rec)
        print(f"    flip={fr:4.2f}  " + "  ".join(
            f"{LABEL[o]}={rec[o]:7.4f}" for o in OPTS)
            + f"   paired AdamW-SR={rec['paired_diff']:+7.4f} (t={rec['t_stat']:+6.2f})")
    return rows


if __name__ == "__main__":
    print("  noisy quadratic")
    q = quadratic()
    print("  logistic regression with label noise")
    lg = logistic()

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    taus = [r["tau"] for r in q]
    for o, st in zip(OPTS, ["v:", "s--", "^-.", "o-"]):
        ax[0].loglog(taus, [r[o] for r in q], st, ms=3, label=LABEL[o])
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

    with open(os.path.join(OUT, "simulation_noise.json"), "w") as f:
        json.dump(dict(quadratic=q, logistic=lg), f, indent=2)
    print("\nwritten")
