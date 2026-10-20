"""
Regenerate every figure in the paper from the stored simulation output.

Reading the figures from the stored results rather than from a fresh Monte Carlo run
keeps the figures and the numbers quoted in the text tied to the same output.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figs")
OUT = os.path.join(HERE, "..", "out")
PAPER = os.path.join(HERE, "..", "paper")

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "lines.linewidth": 1.3,
})


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


# ------------------------------------------------------------------ figure 1: risk-SNR
def fig_risk_snr(res):
    rows = res["A"]
    snr = np.array([r["snr"] for r in rows])
    base = np.array([r["raw"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for key, lab, st in [("adam", r"fixed weight $1-\beta_1$", "s--"),
                         ("stein", r"Stein weight, known $\sigma^2$", "o-"),
                         ("plugin", r"Stein weight, plug-in $\hat\sigma^2$", "^-."),
                         ("oracle", "oracle weight", ":")]:
        ax[0].loglog(snr, np.array([r[key] for r in rows]) / base, st, ms=3, label=lab)
    ax[0].axhline(1.0, color="k", lw=0.8)
    ax[0].set_xlabel(r"$\|\delta\|^2/(p\sigma^2)$")
    ax[0].set_ylabel("risk relative to raw gradient")
    ax[0].legend(loc="upper left")

    ax[1].semilogx(snr, [r["k_oracle"] for r in rows], ":", label="oracle weight")
    ax[1].semilogx(snr, [r["k_stein_mean"] for r in rows], "o-", ms=3,
                   label="mean Stein weight")
    ax[1].axhline(0.1, ls="--", color="C0", label=r"fixed weight $1-\beta_1$")
    ax[1].set_xlabel(r"$\|\delta\|^2/(p\sigma^2)$")
    ax[1].set_ylabel("direction weight")
    ax[1].legend(loc="upper left")
    fig.savefig(os.path.join(FIGS, "sim_risk_snr.pdf"))
    plt.close(fig)


# ------------------------------------------------------------ figure 2: variance bias
def fig_variance(res):
    b = res["B"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(b["t"], b["naive_ratio"], "s--", ms=3, label=r"$\hat v_t-\hat m_t^2$")
    ax.plot(b["t"], b["closed_form"], "k-", lw=0.8, label=r"$1-\phi_t(\beta_1)$")
    ax.plot(b["t"], b["debiased_ratio"], "o-", ms=3,
            label=r"$(\hat v_t-\hat m_t^2)/(1-\phi_t)$")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhline(b["limit_naive"], color="C0", lw=0.8, ls=":")
    ax.text(35, b["limit_naive"] - 0.07, r"$2\beta_1/(1+\beta_1)$", fontsize=7,
            color="C0")
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel(r"estimate divided by $\bar\sigma^2$")
    ax.set_ylim(0.3, 1.12)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIGS, "sim_variance_bias.pdf"))
    plt.close(fig)


# ------------------------------------------------------ figure 3: effective dimension
def fig_effdim(res3):
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.5), sharey=True)
    for ax, snr in zip(axes, [0.25, 1.0, 4.0]):
        rows = res3[f"snr{snr}"]
        pe = [r["peff"] for r in rows]
        ax.semilogx(pe, [r["pooled_std"] for r in rows], "s--", ms=3,
                    label=r"pooled scalar, $c=p-2$")
        ax.semilogx(pe, [r["cons_std"] for r in rows], "^:", ms=3,
                    label=r"coordinatewise, $c=p_{\mathrm{eff}}-2$")
        ax.semilogx(pe, [r["stdz_std"] for r in rows], "o-", ms=3,
                    label=r"coordinatewise, $c=p-2$")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlabel(r"$p_{\mathrm{eff}}$")
        ax.set_title(rf"$\|\delta\|^2_{{\Sigma^{{-1}}}}/p={snr}$")
    axes[0].set_ylabel("risk relative to raw gradient")
    axes[0].set_ylim(0, 2.7)
    axes[0].legend(loc="upper center")
    fig.savefig(os.path.join(FIGS, "sim_effective_dimension.pdf"))
    plt.close(fig)


# -------------------------------------------------------------- figure 4: heavy tails
def fig_heavy(res):
    rows = res["D"]
    nus = [r["nu"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for rho, st in zip([1.0, 0.75, 0.5], ["o-", "s--", "^:"]):
        ax.plot(nus, [r[f"rho{rho}"] for r in rows], st, ms=3, label=rf"$\rho={rho}$")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(nus)
    ax.set_xticklabels([str(v) for v in nus])
    ax.set_xlabel(r"degrees of freedom $\nu$")
    ax.set_ylabel("risk relative to raw gradient")
    ax.legend()
    fig.savefig(os.path.join(FIGS, "sim_heavy_tails.pdf"))
    plt.close(fig)


# ------------------------------------------------------------- figure 5: batch regimes
def fig_batch(rows):
    Bs = np.array([r["B"] for r in rows], dtype=float)
    fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.4))
    ax[0].errorbar(Bs, [r["adam_mean"] for r in rows], yerr=[r["adam_sd"] for r in rows],
                   fmt="s--", ms=3, capsize=2, label="AdamW (AMSGrad)")
    ax[0].errorbar(Bs, [r["sr_mean"] for r in rows], yerr=[r["sr_sd"] for r in rows],
                   fmt="o-", ms=3, capsize=2, label="SR-Adam")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("mini-batch size $B$")
    ax[0].set_ylabel("training loss")
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
    ax[2].set_ylabel("direction weight")
    ax[2].set_ylim(0, 0.35)
    ax[2].legend()
    fig.savefig(os.path.join(FIGS, "sim_batch_size.pdf"))
    plt.close(fig)


# ------------------------------------------------------------- figure 6: noise sweeps
def fig_noise(res):
    q, lg = res["quadratic"], res["logistic"]
    opts = ["sgdm", "adamw", "amsgrad", "sradam"]
    lab = {"sgdm": "SGD with momentum", "adamw": "AdamW",
           "amsgrad": "AdamW (AMSGrad)", "sradam": "SR-Adam"}
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    taus = [r["tau"] for r in q]
    for o, st in zip(opts, ["v:", "s--", "^-.", "o-"]):
        ax[0].loglog(taus, [r[o] for r in q], st, ms=3, label=lab[o])
    ax[0].set_xlabel(r"gradient-noise scale $\tau$")
    ax[0].set_ylabel("terminal objective")
    ax[0].legend(loc="upper left")
    fr = [r["flip"] for r in lg]
    for o, st in zip(opts, ["v:", "s--", "^-.", "o-"]):
        ax[1].plot(fr, [r[o] for r in lg], st, ms=3, label=lab[o])
    ax[1].set_xlabel("label-flip rate")
    ax[1].set_ylabel("test log-loss")
    fig.savefig(os.path.join(FIGS, "sim_noise_levels.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    res = load("simulation_results.json")
    fig_risk_snr(res)
    fig_variance(res)
    fig_heavy(res)
    fig_effdim(load("simulation_results3.json"))
    fig_batch(load("simulation_batch.json"))
    fig_noise(load("simulation_noise.json"))
    # LaTeX includes the figures from ../paper, so mirror every freshly
    # written PDF there too (otherwise the manuscript keeps embedding a
    # stale copy, which is exactly how paper/sim_noise_levels.pdf ended up
    # one revision behind out/simulation_noise.json).
    for name in ("sim_risk_snr", "sim_variance_bias", "sim_effective_dimension",
                 "sim_heavy_tails", "sim_batch_size", "sim_noise_levels"):
        src = os.path.join(FIGS, name + ".pdf")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(PAPER, name + ".pdf"))
    print("figures regenerated in", os.path.normpath(FIGS),
          "and mirrored to", os.path.normpath(PAPER))
