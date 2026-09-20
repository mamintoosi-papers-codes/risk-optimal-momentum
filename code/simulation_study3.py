"""
Study C, corrected.

The quantity that governs a preconditioned optimiser is the estimation error measured in
the geometry the preconditioner imposes, not in the raw Euclidean geometry.  The study
therefore evaluates every rule under the standardised loss

    L( muhat, mu ) = ( muhat - mu )' A^{-1} ( muhat - mu ),

A being the diagonal scale matrix the rule itself uses to whiten.  Three rules are
compared on the same heteroscedastic gradient blocks, with the variance profile
sigma_j^2 proportional to j^{-alpha} normalised to unit mean.

  pooled       k = [ 1 - (p-2) sigmabar^2 / ||Y||^2 ]^+ , one pooled scalar variance and
               the nominal dimension.  This is the rule of the first version of this work
               and the one an all-blocks application puts on every layer.
  conservative k = [ 1 - (peff - 2) / Q ]^+ with Q = sum_j Y_j^2 / sigma_j^2 and
               peff = tr(Sigma)/lambda_max(Sigma).  The rule proposed here.
  standardised k = [ 1 - (p-2) / Q ]^+ , the largest constant admissible under the
               standardised loss when the coordinatewise variances are exact.

The unweighted Euclidean loss is reported alongside, so that the difference the loss
weighting makes is visible rather than assumed.
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")
OUT = os.path.join(os.path.dirname(__file__), "..", "out")

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "lines.linewidth": 1.3,
})


def run(p=400, n=60_000, snr=1.0, seed=31):
    """snr = ||delta||^2_{Sigma^{-1}} / p, the standardised signal-to-noise ratio."""
    rng = np.random.default_rng(seed)
    alphas = np.linspace(0.0, 3.0, 19)
    rows = []
    for a in alphas:
        s2 = (np.arange(1, p + 1.0)) ** (-a)
        s2 = s2 / s2.mean()
        peff = s2.sum() / s2.max()

        # signal with standardised norm ||delta||^2_{Sigma^{-1}} = snr * p
        z = rng.normal(size=p)
        delta = z * np.sqrt(s2)
        delta *= np.sqrt(snr * p) / np.sqrt((delta ** 2 / s2).sum())

        Y = (delta + rng.normal(size=(n, p)) * np.sqrt(s2)).astype(np.float32)
        s2f = s2.astype(np.float32); deltaf = delta.astype(np.float32)
        Qw = (Y ** 2 / s2f).sum(1)               # whitened, A = Sigma
        nrm2 = (Y ** 2).sum(1)                  # unwhitened, A = sigmabar^2 I = I

        def loss_std(k):                        # standardised loss, A = Sigma
            return float((((k[:, None] if np.ndim(k) else k) * Y - deltaf) ** 2 / s2f).sum(1).mean())

        def loss_euc(k):
            return float((((k[:, None] if np.ndim(k) else k) * Y - deltaf) ** 2).sum(1).mean())

        base_std, base_euc = loss_std(1.0), loss_euc(1.0)

        k_pool = np.maximum(0.0, 1 - (p - 2) * 1.0 / nrm2)      # sigmabar^2 = 1
        k_cons = np.maximum(0.0, 1 - max(0.0, peff - 2.0) / Qw)
        k_stdz = np.maximum(0.0, 1 - (p - 2) / Qw)

        rows.append(dict(
            alpha=float(a), peff=float(peff),
            pooled_std=float(loss_std(k_pool) / base_std),
            cons_std=float(loss_std(k_cons) / base_std),
            stdz_std=float(loss_std(k_stdz) / base_std),
            pooled_euc=float(loss_euc(k_pool) / base_euc),
            cons_euc=float(loss_euc(k_cons) / base_euc),
            stdz_euc=float(loss_euc(k_stdz) / base_euc),
        ))
        print(f"    alpha={a:4.2f} peff={peff:8.2f} | standardised loss:"
              f" pooled={rows[-1]['pooled_std']:7.4f}"
              f" conservative={rows[-1]['cons_std']:7.4f}"
              f" standardised={rows[-1]['stdz_std']:7.4f}"
              f" | euclidean: pooled={rows[-1]['pooled_euc']:7.4f}"
              f" conservative={rows[-1]['cons_euc']:7.4f}")
    return rows


if __name__ == "__main__":
    out = {}
    for snr in [0.25, 1.0, 4.0]:
        print(f"  standardised signal-to-noise ratio {snr}")
        out[f"snr{snr}"] = run(snr=snr, seed=31 + int(10 * snr))

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.5), sharey=True)
    for ax, snr in zip(axes, [0.25, 1.0, 4.0]):
        rows = out[f"snr{snr}"]
        pe = [r["peff"] for r in rows]
        ax.semilogx(pe, [r["pooled_std"] for r in rows], "s--", ms=3,
                    label=r"pooled scalar, $c=p-2$")
        ax.semilogx(pe, [r["stdz_std"] for r in rows], "^:", ms=3,
                    label=r"coordinatewise, $c=p-2$")
        ax.semilogx(pe, [r["cons_std"] for r in rows], "o-", ms=3,
                    label=r"coordinatewise, $c=p_{\mathrm{eff}}-2$")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlabel(r"$p_{\mathrm{eff}}=\mathrm{tr}\Sigma/\lambda_{\max}(\Sigma)$")
        ax.set_title(rf"$\|\delta\|^2_{{\Sigma^{{-1}}}}/p = {snr}$")
    axes[0].set_ylabel("risk relative to raw gradient")
    axes[0].set_ylim(0, 2.6)
    axes[2].legend(loc="upper left")
    fig.savefig(os.path.join(FIGS, "sim_effective_dimension.pdf"))
    plt.close(fig)

    with open(os.path.join(OUT, "simulation_results3.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwritten")
