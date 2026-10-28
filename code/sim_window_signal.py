"""
sim_window_signal.py — Phase B / diagnostic item b2.

The dominance window of Theorem 5 / Corollary 6 is only *proved* at mu = 0
(the manuscript says so in the remark after Corollary 6), while Table 4
(heavy tails) reports no signal level.  The diagnostic computes the weighted
mean of the mixing variable at several signal norms (nu = 5, p = 30: 1.00,
1.05, 1.18, 1.36, 1.58 at ||mu|| = 0, 2, 5, 10, 30) and asks for both a
signal sweep and the table to carry it.

Part 1 — Wbar_A(mu):  Monte Carlo for

        Wbar_A(mu) = E[W / Q] / E[1 / Q],      Q = Y' A^{-1} Y,

with Y = mu + sqrt(W) z, z ~ N(0, I_p), W = nu / chi^2_nu (Student),
A = S = (nu/(nu-2)) I the marginal scale (homoscedastic, D = I as in
Corollary 6), sweeping ||mu|| in {0, 2, 5, 10, 30}.  Check: at mu = 0 the
closed form is 1 / E[W^{-1}] = 1 exactly.

Part 2 — heavy-tail risk table with an explicit signal level: risk of the
positive-part rule relative to the raw gradient, for safety factors
rho in {1, 0.75, 0.5} and signal levels ||mu||^2 / (p s^2) in {0, 1, 4},
at nu in {3, 5, 15}.  This is the grid Table 4 should show.

Outputs:
  ../out/sim_window_signal.json
  ../figs/sim_window_signal.pdf/.png   (left: Wbar vs ||mu||; right: risk vs signal)

Run:
    python sim_window_signal.py
    python sim_window_signal.py --smoke      # fewer draws/points, seconds
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def student_noise(rng, n, p, nu, chunk):
    """Y = sqrt(W) z with W = nu/chi2_nu, one row-broadcast W per chunk."""
    done = 0
    while done < n:
        m = min(chunk, n - done)
        W = nu / rng.chisquare(nu, size=(m, 1))
        yield np.sqrt(W) * rng.standard_normal((m, p))
        done += m


# ------------------------------------------------------------------ part one


def wbar_curve(nu, p, mu_norms, n, chunk, seed=7):
    rng = np.random.default_rng(seed)
    s2 = nu / (nu - 2.0)                    # marginal variance, A = S = s2 I
    out = []
    for nm in mu_norms:
        v = rng.standard_normal(p)
        mu = v / np.linalg.norm(v) * nm
        # E[W/Q] and E[1/Q] must both carry W alongside Y = mu + sqrt(W) z
        e_wq = 0.0
        e_invq = 0.0
        total = 0
        done = 0
        while done < n:
            m = min(chunk, n - done)
            W = nu / rng.chisquare(nu, size=(m, 1))
            Y = mu + np.sqrt(W) * rng.standard_normal((m, p))
            nq = (Y * Y).sum(1)
            Q = nq / s2
            e_wq += float((W[:, 0] / Q).sum())
            e_invq += float((1.0 / Q).sum())
            total += m
            done += m
        wb = (e_wq / total) / (e_invq / total)
        closed = 1.0 / 1.0                  # 1/E[W^{-1}] = 1 for Student
        out.append(dict(mu_norm=nm, wbar=wb, mu0_closed_form=closed))
        print(f"  mu||={nm:5.1f}  Wbar={wb:.4f}"
              + ("   (closed form at mu=0: 1.0000)" if nm == 0 else ""),
              flush=True)
    return out


# ------------------------------------------------------------------ part two


def risk_vs_signal(nu, p, rho, snr, n, chunk, seed=21):
    """Relative risk of positive-part rule calibrated to the marginal."""
    rng = np.random.default_rng(seed)
    s2 = nu / (nu - 2.0)                     # marginal variance
    a = rho * (p - 2) * s2                   # JS constant with rho safety factor
    rel, base_rel = [], []
    done = 0
    while done < n:
        m = min(chunk, n - done)
        W = nu / rng.chisquare(nu, size=(m, 1))
        Z = rng.standard_normal((m, p))
        # signal so that E||mu||^2 / (p s2) = snr  ->  ||mu||^2 = snr * p * s2
        mu = rng.standard_normal(p)
        mu = mu / np.linalg.norm(mu) * math.sqrt(snr * p * s2)
        Y = mu[None, :] + np.sqrt(W) * Z
        Q = (Y * Y).sum(1) / s2              # A = S = s2 I
        k = np.maximum(0.0, 1.0 - a / Q)
        base_rel.append(float((((Y - mu) ** 2).sum(1)).sum()))
        rel.append(float((((k[:, None] * Y - mu) ** 2).sum(1)).sum()))
        done += m
    base = sum(base_rel) / done
    r = sum(rel) / done
    return r, base, r / (p * s2), base / (p * s2)


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=200_000)
    args = ap.parse_args()

    nu, p = 5.0, 30
    few = args.smoke
    n = 40_000 if few else args.n
    chunk = 5_000 if few else 20_000
    mus = [0.0, 5.0, 30.0] if few else [0.0, 2.0, 5.0, 10.0, 30.0]
    nus = (5.0,) if few else (3.0, 5.0, 15.0)
    rhos = (1.0, 0.5) if few else (1.0, 0.75, 0.5)
    snrs = (0.0, 4.0) if few else (0.0, 1.0, 4.0)

    t0 = time.time()
    print("part 1: Wbar_A(mu)")
    curve = wbar_curve(nu, p, mus, n, chunk)
    mu0 = next((c["wbar"] for c in curve if c["mu_norm"] == 0), None)

    print("\npart 2: heavy-tail risk by signal level")
    rows = []
    for vv in nus:
        for r_ in rhos:
            for snr in snrs:
                rel, base, rel_n, base_n = risk_vs_signal(vv, p, r_, snr, n, chunk)
                rec = dict(nu=vv, rho=r_, snr=snr, risk_relative=rel_n,
                           risk_raw=base_n, reduction=1.0 - rel_n / base_n)
                rows.append(rec)
                print(f"  nu={vv:4.1f} rho={r_:.2f} snr={snr:3.1f}  "
                      f"rel.risk={rel_n:.4f}  reduction={rec['reduction']:+.3f}",
                      flush=True)

    payload = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  seconds=time.time() - t0, n=n, smoke=few, p=p, nu=nu,
                  note="Part 1: Wbar = E[W/Q]/E[1/Q] with A = marginal scale; "
                       "Part 2: positive-part rule a = rho(p-2)s^2, risk /= p s^2"),
        wbar_curve=curve,
        wbar_mu0_check=dict(empirical=mu0, closed_form=1.0,
                            abs_error=(abs(mu0 - 1.0) if mu0 is not None else None)),
        risk_rows=rows,
    )
    (OUT / "sim_window_signal.json").write_text(json.dumps(payload, indent=2))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax[0].plot([c["mu_norm"] for c in curve], [c["wbar"] for c in curve],
               "o-", label=r"$\bar W_{A}(\mu)$ MC")
    ax[0].axhline(1.0, color="k", ls="--", lw=0.8, label="1 (mu=0 closed form)")
    ax[0].set_xlabel(r"$\|\mu\|$")
    ax[0].set_ylabel(r"$\bar W_{A}$")
    ax[0].set_title(rf"Window multiplier (nu={nu:g}, p={p})")
    ax[0].legend(fontsize=7)
    for vv in nus:
        for r_ in rhos:
            pts = [x for x in rows if x["nu"] == vv and x["rho"] == r_]
            if pts:
                ax[1].plot([x["snr"] for x in pts], [x["risk_relative"] for x in pts],
                           marker="o", ms=3,
                           label=f"nu={vv:g}, rho={r_:g}")
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].set_xlabel(r"signal $\|\mu\|^2/(p s^2)$")
    ax[1].set_ylabel("relative risk (raw = 1)")
    ax[1].set_title("Positive-part risk by signal level")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        try:
            fig.savefig(FIGS / f"sim_window_signal.{ext}")
        except PermissionError:
            pass
    plt.close(fig)

    print(f"\nelapsed {time.time() - t0:.1f}s")
    print("wrote", OUT / "sim_window_signal.json")
    print("wrote", FIGS / "sim_window_signal.pdf")


if __name__ == "__main__":
    main()