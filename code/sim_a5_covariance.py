"""
sim_a5_covariance.py — Phase B / diagnostic item a3.

Assumption (A5) requires sum_k |Cov(shat_j^2, shat_k^2)| <= C uniformly in j
and t, which drives the C/p variance bound of Theorem 8.  Under a shared mixing
variable W  ((A3') / Student noise) the coordinatewise variance estimates are
correlated through W:  Cov(g_j^2, g_k^2) = sigma_j^2 sigma_k^2 Var(W) > 0 for
every pair, and the sum of absolute covariances grows like p.

This script measures exactly that:

  * for each law x p, run the moment recursions of eq. (moments) over T steps
    with R independent replicates and keep shat^2_t (the debiased statistic of
    eq. (debiased));
  * at checkpoints t in {10, 30, 60} form the across-replicate covariance
    matrix of the p coordinatewise estimates and report

      S(p)       = mean_j sum_k |Cov(shat_j^2, shat_k^2)|     (A5 quantity),
      corr(j,k)  = mean off-diagonal correlation (diagnostic quotes 0.21 for
                   Student nu=6 and 0.002 for Gaussian),
      p*Var(mean_j shat_j^2)      (bounded iff the C/p bound of Thm 8 holds),
      excess kurtosis of the raw gradients (4th moment existence: infinite
                   for Student nu <= 4);

  * regress log S(p) on log p: slope ~ 1 falsifies A5 as stated, slope ~ 0
    supports it.

Laws:
  gaussian            (A3), iid across time, W absent          -> A5 can hold
  student_shared      W_t = nu / chi^2_nu shared over coords    -> violates A5
  student_indep       independent W_t per coordinate            -> decorrelated

Run:
    python sim_a5_covariance.py
    python sim_a5_covariance.py --smoke     # one law, small p, seconds
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "out"
OUT.mkdir(exist_ok=True)

BETA1, BETA2 = 0.9, 0.999


def phi(t, b):
    return (1.0 - b) * (1.0 - b ** (2 * t)) / ((1.0 + b) * (1.0 - b ** t) ** 2)


def run_case(law, nu, p, reps, T, checkpoints, rng):
    """Return dict at each checkpoint with the A5 diagnostics."""
    sigma = rng.uniform(0.5, 2.0, p)            # heteroscedastic scale, fixed
    s2 = sigma ** 2
    m = np.zeros((reps, p))
    v = np.zeros((reps, p))
    state4 = None                                # for kurtosis accumulation
    out = {}
    for t in range(1, T + 1):
        if law == "gaussian":
            eps = rng.standard_normal((reps, p)) * sigma
        else:
            chi2 = rng.chisquare(nu, size=(reps, 1 if law.endswith("shared")
                                           else p))
            W = nu / chi2                        # mixing variable
            z = rng.standard_normal((reps, p))
            eps = np.sqrt(W) * sigma * z         # scale-matched to marginal 1
        g = eps + 1.0                            # constant mean 1 for the moments
        m = BETA1 * m + (1.0 - BETA1) * g
        v = BETA2 * v + (1.0 - BETA2) * g * g
        if state4 is None:
            state4 = np.zeros((reps, p, 4))

        if t in checkpoints:
            mh = m / (1.0 - BETA1 ** t)
            vh = v / (1.0 - BETA2 ** t)
            corr = 1.0 - phi(t, BETA1)
            shat2 = np.maximum(vh - mh * mh, 0.0) / corr       # (reps, p)

            C = np.cov(shat2, rowvar=False, ddof=1)
            C = np.atleast_2d(C)
            absC = np.abs(C)
            S = float(absC.sum(1).mean())                      # A5 quantity
            sd = np.sqrt(np.diag(C)) + 1e-30
            Rho = C / np.outer(sd, sd)
            off = float((Rho.sum() - np.trace(Rho)) / max(1, p * p - p))
            pooled = shat2.mean(1)
            pvar = float(p * pooled.var(ddof=1))

            # kurtosis of the raw eps (not the biased-corrected statistic)
            e = g - 1.0
            Kurt = float(((e ** 4).mean() / (e ** 2).mean() ** 2) - 3.0)
            out[t] = dict(sum_abs_cov=S, mean_offdiag_corr=off,
                          p_var_pooled=pvar, excess_kurtosis=Kurt,
                          var_mean=float(pooled.var(ddof=1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--T", type=int, default=60)
    args = ap.parse_args()

    rng = np.random.default_rng(20261017)
    checkpoints = (10, 30, 60)
    if args.smoke:
        cases = [("gaussian", 0.0), ("student_shared", 6.0)]
        ps, reps, T = [10, 40], 150, 60
        checkpoints = (60,)
    else:
        cases = ([("gaussian", 0.0)]
                 + [("student_shared", nu) for nu in (3.0, 4.0, 6.0, 10.0)]
                 + [("student_indep", 5.0)])
        ps, reps, T = [10, 25, 50, 100], args.reps, args.T

    rows = []
    t0 = time.time()
    for law, nu in cases:
        for p in ps:
            res = run_case(law, nu, p, reps, T, checkpoints, rng)
            for tt, vals in res.items():
                rec = dict(law=law, nu=nu, p=p, t=tt, reps=reps, **vals)
                rows.append(rec)
                print(f"  {law:15s} nu={nu:4.1f} p={p:4d} t={tt:3d}  "
                      f"S(p)={vals['sum_abs_cov']:10.4f}  "
                      f"corr={vals['mean_offdiag_corr']:+.4f}  "
                      f"p*Var={vals['p_var_pooled']:10.4f}  "
                      f"kurt={vals['excess_kurtosis']:+.3f}", flush=True)

    # scaling of S(p) with p at the last checkpoint
    slopes = {}
    t_last = max(checkpoints)
    for law, nu in cases:
        pts = [(r["p"], r["sum_abs_cov"]) for r in rows
               if r["law"] == law and r["nu"] == nu and r["t"] == t_last]
        if len(pts) >= 2:
            x = np.log([p for p, _ in pts])
            y = np.log([max(s, 1e-12) for _, s in pts])
            slopes[f"{law}/nu={nu}"] = float(np.polyfit(x, y, 1)[0])
    print("\nlog-log slope of S(p) vs p (last checkpoint):")
    for k, s in slopes.items():
        verdict = ("S grows ~p  -> (A5) FAILS as stated" if s > 0.7 else
                   "S roughly constant -> (A5) plausible")
        print(f"  {k:24s} slope={s:+.2f}   {verdict}")

    payload = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M-%S".replace("-", ":", 0) if False
                                        else "%Y-%m-%d %H:%M:%S"),
                  seconds=time.time() - t0, reps=reps, T=T, ps=ps,
                  beta1=BETA1, beta2=BETA2,
                  note="moments from raw gradients; shat^2 debiased by 1-phi_t(beta1); "
                       "shares and covariances computed across independent replicates"),
        slopes=slopes,
        rows=rows,
    )
    (OUT / "sim_a5_covariance.json").write_text(json.dumps(payload, indent=2))
    print(f"\nelapsed {time.time() - t0:.1f}s")
    print("wrote", OUT / "sim_a5_covariance.json")


if __name__ == "__main__":
    main()