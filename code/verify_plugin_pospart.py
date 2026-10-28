"""
verify_plugin_pospart.py — Phase B / diagnostic item a4.

The positive-part dominance theorem (Thm 4) is proved for A = Sigma and the
standardised loss D = Sigma^{-1}.  Algorithm 1 uses the plug-in pair
A = D^{-1} = Sigmahat, so the standardised observation is no longer spherical
by true variances and the pairing argument does not transfer.  This script
checks empirically, with the POSITIVE-PART rule and a spread of plug-in scales
r_j = sigma_j^2 / sigmahat_j^2, whether

  * the operative constant c = p - 2 (rho = 1) still dominates, and
  * the empirical sign-change of the risk difference lies at or beyond the
    sufficient window of Corollary 2(iv):

        0 < c < 2 ( sum_j r_j - 2 max_j r_j ).

Estimator:  k_c(Y) = [1 - c/Q]^+,  Q = Y' Sigmahat^{-1} Y,
loss:       || muhat - mu ||^2_D with D = Sigmahat^{-1}.
Scale:      sigmahat_j^2 = sigma_j^2 * exp(sd * Z_j), Z_j iid N(0,1)
            (same construction as the mis-specified case in verify_theory2,
            but swept over sd and with the positive part applied).

Configurations swept:  p x sd x {homo, hetero} x mu-mode, Monte Carlo in
chunks so p = 200 with 2e5 draws stays well inside memory.

Run:
    python verify_plugin_pospart.py             # full sweep
    python verify_plugin_pospart.py --smoke     # tiny check (seconds)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "out"
OUT.mkdir(exist_ok=True)


def draws(rng, p, s2, mu, n, chunk):
    """Yield chunks of Y ~ N(mu, diag(s2)) of at most `chunk` rows."""
    done = 0
    while done < n:
        m = min(chunk, n - done)
        yield mu + rng.standard_normal((m, p)) * np.sqrt(s2)
        done += m


def assess(p, sd, profile, mu_mode, n, chunk, seed=0):
    rng = np.random.default_rng(1000 + seed + p + int(10 * sd))
    s2 = (np.ones(p) if profile == "homo"
          else rng.uniform(0.5, 2.0, p))
    sh = s2 * np.exp(sd * rng.standard_normal(p))          # plug-in scale
    r = s2 / sh                                            # r_j
    d = 1.0 / sh                                           # loss weights D
    bound = 2.0 * (r.sum() - 2.0 * r.max())

    if mu_mode == "zero":
        mu = np.zeros(p)
    else:
        snr = 4.0                                          # ||mu||^2_D / p = 4
        mu = rng.standard_normal(p) * np.sqrt(s2 * snr / p)

    # c candidates: multipliers of the bound + the operative constant p - 2
    mult = np.concatenate([np.linspace(0.0, 3.0, 61)])
    cgrid = np.unique(np.concatenate([mult * max(bound, 1e-9),
                                      [float(p - 2)]]))
    crisk = np.zeros(len(cgrid))
    base = 0.0
    total = 0
    for Y in draws(rng, p, s2, mu, n, chunk):
        Q = (Y * Y / sh).sum(1)
        base += float((((Y - mu) ** 2 * d).sum(1)).sum())
        for i, c in enumerate(cgrid):
            if c == 0.0:
                continue
            k = np.maximum(0.0, 1.0 - c / Q)
            crisk[i] += float((((k[:, None] * Y - mu) ** 2 * d).sum(1)).sum())
        total += Y.shape[0]

    crisk /= total
    base /= total
    better = crisk < base
    # empirical crossing = the smallest c > 0 at which the rule becomes WORSE
    # than the raw gradient (risk >= baseline); below it the rule dominates.
    # index 0 is c = 0, where risk equals the baseline by construction.
    worse_idx = [i for i in range(1, len(cgrid)) if crisk[i] >= base]
    cross = float(cgrid[worse_idx[0]]) if worse_idx else float("inf")
    # risk exactly at the operative constant
    op_i = int(np.argmin(np.abs(cgrid - (p - 2))))
    op_ok = bool(crisk[op_i] < base)

    return dict(
        p=p, sd=sd, profile=profile, mu_mode=mu_mode, n=n,
        bound=bound,
        crossing=(None if math.isinf(cross) else cross),
        crossing_over_bound=(None if math.isinf(cross) or bound <= 0
                             else cross / bound),
        never_worse=bool(math.isinf(cross)),
        c_pminus2=float(cgrid[op_i]), risk_at_c=crisk[op_i], risk_base=base,
        dominates_at_c_pminus2=op_ok,
        window_holds=bool(bound > 0),
        crossing_beyond_window=bool(math.isinf(cross) or
                                    (bound > 0 and cross >= 0.97 * bound)),
        sum_r=float(r.sum()), max_r=float(r.max()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    if args.smoke:
        ps, sds, n, chunk = [20], [0.0, 0.4], 20_000, 5_000
    else:
        ps, sds, n, chunk = [20, 60, 200], [0.0, 0.2, 0.4, 0.8, 1.2], 200_000, 10_000
    if args.n:
        n = args.n

    rows = []
    t0 = time.time()
    for p in ps:
        for sd in sds:
            for profile in ("homo", "hetero"):
                for mu_mode in ("zero", "moderate"):
                    rec = assess(p, sd, profile, mu_mode, n, chunk)
                    rows.append(rec)
                    cr = rec["crossing"]
                    cr_s = "never" if cr is None else f"{cr:9.3f}"
                    rb = rec["crossing_over_bound"]
                    rb_s = "  n/a" if rb is None else f"{rb:6.2f}"
                    print(f"  p={p:4d} sd={sd:3.1f} {profile:5s} mu={mu_mode:8s} "
                          f"bound={rec['bound']:9.3f} cross={cr_s} "
                          f"ratio={rb_s} "
                          f"dom@p-2={rec['dominates_at_c_pminus2']}", flush=True)

    all_ok = all(r["dominates_at_c_pminus2"] for r in rows)
    conservative = all(r["crossing_beyond_window"] for r in rows)
    payload = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  seconds=time.time() - t0, n=n, smoke=args.smoke,
                  note="positive-part rule with plug-in scale; "
                       "dominance at c = p-2 (rho=1)"),
        verdict=dict(dominance_at_c_pminus2=bool(all_ok),
                     window_sufficient=bool(conservative),
                     n_cells=len(rows),
                     n_failing=[r for r in rows if not r["dominates_at_c_pminus2"]]),
        rows=rows,
    )
    (OUT / "verify_plugin_pospart.json").write_text(json.dumps(payload, indent=2))
    print(f"\nPASS dominance at c=p-2 in {sum(r['dominates_at_c_pminus2'] for r in rows)}"
          f"/{len(rows)} cells; window sufficient in "
          f"{sum(r['crossing_beyond_window'] for r in rows)}/{len(rows)} cells")
    print(f"elapsed {time.time() - t0:.1f}s")
    print("wrote", OUT / "verify_plugin_pospart.json")


if __name__ == "__main__":
    main()