"""
conv_constants.py — Phase B / diagnostic item b1 (numbers, not proofs).

The convergence theorem's bound (eq. conv_bound) contains

    kappa = p_max / p_min,   p_max = 1/eps,  p_min = 1/(G_inf + eps),

which for eps = 1e-8 is about 1e8 * G_inf and therefore numerically vacuous —
the diagnostic asks for the *numbers* so Dr. Arashi can restate the theorem
with piecewise B_t and a direct bias bound.  This script produces them:

  Section 1  kappa and the bound terms at the eps / G_inf actually used.
  Section 2  realised floor behaviour from out/simulation_batch.json:
             the tr(Sigmahat) column against 1/B, its log-log slope and the
             per-doubling fall factor (manuscript claims 1.86 vs the
             predicted 2).
  Section 3  Monte Carlo for the sharp displacement bound of Lemma 2 vs the
             *bias* it bounds:  E[(1-k)||Y||]  and  ||E[d] - mu||  against
             sqrt(c * lambda_max(Sigma)), showing how loose the floor is and
             supplying 'bound the bias directly' with a number.

Outputs: ../out/conv_constants.json and a Markdown report printed to stdout.

Run:
    python conv_constants.py
    python conv_constants.py --smoke      # fewer Monte Carlo draws
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
HERE = Path(__file__).resolve().parent


# -------------------------------------------------------------- section 1


def kappa_table(epsilons=(1e-4, 1e-6, 1e-8), g_inf=(1.0, 10.0, 100.0)):
    rows = []
    for eps in epsilons:
        for g in g_inf:
            p_max = 1.0 / eps
            p_min = 1.0 / (g + eps)
            k = p_max / p_min
            rows.append(dict(eps=eps, G_inf=g, p_max=p_max, p_min=p_min,
                             kappa=k))
    lines = ["### 1. kappa = p_max / p_min", "",
             "| eps | G_inf | p_max | p_min | kappa |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['eps']:.0e} | {r['G_inf']:g} | {r['p_max']:.3g} | "
                     f"{r['p_min']:.3g} | {r['kappa']:.3g} |")
    lines += ["",
              "The floor term kappa * E[B_t^2] therefore needs E[B_t^2] <~ 1e-9 "
              "to be informative at eps = 1e-8; report these constants in the "
              "restated Theorem 9.", ""]
    return rows, lines


# -------------------------------------------------------------- section 2


def batch_floor():
    p = OUT / "simulation_batch.json"
    if not p.exists():
        return None, ["### 2. floor vs batch size: simulation_batch.json missing", ""]
    rows = json.loads(p.read_text())
    rows = sorted(rows, key=lambda r: r["B"])
    lines = ["### 2. realised floor vs batch size (tr Sigmahat)", "",
             "| B | tr(Sigmahat) | fall vs previous doubling | mean Stein weight |",
             "|---|---|---|---|"]
    falls = []
    prev = None
    for r in rows:
        f = (prev / r["floor"]) if (prev and r["floor"] > 0) else None
        if f:
            falls.append(f)
        lines.append(f"| {r['B']} | {r['floor']:.4g} | "
                     f"{f if f is None else f'{f:.2f}x'} | {r['weight']:.3f} |")
        prev = r["floor"]
    Bs = np.log(np.array([r["B"] for r in rows], float))
    Fl = np.log(np.array([max(r["floor"], 1e-300) for r in rows], float))
    slope = float(np.polyfit(Bs, Fl, 1)[0])
    mean_fall = float(np.mean(falls)) if falls else float("nan")
    lines += ["",
              f"log-log slope of tr(Sigmahat) vs B = {slope:+.3f} "
              f"(prediction: -1); mean per-doubling fall = {mean_fall:.2f}x "
              f"(prediction 2.00x).", ""]
    summary = dict(slope=slope, mean_fall=mean_fall,
                   predicted_fall=2.0, n_rows=len(rows))
    return summary, lines


# -------------------------------------------------------------- section 3


def displacement_vs_bias(p, snr, n, chunk, sd_r=0.4, seed=5, rho=1.0):
    """MC: sharp displacement bound vs mean displacement vs actual bias."""
    rng = np.random.default_rng(seed + p)
    s2 = rng.uniform(0.5, 2.0, p)
    sh = s2 * np.exp(sd_r * rng.standard_normal(p))     # plug-in scale
    delta = rng.standard_normal(p)
    delta = delta / math.sqrt(float((delta ** 2 / s2).sum())) * math.sqrt(snr * p)
    A = sh
    D = 1.0 / sh
    c = rho * (p - 2)
    bound = math.sqrt(c * float(s2.max()))              # Lemma 2 with true lambda_max
    bound_hat = math.sqrt(c * float((sh + 1e-12).max()))

    ek = 0.0                                            # E[k] not enough; accumulate
    acc_ed = np.zeros(p)                                # E[k Y]
    acc_disp = 0.0                                      # E[(1-k)||Y||]
    total = 0
    done = 0
    while done < n:
        m = min(chunk, n - done)
        Y = delta[None, :] + rng.standard_normal((m, p)) * np.sqrt(s2)
        Q = (Y * Y / A).sum(1)
        k = np.maximum(0.0, 1.0 - c / Q)
        d = k[:, None] * Y
        acc_ed += d.sum(0)
        acc_disp += float((np.maximum(0.0, 1.0 - c / Q)
                           * np.linalg.norm(Y, axis=1)).sum())
        total += m
        done += m
    e_d = acc_ed / total
    bias = e_d - delta
    edisp = acc_disp / total
    return dict(p=p, snr=snr, n=total, c=c,
                bound=bound, bound_hat=bound_hat,
                mean_displacement=edisp, bias_norm=float(np.linalg.norm(bias)),
                bound_over_displacement=bound / edisp if edisp > 0 else float("nan"),
                bound_over_bias=(bound / float(np.linalg.norm(bias))
                                 if np.linalg.norm(bias) > 0 else float("nan")),
                bias_over_displacement=(float(np.linalg.norm(bias)) / edisp
                                        if edisp > 0 else float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    krows, klines = kappa_table()
    bsum, blines = batch_floor()

    n = 50_000 if args.smoke else 300_000
    chunk = 10_000
    disp = []
    lines = ["### 3. displacement bound vs actual bias (MC)", "",
             "| p | snr | bound sqrt(c*lammax) | E[(1-k)||Y||] | ||bias|| | "
             "bound/disp | bound/bias | bias/disp |",
             "|---|---|---|---|---|---|---|---|"]
    for p in (20, 100):
        for snr in (0.5, 2.0):
            r = displacement_vs_bias(p, snr, n, chunk)
            disp.append(r)
            lines.append(f"| {r['p']} | {r['snr']:g} | {r['bound']:.3f} | "
                         f"{r['mean_displacement']:.4f} | {r['bias_norm']:.5f} | "
                         f"{r['bound_over_displacement']:.1f}x | "
                         f"{r['bound_over_bias']:.0f}x | "
                         f"{r['bias_over_displacement']:.3f} |")
    bos = [r["bound_over_bias"] for r in disp
           if r["bound_over_bias"] == r["bound_over_bias"]]
    bdis = [r["bias_over_displacement"] for r in disp
            if r["bias_over_displacement"] == r["bias_over_displacement"]]
    lines += ["",
              "The floor term uses the bound (column 3), while the quantity the "
              "theorem actually needs is the bias (column 5).  In these "
              f"configurations the bound is {min(bos):.1f}x to {max(bos):.1f}x "
              f"the bias, and the bias is {min(bdis):.2f}x to {max(bdis):.2f}x "
              "the mean instantaneous displacement — quote these measured "
              "ratios (and re-measure at full --n) when restating the floor "
              "discussion; do not assert a magnitude the numbers do not "
              "support (item b1).", ""]

    payload = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             seconds=time.time() - t0, n=n, smoke=args.smoke),
                   kappa=krows, batch_floor=bsum, displacement=disp)
    (OUT / "conv_constants.json").write_text(json.dumps(payload, indent=2))

    report = "\n".join(klines + blines + lines)
    print(report)
    print("wrote", OUT / "conv_constants.json")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()