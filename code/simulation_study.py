"""
Monte Carlo studies supporting the theoretical results.

Five studies are run.

  A  Conditional risk of the gradient estimate as a function of the signal-to-noise
     ratio, comparing the raw mini-batch gradient, the fixed Adam weight, the oracle
     weight, the Stein weight with known noise variance, and the Stein weight with the
     debiased plug-in variance.
  B  Bias of the moment-based variance estimator, naive against debiased, and the
     resulting position of the shrinkage constant inside the dominance window.
  C  Risk as a function of the effective dimension under a power-law variance profile,
     locating the sign change at peff = 2 and showing what the gate prevents.
  D  Risk under Student-t gradient noise as a function of the degrees of freedom and of
     the safety factor rho.
  E  Optimisation of a synthetic non-convex problem across mini-batch sizes, recording
     the gradient norm, the realised shrinkage weight and the bias floor predicted by
     the convergence theorem.

Outputs: PDF figures in ../figs and a machine-readable summary in ../out.

Run:  python3 simulation_study.py
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
os.makedirs(FIGS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.3,
})

RESULTS = {}


def phi(t, b):
    return (1 - b) * (1 - b ** (2 * t)) / ((1 + b) * (1 - b ** t) ** 2)


# =====================================================================================
# Study A.  Conditional risk versus signal-to-noise ratio.
#
# Conditional on the past, the shrinkage target mhat_{t-1} is fixed.  Write
# delta = grad J(theta_t) - mhat_{t-1} and Y = g_t - mhat_{t-1} ~ N(delta, sigma^2 I_p).
# A direction of the form mhat_{t-1} + k Y carries risk E|| k Y - delta ||^2, so the
# whole comparison reduces to the choice of k.  The oracle weight minimising this risk
# is k_or = ||delta||^2 / ( ||delta||^2 + p sigma^2 ).
# =====================================================================================
def study_A(p=500, sigma2=1.0, n=200_000, beta1=0.9, seed=11):
    rng = np.random.default_rng(seed)
    snr_grid = np.logspace(-2.0, 1.5, 22)          # ||delta||^2 / (p sigma^2)
    w_adam = 1 - beta1                              # limiting Adam weight
    rows = []
    for snr in snr_grid:
        d2 = snr * p * sigma2
        delta = rng.normal(size=p)
        delta *= np.sqrt(d2) / np.linalg.norm(delta)
        Y = delta + rng.normal(size=(n, p)) * np.sqrt(sigma2)
        nrm2 = (Y ** 2).sum(1)

        def risk(k):
            return ((k[:, None] if np.ndim(k) else k) * Y - delta).__pow__(2).sum(1).mean()

        r_raw = risk(1.0)
        r_adam = risk(w_adam)
        k_or = d2 / (d2 + p * sigma2)
        r_oracle = risk(k_or)
        k_st = np.maximum(0.0, 1 - (p - 2) * sigma2 / nrm2)
        r_stein = risk(k_st)

        # plug-in variance: a debiased moment estimate is itself random; emulate it by a
        # multiplicative error with the sampling spread implied by the p^{-1/2} rate
        s2_hat = sigma2 * (1 + rng.normal(size=n) * 0.5 / np.sqrt(p))
        k_pl = np.maximum(0.0, 1 - (p - 2) * s2_hat / nrm2)
        r_plug = risk(k_pl)

        rows.append(dict(snr=snr, raw=r_raw, adam=r_adam, oracle=r_oracle,
                         stein=r_stein, plugin=r_plug,
                         k_oracle=k_or, k_stein_mean=float(k_st.mean())))
    RESULTS["A"] = rows

    snr = np.array([r["snr"] for r in rows])
    base = np.array([r["raw"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for key, lab, st in [("adam", r"fixed Adam weight $1-\beta_1$", "--"),
                         ("stein", "Stein weight, known $\\sigma^2$", "-"),
                         ("plugin", "Stein weight, plug-in $\\hat\\sigma^2$", "-."),
                         ("oracle", "oracle weight", ":")]:
        ax[0].plot(snr, np.array([r[key] for r in rows]) / base, st, label=lab)
    ax[0].axhline(1.0, color="k", lw=0.8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"signal-to-noise ratio $\|\delta\|^2/(p\sigma^2)$")
    ax[0].set_ylabel("risk relative to raw gradient")
    ax[0].legend(loc="lower right")

    ax[1].plot(snr, [r["k_oracle"] for r in rows], ":", label="oracle weight")
    ax[1].plot(snr, [r["k_stein_mean"] for r in rows], "-", label="mean Stein weight")
    ax[1].axhline(w_adam, ls="--", color="C1", label=r"fixed Adam weight $1-\beta_1$")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"signal-to-noise ratio $\|\delta\|^2/(p\sigma^2)$")
    ax[1].set_ylabel("direction weight")
    ax[1].legend(loc="lower right")
    fig.savefig(os.path.join(FIGS, "sim_risk_snr.pdf"))
    plt.close(fig)
    print("  study A written")


# =====================================================================================
# Study B.  Bias of the moment-based variance estimator.
# =====================================================================================
def study_B(p=2048, T=60, reps=400, beta1=0.9, beta2=0.999, seed=12):
    rng = np.random.default_rng(seed)
    s2_true = rng.uniform(0.5, 2.0, size=p)
    mu = rng.normal(size=p)
    m = np.zeros((reps, p))
    v = np.zeros((reps, p))
    naive, deb, ts = [], [], []
    for t in range(1, T + 1):
        g = mu + rng.normal(size=(reps, p)) * np.sqrt(s2_true)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        if t >= 2:
            mh = m / (1 - beta1 ** t)
            vh = v / (1 - beta2 ** t)
            raw = (vh - mh ** 2).clip(min=0)
            naive.append(float(raw.mean() / s2_true.mean()))
            deb.append(float((raw / (1 - phi(t, beta1))).mean() / s2_true.mean()))
            ts.append(t)
    RESULTS["B"] = dict(t=ts, naive_ratio=naive, debiased_ratio=deb,
                        limit_naive=2 * beta1 / (1 + beta1))

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(ts, naive, "--", label=r"$\hat v_t-\hat m_t^2$")
    ax.plot(ts, deb, "-", label=r"$(\hat v_t-\hat m_t^2)/(1-\phi_t)$")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhline(2 * beta1 / (1 + beta1), color="C0", lw=0.8, ls=":")
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel(r"estimate / true $\bar\sigma^2$")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIGS, "sim_variance_bias.pdf"))
    plt.close(fig)
    print("  study B written")


# =====================================================================================
# Study C.  Risk against effective dimension under a power-law variance profile
#           sigma_j^2 proportional to j^{-alpha}, j = 1,...,p.
# =====================================================================================
def study_C(p=500, n=120_000, seed=13):
    rng = np.random.default_rng(seed)
    alphas = np.linspace(0.0, 3.0, 16)
    rows = []
    for a in alphas:
        s2 = (np.arange(1, p + 1.0)) ** (-a)
        s2 = s2 / s2.mean()
        peff = s2.sum() / s2.max()
        delta = rng.normal(size=p) * np.sqrt(s2) * 0.5      # moderate signal
        Y = delta + rng.normal(size=(n, p)) * np.sqrt(s2)
        Q = (Y ** 2 / s2).sum(1)
        base = ((Y - delta) ** 2).sum(1).mean()
        c = max(0.0, peff - 2.0)
        k = np.maximum(0.0, 1 - c / Q)
        r = ((k[:, None] * Y - delta) ** 2).sum(1).mean()
        rows.append(dict(alpha=float(a), peff=float(peff),
                         rel_risk=float(r / base), gate_open=bool(peff > 3.0)))
    RESULTS["C"] = rows

    peffs = np.array([r["peff"] for r in rows])
    rel = np.array([r["rel_risk"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.semilogx(peffs, rel, "o-", ms=3)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axvline(2.0, color="C3", ls="--", lw=0.9)
    ax.axvline(3.0, color="C2", ls=":", lw=0.9)
    ax.set_xlabel(r"effective dimension $p_{\mathrm{eff}}=\mathrm{tr}\Sigma/\lambda_{\max}(\Sigma)$")
    ax.set_ylabel("risk relative to raw gradient")
    ax.annotate(r"$p_{\mathrm{eff}}=2$", xy=(2.0, rel.max()), xytext=(2.1, rel.max()),
                fontsize=7, color="C3")
    ax.annotate("gate", xy=(3.0, rel.min()), xytext=(3.1, rel.min()),
                fontsize=7, color="C2")
    fig.savefig(os.path.join(FIGS, "sim_effective_dimension.pdf"))
    plt.close(fig)
    print("  study C written")


# =====================================================================================
# Study D.  Student-t gradient noise.
# =====================================================================================
def study_D(p=300, n=150_000, seed=14):
    rng = np.random.default_rng(seed)
    nus = [3, 4, 5, 8, 15, 50]
    rhos = [1.0, 0.75, 0.5]
    rows = []
    for nu in nus:
        W = nu / rng.chisquare(nu, size=n)
        delta = rng.normal(size=p) * 0.4
        Y = delta + rng.normal(size=(n, p)) * np.sqrt(W)[:, None]     # scale sigma^2 = 1
        nrm2 = (Y ** 2).sum(1)
        base = ((Y - delta) ** 2).sum(1).mean()
        s2_marg = nu / (nu - 2)          # marginal variance, what the estimator targets
        rec = dict(nu=nu, base=float(base))
        for rho in rhos:
            c = rho * (p - 2) * s2_marg
            k = np.maximum(0.0, 1 - c / nrm2)
            rec[f"rho{rho}"] = float(((k[:, None] * Y - delta) ** 2).sum(1).mean() / base)
        rows.append(rec)
    RESULTS["D"] = rows

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for rho in rhos:
        ax.plot(nus, [r[f"rho{rho}"] for r in rows], "o-", ms=3, label=rf"$\rho={rho}$")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(nus)
    ax.set_xticklabels([str(v) for v in nus])
    ax.set_xlabel(r"degrees of freedom $\nu$ of the gradient noise")
    ax.set_ylabel("risk relative to raw gradient")
    ax.legend()
    fig.savefig(os.path.join(FIGS, "sim_heavy_tails.pdf"))
    plt.close(fig)
    print("  study D written")


# =====================================================================================
# Study E.  Optimisation across mini-batch sizes on a synthetic non-convex problem.
#
# A two-layer network with a fixed teacher is fitted by SR-Adam and by AdamW under
# identical step sizes and identical mini-batch streams.  The convergence theorem says
# that the stationarity gap of SR-Adam is bounded by a term vanishing with the step count
# plus a floor proportional to rho * trace(Sigmahat), and trace(Sigmahat) falls like 1/B
# in the batch size B.  The study records the realised floor alongside the gradient norm.
# =====================================================================================
def study_E(seed=15):
    import torch
    from sr_adam import SRAdam

    d, h, k, N = 120, 48, 8, 40_000
    torch.manual_seed(seed)
    W1 = torch.randn(d, 40) / np.sqrt(d)
    W2 = torch.randn(40, k) / np.sqrt(40)
    X = torch.randn(N, d)
    logits = torch.tanh(X @ W1) @ W2
    y = (logits + 0.5 * torch.randn(N, k)).argmax(1)

    lossf = torch.nn.CrossEntropyLoss()

    def full_grad_norm(net):
        net.zero_grad()
        lossf(net(X), y).backward()
        gn = torch.sqrt(sum((q.grad ** 2).sum() for q in net.parameters())).item()
        net.zero_grad()
        return gn

    def run(opt_name, B, steps=1200):
        torch.manual_seed(100 + seed)
        net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh(),
                                  torch.nn.Linear(h, k))
        if opt_name == "sradam":
            opt = SRAdam(net.parameters(), lr=3e-3, warmup=5, track_stats=True)
        else:
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.0)
        gen = torch.Generator().manual_seed(999)
        floors, weights = [], []
        for i in range(steps):
            idx = torch.randint(0, N, (B,), generator=gen)
            opt.zero_grad()
            lossf(net(X[idx]), y[idx]).backward()
            opt.step()
            if opt_name == "sradam" and i > 200 and opt.stats:
                floors.append(sum(s["sigma2"] * s["numel"] for s in opt.stats
                                  if s["sigma2"] == s["sigma2"]))
                weights.append(np.mean([s["shrink_weight"] for s in opt.stats]))
        with torch.no_grad():
            final_loss = lossf(net(X), y).item()
        return dict(grad_norm=full_grad_norm(net), loss=final_loss,
                    floor=float(np.mean(floors)) if floors else float("nan"),
                    weight=float(np.mean(weights)) if weights else float("nan"))

    batches = [8, 16, 32, 64, 128, 256, 512]
    rows = []
    for B in batches:
        a = run("adamw", B)
        s = run("sradam", B)
        rows.append(dict(B=B, adam_loss=a["loss"], sr_loss=s["loss"],
                         adam_gn=a["grad_norm"], sr_gn=s["grad_norm"],
                         floor=s["floor"], weight=s["weight"]))
        print(f"    B={B:4d}  AdamW loss={a['loss']:.4f} |grad|={a['grad_norm']:.4f}"
              f"   SR-Adam loss={s['loss']:.4f} |grad|={s['grad_norm']:.4f}"
              f"   tr(Sigmahat)={s['floor']:.3e}  mean weight={s['weight']:.3f}")
    RESULTS["E"] = rows

    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.4))
    ax[0].semilogx([r["B"] for r in rows], [r["adam_loss"] for r in rows], "s--", ms=3,
                   label="AdamW")
    ax[0].semilogx([r["B"] for r in rows], [r["sr_loss"] for r in rows], "o-", ms=3,
                   label="SR-Adam")
    ax[0].set_xlabel("mini-batch size $B$")
    ax[0].set_ylabel("training loss after 1200 steps")
    ax[0].legend()

    ax[1].loglog([r["B"] for r in rows], [r["floor"] for r in rows], "o-", ms=3,
                 label=r"$\mathrm{tr}\,\hat\Sigma$")
    ref = rows[0]["floor"] * np.array([rows[0]["B"] / r["B"] for r in rows])
    ax[1].loglog([r["B"] for r in rows], ref, "k:", label=r"$\propto B^{-1}$")
    ax[1].set_xlabel("mini-batch size $B$")
    ax[1].set_ylabel("estimated noise trace")
    ax[1].legend()

    ax[2].semilogx([r["B"] for r in rows], [r["weight"] for r in rows], "o-", ms=3,
                   label="mean Stein weight")
    ax[2].axhline(0.1, ls="--", color="C1", label=r"fixed Adam weight")
    ax[2].set_xlabel("mini-batch size $B$")
    ax[2].set_ylabel("realised direction weight")
    ax[2].legend()
    fig.savefig(os.path.join(FIGS, "sim_batch_size.pdf"))
    plt.close(fig)
    print("  study E written")


if __name__ == "__main__":
    print("study A: risk versus signal-to-noise ratio")
    study_A()
    print("study B: variance estimator bias")
    study_B()
    print("study C: effective dimension")
    study_C()
    print("study D: heavy-tailed gradient noise")
    study_D()
    print("study E: batch-size regimes")
    study_E()
    with open(os.path.join(OUT, "simulation_results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nall studies complete; results in ../out/simulation_results.json")
