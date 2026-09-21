"""
Path-2 phase diagram: controlled optimisation where the theory's assumptions hold.

No named datasets (no CIFAR / UCI). Every problem is synthetic with a known
data-generating process (DGP), so the gradient-noise level, dimension, and tail
weight can be set directly and the SR-Adam mechanism can be measured.

Problems (DGPs)
---------------
1. quadratic   — (1/2)(θ-θ*)'H(θ-θ*), H diagonal, exact gradient + additive noise
                 ε ~ N(0, τ²I) or Student-t scaled to the same variance.
2. ridge       — linear regression y = Xθ* + σ_y ξ, X ~ N(0, Σ_x) with optional
                 condition number; mini-batch (or full) least-squares gradient.
3. logistic    — multinomial logistic, teacher W*, optional label flips (paper §opt).
4. mlp         — 1-hidden-layer teacher → student, mini-batch CE/MSE, true
                 stochastic gradients from data noise.

Axes swept (subset via CLI)
---------------------------
  p      parameter dimension (block size)
  tau    additive gradient-noise scale (quadratic) or label/feature noise
  law    gaussian | student (Student-t, ν=5, variance-matched)
  batch  mini-batch size (ridge / logistic / mlp)

Outputs
-------
  ../out/phase_diagram.json
  ../figs/phase_diagram_heatmap.pdf
  ../figs/phase_diagram_slices.pdf

Run (smoke):  python phase_diagram.py --smoke
Run (full):   python phase_diagram.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sr_adam import SRAdam

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figs")
OUT = os.path.join(HERE, "..", "out")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "lines.linewidth": 1.3,
})

OPTS = ["sgdm", "adamw", "amsgrad", "sradam"]
LABEL = {"sgdm": "SGD-M", "adamw": "AdamW", "amsgrad": "AMSGrad", "sradam": "SR-Adam"}


def build(name: str, params, lr: float):
    if name == "sgdm":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if name == "amsgrad":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, amsgrad=True)
    return SRAdam(params, lr=lr, warmup=5)


def student_noise(shape, tau: float, gen: torch.Generator) -> torch.Tensor:
    nu = 5.0
    z = torch.randn(shape, generator=gen)
    # variance of t_nu is nu/(nu-2); scale to variance tau^2
    scale = tau * math.sqrt((nu - 2.0) / nu)
    if len(shape) == 1:
        chi = torch.randn(shape, generator=gen).abs().add(0.1)  # fallback unused
        chi = torch.distributions.Chi2(torch.tensor([nu])).sample((shape[0],)).view(-1)
    else:
        chi = torch.distributions.Chi2(torch.tensor([nu])).sample((shape[0],)).squeeze(-1)
        chi = chi.view(shape[0], *([1] * (len(shape) - 1)))
    return scale * z / (chi.sqrt() + 1e-8)


def paired_summary(per, seeds):
    rec = {}
    for o in OPTS:
        a = np.array(per[o], dtype=float)
        rec[o] = float(a.mean())
        rec[o + "_sd"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        rec[o + "_all"] = [float(x) for x in a]
    d = np.array(per["adamw"]) - np.array(per["sradam"])
    rec["paired_diff"] = float(d.mean())
    rec["t_stat"] = float(d.mean() / (d.std(ddof=1) / math.sqrt(len(seeds)))) if d.std(ddof=1) > 0 else 0.0
    rec["win_sradam"] = bool(d.mean() > 0)  # lower objective better → adamw - sradam > 0
    return rec


# --------------------------------------------------------------------- problem: quadratic
def run_quadratic(p=400, tau=1.0, law="gaussian", steps=2500, seeds=(1, 2, 3, 4, 5),
                  cond=40.0, lr=None):
    H = torch.linspace(0.05, 0.05 * cond, p)
    per = {o: [] for o in OPTS}
    k_means = []
    for seed in seeds:
        torch.manual_seed(seed)
        x_star = torch.randn(p)
        gen = torch.Generator().manual_seed(1000 + seed)
        for o in OPTS:
            torch.manual_seed(seed)
            x = torch.zeros(p, requires_grad=True)
            opt = build(o, [x], lr or (2e-2 if o != "sgdm" else 5e-2))
            ks = []
            for _ in range(steps):
                if law == "gaussian":
                    eps = tau * torch.randn(p, generator=gen)
                else:
                    eps = student_noise((p,), tau, gen)
                x.grad = H * (x.detach() - x_star) + eps
                opt.step()
                if o == "sradam" and getattr(opt, "stats", None):
                    ks.append(opt.stats[0]["shrink_weight"])
            per[o].append(float(0.5 * (H * (x.detach() - x_star) ** 2).sum()))
            if o == "sradam" and ks:
                k_means.append(float(np.mean(ks)))
    rec = paired_summary(per, seeds)
    rec.update(problem="quadratic", p=p, tau=tau, law=law, cond=cond,
                mean_k=float(np.mean(k_means)) if k_means else None)
    return rec


# ------------------------------------------------------------------- problem: ridge regression
def run_ridge(p=100, N=2000, batch=64, noise_y=1.0, cond=20.0, steps=1500,
              seeds=(1, 2, 3), law="gaussian", lr=None):
    """Mini-batch ridge-style squared loss on synthetic linear regression."""
    per = {o: [] for o in OPTS}
    for seed in seeds:
        torch.manual_seed(seed)
        evals = torch.linspace(1.0, cond, p)
        # correlated features: X = Z * sqrt(S) via random orthonormal-ish scaling
        Q, _ = torch.linalg.qr(torch.randn(p, p))
        sqrt_S = torch.diag(evals.clamp(min=0.1).rsqrt() * math.sqrt(p))
        # X = U diag(sqrt(s)) so Cov ≈ diag(1/s) — use s as condition of Hessian
        W_true = torch.randn(p)
        X = torch.randn(N, p) @ (Q.T * evals.rsqrt())
        y = X @ W_true
        if law == "gaussian":
            y = y + noise_y * torch.randn(N)
        else:
            y = y + student_noise((N,), noise_y, torch.Generator().manual_seed(seed + 55))
        Xte = torch.randn(2000, p) @ (Q.T * evals.rsqrt())
        yte = Xte @ W_true + 0.0  # evaluate at noise-free truth for risk
        for o in OPTS:
            torch.manual_seed(seed)
            w = torch.zeros(p, requires_grad=True)
            opt = build(o, [w], lr or (1e-2 if o != "sgdm" else 3e-2))
            gen = torch.Generator().manual_seed(200 + seed)
            for _ in range(steps):
                idx = torch.randint(0, N, (batch,), generator=gen)
                xb, yb = X[idx], y[idx]
                pred = xb @ w
                loss = 0.5 * (pred - yb).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                mse = ((Xte @ w - yte) ** 2).mean()
                per[o].append(float(mse))
    rec = paired_summary(per, seeds)
    rec.update(problem="ridge", p=p, N=N, batch=batch, noise_y=noise_y, cond=cond,
               law=law, mean_k=None)
    return rec


# ------------------------------------------------------------------- problem: logistic
def run_logistic(p=200, K=10, N=8000, batch=64, flip=0.1, steps=1500, seeds=(1, 2, 3),
                 law="gaussian", lr=None):
    per = {o: [] for o in OPTS}
    lossf = torch.nn.CrossEntropyLoss()
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
            opt = build(o, [W], lr or (3e-2 if o != "sgdm" else 1e-1))
            gen = torch.Generator().manual_seed(300 + seed)
            for _ in range(steps):
                idx = torch.randint(0, N, (batch,), generator=gen)
                opt.zero_grad()
                lossf(X[idx] @ W, y[idx]).backward()
                opt.step()
            with torch.no_grad():
                per[o].append(float(lossf(Xte @ W, yte)))
    rec = paired_summary(per, seeds)
    rec.update(problem="logistic", p=p, K=K, N=N, batch=batch, flip=flip, law=law,
               mean_k=None)
    return rec


# ------------------------------------------------------------------- problem: small MLP
def run_mlp(d=50, h=40, N=4000, batch=64, noise_y=0.3, steps=1200, seeds=(1, 2, 3),
            lr=None):
    per = {o: [] for o in OPTS}
    for seed in seeds:
        torch.manual_seed(seed)
        # teacher
        t1 = torch.nn.Linear(d, h)
        t2 = torch.nn.Linear(h, 1)
        with torch.no_grad():
            X = torch.randn(N, d)
            y = t2(torch.tanh(t1(X))).squeeze(-1) + noise_y * torch.randn(N)
            Xte = torch.randn(2000, d)
            yte = t2(torch.tanh(t1(Xte))).squeeze(-1)
        for o in OPTS:
            torch.manual_seed(seed + 1)
            net = torch.nn.Sequential(
                torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, 1))
            opt = build(o, net.parameters(), lr or (3e-3 if o != "sgdm" else 1e-2))
            gen = torch.Generator().manual_seed(400 + seed)
            for _ in range(steps):
                idx = torch.randint(0, N, (batch,), generator=gen)
                opt.zero_grad()
                pred = net(X[idx]).squeeze(-1)
                loss = 0.5 * (pred - y[idx]).pow(2).mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                mse = ((net(Xte).squeeze(-1) - yte) ** 2).mean()
                per[o].append(float(mse))
    rec = paired_summary(per, seeds)
    rec.update(problem="mlp", d=d, h=h, N=N, batch=batch, noise_y=noise_y, mean_k=None)
    return rec


def log(msg: str = "") -> None:
    print(msg, flush=True)


def run_suite(smoke: bool = False, profile: str = "full") -> list:
    rows = []
    if smoke or profile == "smoke":
        ps = [50, 200]
        taus = [0.2, 2.0]
        seeds_q = (1, 2)
        seeds_r = (1, 2)
        steps_q, steps_r, steps_l, steps_m = 400, 300, 300, 250
        batches_r = [32]
        flips = [0.0, 0.3]
        log("SMOKE suite (small)")
    elif profile in ("cpu", "medium", "fast"):
        # CPU-sized but decision-grade: enough points to see the win region
        ps = [50, 150, 300]
        taus = [0.1, 0.5, 1.0, 2.0, 4.0]
        seeds_q = (1, 2, 3)
        seeds_r = (1, 2, 3)
        steps_q, steps_r, steps_l, steps_m = 800, 600, 600, 400
        batches_r = [32, 256]
        flips = [0.0, 0.1, 0.3]
        log("CPU/MEDIUM suite")
    else:
        ps = [50, 200, 400]
        taus = [0.1, 0.5, 1.0, 2.0, 4.0]
        seeds_q = (1, 2, 3, 4, 5)
        seeds_r = (1, 2, 3)
        steps_q, steps_r, steps_l, steps_m = 2500, 1500, 1500, 1200
        batches_r = [32, 256]
        flips = [0.0, 0.1, 0.2, 0.4]
        log("FULL suite")

    log("\n=== quadratic: p × tau (gaussian) ===")
    for p in ps:
        for tau in taus:
            r = run_quadratic(p=p, tau=tau, law="gaussian",
                              steps=steps_q, seeds=seeds_q)
            rows.append(r)
            log(f"  p={p:4d} tau={tau:4.2f} gaussian  "
                f"AdamW={r['adamw']:10.4f}  SR-Adam={r['sradam']:10.4f}  "
                f"Δ={r['paired_diff']:+10.4f} (t={r['t_stat']:+6.2f})  win={r['win_sradam']}")

    log("\n=== quadratic: heavy tails (student) at p=150 ===")
    p_ht = ps[len(ps) // 2]
    for tau in taus:
        r = run_quadratic(p=p_ht, tau=tau, law="student",
                          steps=steps_q, seeds=seeds_q)
        rows.append(r)
        log(f"  p={p_ht} tau={tau:4.2f} student   "
            f"AdamW={r['adamw']:10.4f}  SR-Adam={r['sradam']:10.4f}  "
            f"Δ={r['paired_diff']:+10.4f} win={r['win_sradam']}")

    log("\n=== ridge regression ===")
    for p in ps[:2]:
        for batch in batches_r:
            r = run_ridge(p=p, N=2000, batch=batch, noise_y=1.0,
                          cond=30.0, steps=steps_r, seeds=seeds_r)
            rows.append(r)
            log(f"  ridge p={p} B={batch}  "
                f"AdamW={r['adamw']:.4f}  SR-Adam={r['sradam']:.4f}  "
                f"Δ={r['paired_diff']:+.4f} win={r['win_sradam']}")

    log("\n=== logistic with label flips ===")
    for flip in flips:
        r = run_logistic(p=150, N=4000, batch=64,
                         flip=flip, steps=steps_l, seeds=seeds_r)
        rows.append(r)
        log(f"  logistic flip={flip}  "
            f"AdamW={r['adamw']:.4f}  SR-Adam={r['sradam']:.4f}  "
            f"Δ={r['paired_diff']:+.4f} win={r['win_sradam']}")

    log("\n=== small MLP regressor ===")
    r = run_mlp(d=60, h=48, N=2500, batch=64, noise_y=0.5,
                steps=steps_m, seeds=seeds_r)
    rows.append(r)
    log(f"  mlp  AdamW={r['adamw']:.4f}  SR-Adam={r['sradam']:.4f}  "
        f"Δ={r['paired_diff']:+.4f} win={r['win_sradam']}")

    return rows


def make_figures(rows):
    q = [r for r in rows if r["problem"] == "quadratic" and r["law"] == "gaussian"]
    # heatmap-like slice: Δ (AdamW - SR-Adam) vs tau for each p
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ps = sorted({r["p"] for r in q})
    taus = sorted({r["tau"] for r in q})
    for p in ps:
        series = sorted([r for r in q if r["p"] == p], key=lambda r: r["tau"])
        ax[0].plot([r["tau"] for r in series], [r["paired_diff"] for r in series],
                   "o-", ms=3, label=f"p={p}")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel(r"noise scale $\tau$")
    ax[0].set_ylabel(r"paired diff  AdamW $-$ SR-Adam (objective)")
    ax[0].set_xscale("log")
    ax[0].set_yscale("symlog", linthresh=1e-2)
    ax[0].legend(loc="upper left", fontsize=7)
    ax[0].set_title("Quadratic (lower objective better)")

    labs, wins, fails = [], [], []
    for r in rows:
        name = r["problem"]
        if name == "quadratic":
            tag = f"quad p={r['p']} τ={r['tau']} {r['law'][:3]}"
        elif name == "ridge":
            tag = f"ridge p={r['p']} B={r['batch']}"
        elif name == "logistic":
            tag = f"logist flip={r['flip']}"
        else:
            tag = "mlp"
        labs.append(tag)
        wins.append(1 if r["win_sradam"] else 0)
        fails.append(r["paired_diff"])
    y = np.arange(len(labs))
    colors = ["#2a9d8f" if w else "#e76f51" for w in wins]
    ax[1].barh(y, fails, color=colors)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(labs, fontsize=6)
    ax[1].set_xlabel(r"AdamW $-$ SR-Adam (positive $=$ SR better)")
    ax[1].set_title("Win / lose map (Path 2)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        dest = os.path.join(FIGS, f"phase_diagram_slices.{ext}")
        try:
            fig.savefig(dest)
        except PermissionError:
            dest = os.path.join(FIGS, f"phase_diagram_slices_{int(time.time())}.{ext}")
            fig.savefig(dest)
            log(f"note: default figure path locked, wrote {dest}")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    # tau × p grid for quadratic
    grid = {}
    for r in q:
        grid[(r["p"], r["tau"])] = r["paired_diff"]
    if grid:
        ps = sorted({p for p, _ in grid})
        taus = sorted({t for _, t in grid})
        M = np.zeros((len(ps), len(taus)))
        for i, p in enumerate(ps):
            for j, t in enumerate(taus):
                M[i, j] = grid.get((p, t), np.nan)
        im = ax[0].imshow(M, aspect="auto", cmap="RdYlGn")
        ax[0].set_xticks(range(len(taus)))
        ax[0].set_xticklabels([f"{t:g}" for t in taus], fontsize=7)
        ax[0].set_yticks(range(len(ps)))
        ax[0].set_yticklabels([str(p) for p in ps], fontsize=7)
        ax[0].set_xlabel(r"$\tau$")
        ax[0].set_ylabel("p")
        ax[0].set_title("AdamW − SR-Adam (quadratic, gaussian)")
        fig.colorbar(im, ax=ax[0], fraction=0.046)

    # mean_k if present
    ks = [(r.get("tau"), r.get("mean_k"), r.get("p")) for r in q if r.get("mean_k") is not None]
    if ks:
        for p in ps:
            sub = [(t, k) for t, k, pp in ks if pp == p and t is not None]
            if sub:
                sub = sorted(sub)
                ax[1].plot([t for t, _ in sub], [k for _, k in sub], "o-", ms=3, label=f"p={p}")
        ax[1].set_xlabel(r"$\tau$")
        ax[1].set_ylabel("mean Stein weight $k_t$")
        ax[1].set_xscale("log")
        ax[1].legend(fontsize=7)
        ax[1].set_title("Realised shrinkage weight")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        dest = os.path.join(FIGS, f"phase_diagram_heatmap.{ext}")
        try:
            fig.savefig(dest)
        except PermissionError:
            dest = os.path.join(FIGS, f"phase_diagram_heatmap_{int(time.time())}.{ext}")
            fig.savefig(dest)
            log(f"note: default figure path locked, wrote {dest}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny grid for a quick check")
    ap.add_argument("--profile", default="full",
                    choices=["smoke", "cpu", "medium", "fast", "full"],
                    help="cpu/medium/fast: decision grid for one machine without a GPU")
    args = ap.parse_args()
    if args.smoke:
        args.profile = "smoke"
    t0 = time.time()
    rows = run_suite(smoke=args.smoke, profile=args.profile)
    path = os.path.join(OUT, "phase_diagram.json")
    with open(path, "w") as f:
        json.dump(dict(meta=dict(smoke=args.smoke, profile=args.profile,
                                 created=time.strftime("%Y-%m-%d %H:%M:%S"),
                                 seconds=time.time() - t0),
                       rows=rows), f, indent=2)
    make_figures(rows)
    wins = sum(1 for r in rows if r["win_sradam"])
    log(f"\n{wins}/{len(rows)} settings: SR-Adam better than AdamW (paired mean)")
    log(f"wrote {path}")
    log(f"figures: {os.path.join(FIGS, 'phase_diagram_slices.pdf')}, "
        f"{os.path.join(FIGS, 'phase_diagram_heatmap.pdf')}")
    log(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
