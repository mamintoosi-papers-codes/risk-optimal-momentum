"""
Path 3 — real regression benchmarks (UCI / OpenML / sklearn built-ins).

Goal: find on which real tabular / low-dimensional regression problems SR-Adam
beats well-known optimizers (AdamW, AMSGrad, SGD-M), and which problems it does
not.  Method is treated as a tool with a regime of strength, not a universal
dominance claim.

Problems (all real named datasets, not synthetic DGPs):
  diabetes          sklearn built-in (n=442, d=10) — classic small regression
  california_housing sklearn (n=20640, d=8)
  wine_quality      UCI/OpenML wine-quality-red (regression on quality score)
  energy_efficiency UCI (OpenML energy-efficiency) if available
  concrete          UCI concrete compressive strength if available

Models:
  linear  — one weight vector (d + bias), squared loss (the estimation-heavy case)
  mlp     — 1 hidden layer teacher-student is NOT used here; a small real MLP fit

Protocol:
  standardize X (train stats), standardize y (optional), 70/30 split,
  5 seeds, same lr policy per optimizer family, final test MSE (and MAE).
  Paired comparison AdamW - SR-Adam (positive = SR better, lower loss).

Run:  python -u path3_uci.py          # default suite
      python -u path3_uci.py --smoke  # skip slow downloads / large sets
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sr_adam import SRAdam

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs"
OUT = HERE.parent / "out"
REPORT_DIR = HERE.parent / "reports" / "PATH3_uci_regression"
for p in (FIGS, OUT, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)

OPTS = ["sgdm", "adamw", "amsgrad", "sradam"]
LABEL = {"sgdm": "SGD-M", "adamw": "AdamW", "amsgrad": "AMSGrad", "sradam": "SR-Adam"}


def build(name, params, lr):
    if name == "sgdm":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if name == "amsgrad":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, amsgrad=True)
    return SRAdam(params, lr=lr, warmup=5)


def load_xy(name: str):
    """Return X, y as float32 arrays. Raises if dataset unavailable."""
    from sklearn.datasets import (
        load_diabetes,
        fetch_california_housing,
        fetch_openml,
    )

    if name == "diabetes":
        data = load_diabetes()
        return data.data.astype(np.float32), data.target.astype(np.float32)

    if name == "california_housing":
        data = fetch_california_housing()
        return data.data.astype(np.float32), data.target.astype(np.float32)

    if name == "wine_quality":
        # red wine quality, 11 features, score 3–8
        X, y = fetch_openml(name="wine-quality-red", version=1, as_frame=False,
                            return_X_y=True)
        return X.astype(np.float32), y.astype(np.float32)

    if name == "energy_efficiency":
        X, y = fetch_openml(data_id=128, as_frame=False, return_X_y=True)
        # target is heating + cooling load (2 cols) — take heating
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:
            y = y[:, 0]
        return X.astype(np.float32), y.astype(np.float32)

    if name == "concrete":
        X, y = fetch_openml(data_id=165, as_frame=False, return_X_y=True)
        return X.astype(np.float32), y.astype(np.float32)

    raise ValueError(name)


def split_scale(X, y, test_frac=0.3, seed=0):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_te = max(1, int(n * test_frac))
    te, tr = idx[:n_te], idx[n_te:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ymu, ysd = y[tr].mean(), y[tr].std() + 1e-8
    ytr, yte = (y[tr] - ymu) / ysd, (y[te] - ymu) / ysd
    return (torch.from_numpy(Xtr), torch.from_numpy(ytr),
            torch.from_numpy(Xte), torch.from_numpy(yte), ysd)


def run_linear(Xtr, ytr, Xte, yte, opt_name, steps, lr, batch, seed):
    d = Xtr.shape[1]
    torch.manual_seed(seed)
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = build(opt_name, [w, b], lr)
    gen = torch.Generator().manual_seed(100 + seed)
    n = Xtr.shape[0]
    B = min(batch, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (B,), generator=gen)
        pred = Xtr[idx] @ w + b
        loss = 0.5 * (pred - ytr[idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse = ((Xte @ w + b - yte) ** 2).mean().item()
        mae = (Xte @ w + b - yte).abs().mean().item()
    return mse, mae


def run_mlp(Xtr, ytr, Xte, yte, opt_name, steps, lr, batch, seed, h=32):
    d = Xtr.shape[1]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, 1))
    opt = build(opt_name, net.parameters(), lr)
    gen = torch.Generator().manual_seed(200 + seed)
    n = Xtr.shape[0]
    B = min(batch, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (B,), generator=gen)
        pred = net(Xtr[idx]).squeeze(-1)
        loss = 0.5 * (pred - ytr[idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse = ((net(Xte).squeeze(-1) - yte) ** 2).mean().item()
        mae = (net(Xte).squeeze(-1) - yte).abs().mean().item()
    return mse, mae


def paired(per):
    rec = {}
    for o in OPTS:
        a = np.asarray(per[o], dtype=float)
        rec[o] = float(a.mean())
        rec[o + "_sd"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        rec[o + "_all"] = [float(x) for x in a]
    d = np.asarray(per["adamw"]) - np.asarray(per["sradam"])
    rec["paired_diff"] = float(d.mean())
    if len(d) > 1 and d.std(ddof=1) > 0:
        rec["t_stat"] = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    else:
        rec["t_stat"] = 0.0
    rec["win_sradam"] = bool(d.mean() > 0)
    return rec


DATASETS = ["diabetes", "california_housing", "wine_quality",
            "energy_efficiency", "concrete"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()

    names = ["diabetes", "california_housing"] if args.smoke else DATASETS
    seeds = list(range(1, (2 if args.smoke else args.seeds) + 1))
    steps = 300 if args.smoke else args.steps
    rows = []
    t0 = time.time()

    for name in names:
        print(f"\n=== dataset: {name} ===", flush=True)
        try:
            X, y = load_xy(name)
        except Exception as e:
            print(f"  SKIP ({type(e).__name__}: {e})", flush=True)
            rows.append(dict(dataset=name, error=str(e), status="skipped"))
            continue

        n, d = X.shape
        print(f"  n={n} d={d} y_std={y.std():.3g}", flush=True)

        for model in ("linear", "mlp"):
            per = {o: [] for o in OPTS}
            per_mae = {o: [] for o in OPTS}
            for seed in seeds:
                Xtr, ytr, Xte, yte, ysd = split_scale(X, y, test_frac=0.3, seed=seed)
                batch = 32 if n > 500 else max(8, n // 4)
                lr = 3e-2 if model == "linear" else 1e-2
                for o in OPTS:
                    if model == "linear":
                        mse, mae = run_linear(Xtr, ytr, Xte, yte, o, steps, lr, batch, seed)
                    else:
                        mse, mae = run_mlp(Xtr, ytr, Xte, yte, o, steps, lr, batch, seed)
                    per[o].append(mse)
                    per_mae[o].append(mae)
            rec = paired(per)
            rec.update(dataset=name, model=model, n=int(n), d=int(d),
                       steps=steps, seeds=seeds, status="ok")
            rec["adamw_mae"] = float(np.mean(per_mae["adamw"]))
            rec["sradam_mae"] = float(np.mean(per_mae["sradam"]))
            rows.append(rec)
            print(
                f"  [{model}]  AdamW={rec['adamw']:.4f}±{rec['adamw_sd']:.4f}  "
                f"SR-Adam={rec['sradam']:.4f}±{rec['sradam_sd']:.4f}  "
                f"Δ={rec['paired_diff']:+.4f} (t={rec['t_stat']:+.2f})  "
                f"win={rec['win_sradam']}",
                flush=True,
            )

    payload = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  seconds=time.time() - t0, smoke=args.smoke,
                  seeds=seeds, steps=steps,
                  note="test MSE on standardized y (dimensionless); positive Δ = SR better"),
        rows=rows,
    )
    out_path = OUT / "path3_uci.json"
    out_path.write_text(json.dumps(payload, indent=2))
    (REPORT_DIR / "path3_uci.json").write_text(json.dumps(payload, indent=2))

    # figure
    ok = [r for r in rows if r.get("status") == "ok"]
    labels, diffs, wins = [], [], []
    for r in ok:
        labels.append(f"{r['dataset'][:12]}\n{r['model']}")
        diffs.append(r["paired_diff"])
        wins.append(r["win_sradam"])
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    colors = ["#2a9d8f" if w else "#e76f51" for w in wins]
    ax.barh(np.arange(len(labels)), diffs, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(r"AdamW $-$ SR-Adam  test MSE (positive $=$ SR better)")
    ax.set_title("Path 3: real regression benchmarks")
    fig.tight_layout()
    for dest in (FIGS / "path3_uci.png", REPORT_DIR / "path3_uci.png",
                 FIGS / "path3_uci.pdf", REPORT_DIR / "path3_uci.pdf"):
        try:
            fig.savefig(dest)
        except PermissionError:
            pass
    plt.close(fig)

    wins_n = sum(1 for r in ok if r["win_sradam"])
    print(f"\n{wins_n}/{len(ok)} (dataset, model) cells: SR-Adam better than AdamW",
          flush=True)
    print(f"wrote {out_path}", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
