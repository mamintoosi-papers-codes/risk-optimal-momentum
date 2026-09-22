"""
Path 3b — real regression with *label noise* (theory-favourable stress).

Same datasets as path3_uci (where they load), but y is corrupted with
controllable label noise.  If the decision-theoretic story is right, SR-Adam's
edge should grow when labels (hence gradients) are noisier.

Run:  python -u path3b_label_noise.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from path3_uci import OPTS, build, load_xy, split_scale, paired

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
REPORT_DIR = HERE.parent / "reports" / "REPORT_for_Arashi"
for p in (OUT, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)


def run_linear(Xtr, ytr, Xte, yte, opt_name, steps, lr, batch, seed):
    d = Xtr.shape[1]
    torch.manual_seed(seed)
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = build(opt_name, [w, b], lr)
    gen = torch.Generator().manual_seed(300 + seed)
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
        return float(((Xte @ w + b - yte) ** 2).mean())


def main():
    # clean target is yte without noise; train y gets extra N(0, noise^2 * Var(y))
    datasets = ["diabetes", "wine_quality", "california_housing"]
    noises = [0.0, 0.5, 1.0]
    seeds = [1, 2, 3]
    steps = 600
    rows = []
    t0 = time.time()

    for name in datasets:
        try:
            X, y = load_xy(name)
        except Exception as e:
            print(f"skip {name}: {e}", flush=True)
            continue
        for noise in noises:
            per = {o: [] for o in OPTS}
            for seed in seeds:
                Xtr, ytr, Xte, yte, ysd = split_scale(X, y, test_frac=0.3, seed=seed)
                if noise > 0:
                    gen = torch.Generator().manual_seed(900 + seed)
                    ytr = ytr + noise * torch.randn(ytr.shape, generator=gen)
                batch = 32 if X.shape[0] > 500 else max(8, X.shape[0] // 4)
                for o in OPTS:
                    per[o].append(run_linear(Xtr, ytr, Xte, yte, o, steps, 3e-2, batch, seed))
            rec = paired(per)
            rec.update(dataset=name, noise=noise, model="linear", n=int(X.shape[0]),
                       d=int(X.shape[1]))
            rows.append(rec)
            print(f"{name:20s} noise={noise:.1f}  AdamW={rec['adamw']:.4f}  "
                  f"SR={rec['sradam']:.4f}  Δ={rec['paired_diff']:+.4f}  "
                  f"win={rec['win_sradam']}", flush=True)

    payload = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             seconds=time.time() - t0),
                   rows=rows)
    (OUT / "path3b_label_noise.json").write_text(json.dumps(payload, indent=2))
    (REPORT_DIR / "path3b_label_noise.json").write_text(json.dumps(payload, indent=2))
    wins = sum(1 for r in rows if r["win_sradam"])
    print(f"\n{wins}/{len(rows)} cells: SR-Adam better", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
