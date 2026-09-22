"""
Path 5 — more real regression datasets (extend Path 3).

Goal (item 8 of the next-steps list): add year/protein and fix energy/concrete
if OpenML provides them; report the same paired AdamW vs SR-Adam comparison
on linear + small MLP.

Datasets (best-effort; skip on loader failure):
  year_prediction     OpenML year (large, subset to first 20k rows for CPU)
  protein             OpenML protein tertiary structure (subset)
  energy_efficiency   OpenML data_id 128 (numeric coercion)
  concrete            OpenML concrete_compressive_strength by name
  linnerud            sklearn built-in (tiny multivariate)
  wine_quality_red    already in Path 3 (re-run if present for consistency)

Run:  python -u path5_more_datasets.py
      python -u path5_more_datasets.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from path3_uci import OPTS, build, paired, split_scale, run_linear, run_mlp

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def _to_float_xy(X, y, max_n=8000, seed=0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2:
        y = y[:, 0]
    # drop non-numeric columns
    if X.ndim == 2 and X.shape[1] > 0:
        keep = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.issubdtype(col.dtype, np.number):
                keep.append(j)
            else:
                try:
                    col.astype(np.float64)
                    keep.append(j)
                except Exception:
                    pass
        if keep:
            X = np.column_stack([
                X[:, j].astype(np.float64) if X[:, j].dtype != object
                else np.array([float(v) for v in X[:, j]], dtype=np.float64)
                for j in range(X.shape[1])
            ])
        # drop columns that failed
        finite_cols = []
        for j in range(X.shape[1]):
            if np.isfinite(X[:, j]).mean() > 0.5:
                finite_cols.append(j)
        X = X[:, finite_cols]
    mask = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if X.shape[0] > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], max_n, replace=False)
        X, y = X[idx], y[idx]
    return X.astype(np.float32), y.astype(np.float32)


def load_named(name: str):
    from sklearn.datasets import fetch_openml, load_linnerud
    import pandas as pd

    def from_frame(df, y=None):
        """Numeric matrix from pandas frame (one-hot categoricals)."""
        if y is None:
            y = df.iloc[:, -1]
            Xdf = df.iloc[:, :-1]
        else:
            Xdf = df
        Xnum = pd.get_dummies(Xdf, drop_first=True).to_numpy(dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        return Xnum, y

    if name == "year_prediction":
        X, y = fetch_openml(data_id=218, as_frame=True, return_X_y=True)
        Xnum, yn = from_frame(X, y)
        return _to_float_xy(Xnum, yn, max_n=8000)

    if name == "protein":
        X, y = fetch_openml(data_id=42730, as_frame=False, return_X_y=True)
        return _to_float_xy(X, y, max_n=4000)

    if name == "energy_efficiency":
        X, y = fetch_openml(name="energy_efficiency", version=1, as_frame=True,
                            return_X_y=True)
        Xnum, yn = from_frame(X, y)
        return _to_float_xy(Xnum, yn, max_n=8000)

    if name == "concrete":
        X, y = fetch_openml(data_id=187, as_frame=False, return_X_y=True)
        return _to_float_xy(X, y, max_n=8000)

    if name == "linnerud":
        d = load_linnerud()
        X = d.data.astype(np.float32)
        y = d.target[:, 0].astype(np.float32)
        return X, y

    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer steps/seeds")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    names = ["linnerud", "energy_efficiency", "concrete", "year_prediction", "protein"]
    seeds = [1, 2] if args.quick else list(range(1, args.seeds + 1))
    steps = 250 if args.quick else args.steps
    rows = []
    t0 = time.time()

    for name in names:
        print(f"\n=== {name} ===", flush=True)
        try:
            X, y = load_named(name)
        except Exception as e:
            print(f"  SKIP {type(e).__name__}: {e}", flush=True)
            rows.append(dict(dataset=name, status="skipped", error=str(e)[:300]))
            continue
        print(f"  n={X.shape[0]} d={X.shape[1]} y_std={float(y.std()):.3g}", flush=True)
        for model in ("linear", "mlp"):
            per = {o: [] for o in OPTS}
            for seed in seeds:
                Xtr, ytr, Xte, yte, ysd = split_scale(X, y, test_frac=0.3, seed=seed)
                batch = 32 if X.shape[0] > 500 else max(8, X.shape[0] // 4)
                lr = 3e-2 if model == "linear" else 1e-2
                for o in OPTS:
                    if model == "linear":
                        mse, _mae = run_linear(Xtr, ytr, Xte, yte, o, steps, lr, batch, seed)
                    else:
                        mse, _mae = run_mlp(Xtr, ytr, Xte, yte, o, steps, lr, batch, seed)
                    per[o].append(mse)
            rec = paired(per)
            rec.update(dataset=name, model=model, n=int(X.shape[0]), d=int(X.shape[1]),
                       steps=steps, status="ok")
            rows.append(rec)
            print(f"  [{model}] AdamW={rec['adamw']:.4f}  SR={rec['sradam']:.4f}  "
                  f"Δ={rec['paired_diff']:+.4f}  win={rec['win_sradam']}", flush=True)

    payload = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             seconds=time.time() - t0,
                             seeds=seeds, steps=steps, quick=args.quick),
                   rows=rows)
    (OUT / "path5_more_datasets.json").write_text(json.dumps(payload, indent=2))

    ok = [r for r in rows if r.get("status") == "ok"]
    wins = sum(1 for r in ok if r["win_sradam"])
    print(f"\n{wins}/{len(ok)} (dataset, model) cells: SR-Adam better", flush=True)
    print(f"wrote {OUT / 'path5_more_datasets.json'}", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
