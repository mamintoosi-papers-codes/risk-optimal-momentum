"""
Build two paper tables from real-data experiments:

  Table A — Regression   (linear and MLP): paired gap in test MSE, AdamW - SR-Adam
  Table B — Classification (logistic, MLP, GCN): test accuracy, AdamW vs SR-Adam

Datasets
--------
Regression: diabetes, california_housing, wine_quality, linnerud, protein,
            blogfeedback (if cached), energy/concrete when OpenML works.
Classification: digits (logistic/MLP, optional label noise),
                breast_cancer, wine, ionosphere (logistic/MLP),
                Cora / CiteSeer / PubMed (GCN) when PyG data is present.

Outputs
-------
  ../out/paper_tables.json
  ../paper/tab_regression.tex
  ../paper/tab_classification.tex

Run:
  python -u paper_tables.py --quick
  python -u paper_tables.py --seeds 5 --steps 800
"""

from __future__ import annotations

import sys

# Windows redirected stdio uses the ANSI codepage (cp1252 here), which cannot
# encode printed Unicode (Δ, k̄, …).  Keep non-interactive log output working.
if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure") and not sys.stderr.isatty():
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from path3_uci import build, paired, split_scale, OPTS
from dataset_names import pretty
from tune_helper import (add_tuning_args, run_tuning, Stopwatch, EarlyStopper,
                         gate_stats)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
PAPER = HERE.parent / "paper"
DATA = HERE.parent / "data"
for p in (OUT, PAPER):
    p.mkdir(parents=True, exist_ok=True)


def split_clf(X, y, test_frac=0.3, seed=0):
    """Train/test split with X standardised; integer labels left unchanged."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_te = max(1, int(n * test_frac))
    te, tr = idx[:n_te], idx[n_te:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr = torch.from_numpy((X[tr] - mu) / sd).float()
    Xte = torch.from_numpy((X[te] - mu) / sd).float()
    ytr = torch.from_numpy(y[tr].astype(np.int64))
    yte = torch.from_numpy(y[te].astype(np.int64))
    return Xtr, ytr, Xte, yte


def run_linear(Xtr, ytr, Xte, yte, opt_name, steps, lr, batch, seed,
               val=None, patience=0, timing=False, stats_out=None, **optkw):
    d = Xtr.shape[1]
    torch.manual_seed(seed)
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = build(opt_name, [w, b], lr, **optkw)
    gen = torch.Generator().manual_seed(11 + seed)
    n = Xtr.shape[0]
    B = min(batch, n)
    sw = Stopwatch(timing)
    es = EarlyStopper(patience, mode="min") if patience and val is not None else None
    best_state, stopped_at = None, None
    for step in range(steps):
        idx = torch.randint(0, n, (B,), generator=gen)
        pred = Xtr[idx] @ w + b
        loss = 0.5 * (pred - ytr[idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sw.tick()
        if es is not None and es.due(step):
            Xv, yv = val
            with torch.no_grad():
                v = float(((Xv @ w + b - yv) ** 2).mean())
            if es.update(v):
                best_state = [w.detach().clone(), b.detach().clone()]
            if es.should_stop:
                stopped_at = step + 1
                break
    with torch.no_grad():
        if best_state is not None:
            w.data.copy_(best_state[0])
            b.data.copy_(best_state[1])
        score = float(((Xte @ w + b - yte) ** 2).mean())
    if stats_out is not None:
        stats_out["ms_per_step"] = sw.ms_per_step
        stats_out["stopped_at"] = stopped_at
        stats_out["best_val"] = es.best if es is not None else None
        stats_out.update({k: v for k, v in gate_stats(opt, opt_name).items()
                          if v is not None})
    return score


def run_mlp_reg(Xtr, ytr, Xte, yte, opt_name, steps, lr, batch, seed, h=32,
                val=None, patience=0, timing=False, stats_out=None, **optkw):
    d = Xtr.shape[1]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(d, h), torch.nn.Tanh(), torch.nn.Linear(h, 1))
    opt = build(opt_name, net.parameters(), lr, **optkw)
    gen = torch.Generator().manual_seed(22 + seed)
    n = Xtr.shape[0]
    B = min(batch, n)
    sw = Stopwatch(timing)
    es = EarlyStopper(patience, mode="min") if patience and val is not None else None
    best_state, stopped_at = None, None
    for step in range(steps):
        idx = torch.randint(0, n, (B,), generator=gen)
        pred = net(Xtr[idx]).squeeze(-1)
        loss = 0.5 * (pred - ytr[idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sw.tick()
        if es is not None and es.due(step):
            Xv, yv = val
            with torch.no_grad():
                v = float(((net(Xv).squeeze(-1) - yv) ** 2).mean())
            if es.update(v):
                best_state = [t.detach().clone() for t in net.parameters()]
            if es.should_stop:
                stopped_at = step + 1
                break
    with torch.no_grad():
        if best_state is not None:
            for t, s in zip(net.parameters(), best_state):
                t.data.copy_(s)
        score = float(((net(Xte).squeeze(-1) - yte) ** 2).mean())
    if stats_out is not None:
        stats_out["ms_per_step"] = sw.ms_per_step
        stats_out["stopped_at"] = stopped_at
        stats_out["best_val"] = es.best if es is not None else None
        stats_out.update({k: v for k, v in gate_stats(opt, opt_name).items()
                          if v is not None})
    return score


def run_logistic_clf(Xtr, ytr, Xte, yte, opt_name, steps, lr, batch, seed, K,
                     val=None, patience=0, timing=False, stats_out=None, **optkw):
    ytr = ytr.long()
    yte = yte.long()
    d = Xtr.shape[1]
    torch.manual_seed(seed)
    W = torch.zeros(d, K, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    opt = build(opt_name, [W, b], lr, **optkw)
    gen = torch.Generator().manual_seed(33 + seed)
    n = Xtr.shape[0]
    B = min(batch, n)
    lossf = torch.nn.CrossEntropyLoss()
    sw = Stopwatch(timing)
    es = EarlyStopper(patience, mode="max") if patience and val is not None else None
    best_state, stopped_at = None, None
    for step in range(steps):
        idx = torch.randint(0, n, (B,), generator=gen)
        opt.zero_grad()
        lossf(Xtr[idx] @ W + b, ytr[idx]).backward()
        opt.step()
        sw.tick()
        if es is not None and es.due(step):
            Xv, yv = val[0], val[1].long()
            with torch.no_grad():
                v = float(((Xv @ W + b).argmax(1) == yv).float().mean())
            if es.update(v):
                best_state = [W.detach().clone(), b.detach().clone()]
            if es.should_stop:
                stopped_at = step + 1
                break
    with torch.no_grad():
        if best_state is not None:
            W.data.copy_(best_state[0])
            b.data.copy_(best_state[1])
        pred = (Xte @ W + b).argmax(1)
        score = float((pred == yte).float().mean())
    if stats_out is not None:
        stats_out["ms_per_step"] = sw.ms_per_step
        stats_out["stopped_at"] = stopped_at
        stats_out["best_val"] = es.best if es is not None else None
        stats_out.update({k: v for k, v in gate_stats(opt, opt_name).items()
                          if v is not None})
    return score


def run_mlp_clf(Xtr, ytr, Xte, yte, opt_name, steps, lr, batch, seed, K, h=32,
                val=None, patience=0, timing=False, stats_out=None, **optkw):
    ytr = ytr.long()
    yte = yte.long()
    d = Xtr.shape[1]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, K))
    opt = build(opt_name, net.parameters(), lr, **optkw)
    gen = torch.Generator().manual_seed(44 + seed)
    n = Xtr.shape[0]
    B = min(batch, n)
    lossf = torch.nn.CrossEntropyLoss()
    sw = Stopwatch(timing)
    es = EarlyStopper(patience, mode="max") if patience and val is not None else None
    best_state, stopped_at = None, None
    for step in range(steps):
        idx = torch.randint(0, n, (B,), generator=gen)
        opt.zero_grad()
        lossf(net(Xtr[idx]), ytr[idx]).backward()
        opt.step()
        sw.tick()
        if es is not None and es.due(step):
            Xv, yv = val[0], val[1].long()
            with torch.no_grad():
                v = float((net(Xv).argmax(1) == yv).float().mean())
            if es.update(v):
                best_state = [t.detach().clone() for t in net.parameters()]
            if es.should_stop:
                stopped_at = step + 1
                break
    with torch.no_grad():
        if best_state is not None:
            for t, s in zip(net.parameters(), best_state):
                t.data.copy_(s)
        pred = net(Xte).argmax(1)
        score = float((pred == yte).float().mean())
    if stats_out is not None:
        stats_out["ms_per_step"] = sw.ms_per_step
        stats_out["stopped_at"] = stopped_at
        stats_out["best_val"] = es.best if es is not None else None
        stats_out.update({k: v for k, v in gate_stats(opt, opt_name).items()
                          if v is not None})
    return score


def load_reg(name):
    from path3_uci import load_xy
    if name == "blogfeedback":
        from path6_year_blog_clf import load_blog
        return load_blog(max_rows=8000)
    if name == "linnerud":
        from sklearn.datasets import load_linnerud as _ll
        d = _ll()
        return d.data.astype(np.float32), d.target[:, 0].astype(np.float32)
    return load_xy(name)


def load_clf(name):
    from sklearn.datasets import (
        load_digits, load_breast_cancer, load_wine, load_iris,
        fetch_openml,
    )
    if name == "digits":
        d = load_digits()
        return d.data.astype(np.float32), d.target.astype(np.int64)
    if name == "breast_cancer":
        d = load_breast_cancer()
        return d.data.astype(np.float32), d.target.astype(np.int64)
    if name == "wine":
        d = load_wine()
        return d.data.astype(np.float32), d.target.astype(np.int64)
    if name == "iris":
        d = load_iris()
        return d.data.astype(np.float32), d.target.astype(np.int64)
    if name == "ionosphere":
        X, y = fetch_openml(data_id=59, as_frame=False, return_X_y=True)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        # map labels to 0..K-1
        cats = {c: i for i, c in enumerate(sorted(set(y.tolist())))}
        y = np.array([cats[v] for v in y], dtype=np.int64)
        m = np.isfinite(X).all(1)
        return X[m], y[m]
    raise ValueError(name)


def gnn_rows(seeds, label_noise=(0.0, 0.3)):
    """Reuse gnn_pilot protocol if PyG + data exist; else skip."""
    rows = []
    try:
        from torch_geometric.datasets import Planetoid
        from gnn_pilot import GCN, build_opt
    except Exception as e:
        return [dict(status="skipped", error=f"pyg: {e}")]

    for name in ("Cora", "CiteSeer", "PubMed"):
        root = DATA / name
        if not root.exists() and name != "Cora":
            root = DATA / "Cora" if name == "Cora" else root
        try:
            dataset = Planetoid(root=str(DATA / name), name=name)
            data = dataset[0]
        except Exception as e:
            rows.append(dict(dataset=name, status="skipped", error=str(e)[:200]))
            continue
        for noise in label_noise:
            per = {"adamw": [], "sradam": []}
            for seed in seeds:
                torch.manual_seed(seed)
                y = data.y.clone()
                if noise > 0:
                    m = torch.rand(y.shape[0]) < noise
                    y[m] = torch.randint(0, int(data.y.max()) + 1, (int(m.sum()),))
                for o, key in (("adamw", "adamw"), ("sradam", "sradam")):
                    torch.manual_seed(seed)
                    model = GCN(data.num_features, 16, int(data.y.max().item() + 1))
                    opt = build_opt(o, model.parameters(), 0.01)
                    for _ in range(150):
                        model.train()
                        opt.zero_grad()
                        F.cross_entropy(model(data)[data.train_mask],
                                        y[data.train_mask]).backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        pred = model(data).argmax(1)
                        acc = (pred[data.test_mask] == data.y[data.test_mask]) \
                            .float().mean().item()
                    per[key].append(acc)
            a = np.asarray(per["adamw"])
            s = np.asarray(per["sradam"])
            rows.append(dict(
                dataset=name, model="GCN", n=int(data.x.shape[0]),
                p=int(data.x.shape[1]), K=int(data.y.max().item() + 1),
                label_noise=noise, status="ok", metric="acc_frac",
                adamw=float(a.mean()), adamw_sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                sradam=float(s.mean()), sradam_sd=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                paired_diff=float(s.mean() - a.mean()),
                win_sradam=bool(s.mean() > a.mean()),
            ))
    return rows


def write_tex(reg_rows, clf_rows):
    # Table A: regression
    lines = [
        r"% Auto-generated by paper_tables.py — do not edit by hand",
        r"\begin{tabular}{llrrc}",
        r"\toprule",
        r"Dataset & Model & $n$ & $p$ & Gap (MSE) \\",
        r"\midrule",
        r"% Model: linear = linear regressor; MLP = one hidden layer, nonlinear",
    ]
    for r in reg_rows:
        if r.get("status") != "ok":
            continue
        model = "linear" if r.get("model") == "linear" else "MLP"
        gap = r["paired_diff"]
        gap_s = f"${gap:+.3f}$"
        if gap > 0:
            gap_s = r"$\bm{" + f"{gap:+.3f}" + r"}$"
        lines.append(
            f"{pretty(r['dataset'])} & {model} & {r['n']} & {r['p']} & "
        f"{gap_s} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tab_regression.tex").write_text("\n".join(lines) + "\n")

    # Table B: classification
    lines = [
        r"% Auto-generated by paper_tables.py — do not edit by hand",
        r"\begin{tabular}{llrrccc}",
        r"\toprule",
        r"Dataset & Model & $n$ & $p$ & $K$ & AdamW & SR-Adam \\",
        r"\midrule",
    ]
    for r in clf_rows:
        if r.get("status") != "ok":
            continue
        if r.get("label_noise"):
            continue  # noisy rows belong in the dedicated noise table
        nm = pretty(r["dataset"])
        if r.get("label_noise"):
            nm += " (noisy)"
        mkey = str(r.get("model", "")).lower()
        model = "MLP" if mkey == "mlp" else (
            "GCN" if mkey == "gcn" else (
                "logistic" if mkey == "logistic" else r.get("model", "")))
        aw = 100.0 * float(r["adamw"])
        sr = 100.0 * float(r["sradam"])
        aw_s = f"{aw:.1f}"
        sr_s = f"{sr:.1f}"
        bold = r"\textbf{" + sr_s + "}" if r.get("win_sradam") else sr_s
        K = r.get("K", r.get("classes", "---"))
        lines.append(
            f"{nm} & {model} & {r.get('n', '---')} & {r.get('p', '---')} & {K} & "
            f"{aw_s} & {bold} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tab_classification.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=800)
    add_tuning_args(ap)
    args = ap.parse_args()
    seeds = [1, 2] if args.quick else list(range(1, args.seeds + 1))
    steps = 250 if args.quick else args.steps
    t0 = time.time()
    reg_rows, clf_rows = [], []

    # ---- Phase C helpers -------------------------------------------------
    def _carve(Xtr, ytr, frac=0.2, seed=0):
        """Carve a validation split off the training data (early stopping)."""
        n = int(Xtr.shape[0])
        m = max(1, int(n * frac))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        idx_va, idx_tr = perm[:m], perm[m:]
        return Xtr[idx_tr], ytr[idx_tr], Xtr[idx_va], ytr[idx_va]

    def _ow(cfg, o, a):
        kw = {}
        if cfg.get("wd"):
            kw["wd"] = cfg["wd"]
        if cfg.get("beta1") is not None:
            kw["beta1"] = cfg["beta1"]
        if getattr(a, "gate_log", False) and o == "sradam":
            kw["track_stats"] = True
        return kw

    def _sel(cfgs, o, default):
        c = cfgs.get(o) or {}
        return c.get("lr") or default

    def _rec(mss, gfs, st):
        if st.get("ms_per_step"):
            mss.append(st["ms_per_step"])
        if st.get("gate_open_frac") is not None:
            gfs.append(st["gate_open_frac"])

    def _attach(rec, mss, gfs):
        if mss:
            rec["ms_per_step"] = float(np.mean(mss))
        if gfs:
            rec["gate_open_frac"] = float(np.mean(gfs))

    # ---- Phase C: fair-baseline tuning (item a7) ------------------------
    cfg_reg, cfg_clf = {}, {}
    if args.tune:
        print("=== TUNING (a7): identical grid across optimisers ===", flush=True)
        REG_GRID = {"sgdm": [0.03, 0.1, 0.3],
                    "adamw": [3e-3, 1e-2, 3e-2, 1e-1],
                    "amsgrad": [3e-3, 1e-2, 3e-2, 1e-1],
                    "sradam": [3e-3, 1e-2, 3e-2, 1e-1]}
        CLF_GRID = {"sgdm": [0.01, 0.03, 0.1],
                    "adamw": [1e-2, 3e-2, 5e-2, 1e-1],
                    "amsgrad": [1e-2, 3e-2, 5e-2, 1e-1],
                    "sradam": [1e-2, 3e-2, 5e-2, 1e-1]}
        tune_steps = 300 if args.quick else min(steps, 400)
        X0 = y0 = None
        for nm0 in ("diabetes", "wine_quality", "linnerud"):
            try:
                X0, y0 = load_reg(nm0)
                break
            except Exception:
                continue
        if X0 is not None:
            def pilot_reg(cfg, seed, X0=X0, y0=y0):
                Xtr, ytr, Xte, yte, _ = split_scale(X0, y0, 0.3, seed)
                return run_linear(Xtr, ytr, Xte, yte, cfg["opt"], tune_steps,
                                  cfg["lr"] or 3e-2, 64, seed,
                                  **_ow(cfg, cfg["opt"], args))
            _, cfg_reg = run_tuning(args, OPTS, pilot_reg, "paper_tables_reg",
                                    default_grids=REG_GRID)
        Xc = yc = None
        try:
            Xc, yc = load_clf("digits")
        except Exception:
            pass
        if Xc is not None:
            def pilot_clf(cfg, seed, Xc=Xc, yc=yc):
                Xtr, ytr, Xte, yte = split_clf(Xc, yc, 0.3, seed)
                return -run_logistic_clf(Xtr, ytr, Xte, yte, cfg["opt"],
                                          tune_steps, cfg["lr"] or 5e-2,
                                          64, seed, 10,
                                          **_ow(cfg, cfg["opt"], args))
            _, cfg_clf = run_tuning(args, OPTS, pilot_clf, "paper_tables_clf",
                                    lower_is_better=False,
                                    default_grids=CLF_GRID)

    reg_sets = ["diabetes", "california_housing", "wine_quality", "linnerud",
                "protein", "blogfeedback"]
    print("=== REGRESSION ===", flush=True)
    for name in reg_sets:
        print(f"-- {name}", flush=True)
        try:
            X, y = load_reg(name)
        except Exception as e:
            reg_rows.append(dict(dataset=name, status="skipped", error=str(e)[:200]))
            print("  skip", e, flush=True)
            continue
        for model in ("linear", "mlp"):
            per = {o: [] for o in OPTS}
            mss, gfs = [], []
            for seed in seeds:
                Xtr, ytr, Xte, yte, _ = split_scale(X, y, 0.3, seed)
                batch = 64 if X.shape[0] > 1000 else max(8, X.shape[0] // 4)
                lr = 3e-2 if model == "linear" else 1e-2
                val = None
                if args.early_stop > 0 and Xtr.shape[0] > 20:
                    Xtr, ytr, Xv, yv = _carve(Xtr, ytr, 0.2, seed)
                    val = (Xv, yv)
                for o in OPTS:
                    fn = run_linear if model == "linear" else run_mlp_reg
                    st = {}; need_st = (args.timing or args.early_stop or
                                         args.gate_log)
                    per[o].append(fn(Xtr, ytr, Xte, yte, o, steps,
                                     _sel(cfg_reg, o, lr), batch, seed,
                                     val=val, patience=args.early_stop,
                                     timing=args.timing,
                                     stats_out=(st if need_st else None),
                                     **_ow(cfg_reg.get(o) or {}, o, args)))
                    if need_st:
                        _rec(mss, gfs, st)
            rec = paired(per)
            rec.update(dataset=name, model=model, n=int(X.shape[0]), p=int(X.shape[1]),
                       status="ok", metric="mse")
            if args.tune:
                rec["tuned"] = cfg_reg
            _attach(rec, mss, gfs)
            reg_rows.append(rec)
            print(f"  {model} Δ={rec['paired_diff']:+.4f} win={rec['win_sradam']}",
                  flush=True)

    print("\n=== CLASSIFICATION (tabular + digits) ===", flush=True)
    clf_specs = [
        ("digits", "logistic", 10, 0.0),
        ("digits", "logistic", 10, 0.3),
        ("digits", "mlp", 10, 0.3),
        ("breast_cancer", "logistic", 2, 0.0),
        ("breast_cancer", "mlp", 2, 0.0),
        ("ionosphere", "logistic", 2, 0.0),
        ("iris", "logistic", 3, 0.0),
    ]
    cache = {}
    for name, model, K, noise in clf_specs:
        print(f"-- {name} {model} noise={noise}", flush=True)
        if name not in cache:
            try:
                cache[name] = load_clf(name)
            except Exception as e:
                clf_rows.append(dict(dataset=name, model=model, status="skipped",
                                     error=str(e)[:200]))
                print("  skip", e, flush=True)
                continue
        X, y = cache[name]
        per = {o: [] for o in OPTS}
        mss, gfs = [], []
        for seed in seeds:
            Xtr, ytr, Xte, yte = split_clf(X, y, 0.3, seed)
            val = None
            if args.early_stop > 0 and Xtr.shape[0] > 20:
                Xtr, ytr, Xv, yv = _carve(Xtr, ytr, 0.2, seed)
                val = (Xv, yv)
            if noise > 0:
                gen = torch.Generator().manual_seed(900 + seed)
                m = torch.rand(ytr.shape[0]) < noise
                ytr = ytr.clone()
                ytr[m] = torch.randint(0, K, (int(m.sum()),), generator=gen)
            batch = 64 if X.shape[0] > 500 else max(8, X.shape[0] // 4)
            for o in OPTS:
                fn = run_logistic_clf if model == "logistic" else run_mlp_clf
                lr0 = 5e-2 if model == "logistic" else 1e-2
                st = {}; need_st = (args.timing or args.early_stop or args.gate_log)
                acc = fn(Xtr, ytr, Xte, yte, o, steps, _sel(cfg_clf, o, lr0),
                         batch, seed, K, val=val, patience=args.early_stop,
                         timing=args.timing, stats_out=(st if need_st else None),
                         **_ow(cfg_clf.get(o) or {}, o, args))
                per[o].append(acc)
                if need_st:
                    _rec(mss, gfs, st)
        a = np.asarray(per["adamw"])
        s = np.asarray(per["sradam"])
        clf_rows.append(dict(
            dataset=name, model=model, K=K, label_noise=noise,
            n=int(X.shape[0]), p=int(X.shape[1]), status="ok", metric="acc_frac",
            adamw=float(a.mean()), adamw_sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            sradam=float(s.mean()), sradam_sd=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            paired_diff=float(s.mean() - a.mean()),
            win_sradam=bool(s.mean() > a.mean()),
            adamw_all=[float(x) for x in per["adamw"]],
            sradam_all=[float(x) for x in per["sradam"]],
            paired_all=[float(x) for x in (s - a)],
            tuned=(cfg_clf if args.tune else None),
            **({} if not gfs else {"gate_open_frac": float(np.mean(gfs))}),
            **({} if not mss else {"ms_per_step": float(np.mean(mss))}),
        ))
        print(f"  AdamW={100*a.mean():.1f}  SR={100*s.mean():.1f}  "
              f"Δ={100*(s.mean()-a.mean()):+.1f}pp", flush=True)

    print("\n=== GNN ===", flush=True)
    clf_rows.extend(gnn_rows(seeds, label_noise=(0.0, 0.3)))

    write_tex(reg_rows, clf_rows)
    payload = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             seconds=time.time() - t0, seeds=seeds, steps=steps),
                   regression=reg_rows, classification=clf_rows)
    (OUT / "paper_tables.json").write_text(json.dumps(payload, indent=2))
    print("wrote", OUT / "paper_tables.json", flush=True)
    print("wrote", PAPER / "tab_regression.tex", flush=True)
    print("wrote", PAPER / "tab_classification.tex", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
