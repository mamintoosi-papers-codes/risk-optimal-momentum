"""
Large public regression + one noisy logistic classification (with qualitative plots).

Regression:
  year        UCI YearPredictionMSD (subset rows if needed)
  blogfeedback UCI BlogFeedback (first target, subset)
  protein     OpenML id 42730 (as in earlier real-data suite)

Classification (qualitative figure):
  sklearn digits + 30% train-label noise, multinomial logistic
  → test acc + matplotlib panel of sample digits with AdamW vs SR-Adam predictions

Run:  python -u path6_year_blog_clf.py
      python -u path6_year_blog_clf.py --quick
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from path3_uci import OPTS, build, paired, split_scale

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "large"
OUT = HERE.parent / "out"
FIGS = HERE.parent / "figs"
for p in (DATA, OUT, FIGS):
    p.mkdir(parents=True, exist_ok=True)

YEAR_URLS = (
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/year_prediction",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00207/YearPredictionMSD.txt.zip",
)
BLOG_URL = "https://archive.ics.uci.edu/static/public/304/blogfeedback.zip"


def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  download {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        # some research mirrors have incomplete CA chains
        import ssl
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def load_year(max_rows=20000, seed=0):
    """LIBSVM year_prediction (label first) or UCI YearPredictionMSD (label first)."""
    last_err = None
    for url in YEAR_URLS:
        dest = DATA / ("year_" + url.rsplit("/", 1)[-1])
        try:
            _fetch(url, dest)
            # optional bz2
            if dest.name.endswith(".bz2") or dest.read_bytes()[:3] == b"BZh":
                import bz2
                text = bz2.decompress(dest.read_bytes()).decode("utf-8", errors="ignore")
            elif dest.suffix == ".zip":
                with zipfile.ZipFile(dest) as zf:
                    text = zf.read(zf.namelist()[0]).decode("utf-8", errors="ignore")
            else:
                text = dest.read_text(errors="ignore")
        except Exception as e:
            last_err = e
            continue
        text = dest.read_text(errors="ignore") if dest.suffix != ".zip" else None
        if text is None:
            with zipfile.ZipFile(dest) as zf:
                text = zf.read(zf.namelist()[0]).decode("utf-8", errors="ignore")
        rows = []
        dim = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("0:") or ":" in line.split()[0] or ":" in line:
                # libsvm sparse: label idx:val ...
                parts = line.split()
                try:
                    yv = float(parts[0])
                except ValueError:
                    continue
                feats = {}
                for tok in parts[1:]:
                    j, v = tok.split(":")
                    feats[int(j)] = float(v)
                if not feats:
                    continue
                dmax = max(feats)
                dim = max(dim or 0, dmax)
                rows.append((yv, feats))
            else:
                parts = line.replace(",", " ").split()
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue
                if len(vals) < 3:
                    continue
                yv, fv = vals[0], vals[1:]
                rows.append((yv, {i: v for i, v in enumerate(fv)}))
        if not rows:
            raise RuntimeError(f"empty year file from {url}")
        if dim is None:
            dim = max(max(f) for _, f in rows if f)
        X = np.zeros((len(rows), dim), dtype=np.float32)
        y = np.zeros(len(rows), dtype=np.float32)
        for i, (yv, feats) in enumerate(rows):
            y[i] = yv
            for j, v in feats.items():
                if 0 <= j < dim:
                    X[i, j] = v
        if X.shape[0] > max_rows:
            rng = np.random.RandomState(seed)
            idx = rng.choice(X.shape[0], max_rows, replace=False)
            X, y = X[idx], y[idx]
        print(f"  year loaded from {dest.name}: n={X.shape[0]} d={X.shape[1]}",
              flush=True)
        return X, y
    raise RuntimeError(f"year download failed: {last_err}")


def load_blog(max_rows=15000, seed=0):
    """Concatenate several daily CSVs from the UCI BlogFeedback zip."""
    z = _fetch(BLOG_URL, DATA / "blogfeedback.zip")
    rows = []
    with zipfile.ZipFile(z) as zf:
        members = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        for member in members:
            raw = zf.read(member).decode("utf-8", errors="ignore")
            for line in raw.splitlines():
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue
                rows.append(vals)
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break
    arr = np.asarray(rows, dtype=np.float32)
    if arr.size == 0:
        raise RuntimeError("empty blogfeedback")
    return arr[:, :-1], arr[:, -1]


def load_protein(max_rows=8000, seed=0):
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml(data_id=42730, as_frame=False, return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    m = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[m], y[m]
    if X.shape[0] > max_rows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], max_rows, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def run_linear(Xtr, ytr, Xte, yte, opt_name, steps, lr, batch, seed):
    d = Xtr.shape[1]
    torch.manual_seed(seed)
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = build(opt_name, [w, b], lr)
    gen = torch.Generator().manual_seed(400 + seed)
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
        mse = float(((Xte @ w + b - yte) ** 2).mean())
    return mse


def run_logistic_digits(opt_name, seed, noise=0.3, steps=400, lr=0.05, batch=64):
    """sklearn digits, noisy train labels; returns test acc and (Xte, yte, pred)."""
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)
    torch.manual_seed(seed)
    n = X.shape[0]
    idx = torch.randperm(n)
    n_te = 360
    te, tr = idx[:n_te], idx[n_te:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr = torch.from_numpy((X[tr] - mu) / sd)
    Xte = torch.from_numpy((X[te] - mu) / sd)
    ytr = torch.from_numpy(y[tr])
    yte = y[te]
    if noise > 0:
        mask = torch.rand(ytr.shape[0]) < noise
        ytr = ytr.clone()
        ytr[mask] = torch.randint(0, 10, (int(mask.sum()),), dtype=ytr.dtype)
    d, K = Xtr.shape[1], 10
    W = torch.zeros(d, K, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    opt = build(opt_name, [W, b], lr)
    gen = torch.Generator().manual_seed(700 + seed)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        bi = torch.randint(0, Xtr.shape[0], (batch,), generator=gen)
        lossf(Xtr[bi] @ W + b, ytr[bi])
        opt.zero_grad()
        lossf(Xtr[bi] @ W + b, ytr[bi]).backward()
        opt.step()
    with torch.no_grad():
        logits = Xte @ W + b
        pred = logits.argmax(1).numpy()
    acc = float((pred == yte).mean())
    return acc, Xte.numpy(), yte, pred, (mu, sd)


def digits_qualitative_figure():
    """One noisy run; panel of samples where AdamW and SR-Adam disagree."""
    acc_a, Xa, yte, pa, _ = run_logistic_digits("adamw", seed=1, noise=0.3)
    acc_s, _, _, ps, _ = run_logistic_digits("sradam", seed=1, noise=0.3)
    # recover pixel-like 8x8 from sklearn (use original pixels for display)
    from sklearn.datasets import load_digits
    digits = load_digits()
    Xall, yall = digits.data, digits.target
    # rebuild test indices the same way
    torch.manual_seed(1)
    n = Xall.shape[0]
    idx = torch.randperm(n)
    n_te = 360
    te = idx[:n_te].numpy()
    imgs = Xall[te].reshape(-1, 8, 8)

    disagree = np.where(pa != ps)[0]
    # prefer interesting cases: one correct one wrong
    interesting = [i for i in disagree if (pa[i] == yte[i]) != (ps[i] == yte[i])]
    show = interesting[:12] if interesting else list(disagree[:12])
    if not show:
        show = list(range(12))

    k = min(12, len(show))
    fig, axes = plt.subplots(2, k, figsize=(1.1 * k, 2.6))
    if k == 1:
        axes = axes.reshape(2, 1)
    for j, i in enumerate(show):
        axes[0, j].imshow(imgs[i], cmap="gray_r")
        axes[0, j].set_title(f"t={int(yte[i])}", fontsize=7)
        axes[0, j].axis("off")
        ok_a, ok_s = pa[i] == yte[i], ps[i] == yte[i]
        axes[1, j].text(
            0.5, 0.55, f"A:{int(pa[i])}\nS:{int(ps[i])}",
            ha="center", va="center", fontsize=8,
            color=("tab:green" if ok_s and not ok_a else
                   "tab:red" if ok_a and not ok_s else "black"),
        )
        axes[1, j].axis("off")
    fig.suptitle(
        f"Digits + 30% label noise — AdamW vs SR-Adam (seed 1)\n"
        f"test acc A={acc_a:.3f}  S={acc_s:.3f}  "
        f"(green=only SR correct, red=only Adam correct)",
        fontsize=9,
    )
    fig.tight_layout()
    for dest in (FIGS / "clf_digits_noise_qualitative.png",
                 FIGS / "clf_digits_noise_qualitative.pdf"):
        try:
            fig.savefig(dest, dpi=150)
        except PermissionError:
            pass
    plt.close(fig)
    return dict(adamw=acc_a, sradam=acc_s, n_disagree=int(len(disagree)),
                n_interesting=int(len(interesting)), n_shown=int(len(show)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    seeds = [1, 2] if args.quick else list(range(1, args.seeds + 1))
    steps = 250 if args.quick else args.steps
    rows = []
    t0 = time.time()

    loaders = {
        "year": load_year,
        "blogfeedback": load_blog,
        "protein": load_protein,
    }

    for name, fn in loaders.items():
        print(f"\n=== regression: {name} ===", flush=True)
        try:
            X, y = fn()
        except Exception as e:
            print(f"  SKIP {type(e).__name__}: {e}", flush=True)
            rows.append(dict(dataset=name, status="skipped", error=str(e)[:300]))
            continue
        print(f"  n={X.shape[0]} d={X.shape[1]}", flush=True)
        if not np.isfinite(X).all():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(y).all():
            m = np.isfinite(y)
            X, y = X[m], y[m]
        per = {o: [] for o in OPTS}
        for seed in seeds:
            Xtr, ytr, Xte, yte, ysd = split_scale(X, y, test_frac=0.3, seed=seed)
            batch = 64 if X.shape[0] > 1000 else max(8, X.shape[0] // 4)
            for o in OPTS:
                per[o].append(run_linear(Xtr, ytr, Xte, yte, o, steps, 2e-2, batch, seed))
        rec = paired(per)
        rec.update(dataset=name, model="linear", n=int(X.shape[0]), d=int(X.shape[1]),
                   status="ok")
        rows.append(rec)
        print(f"  AdamW={rec['adamw']:.4f}  SR={rec['sradam']:.4f}  "
              f"Δ={rec['paired_diff']:+.4f}  win={rec['win_sradam']}", flush=True)

    print("\n=== classification: digits + 30% label noise ===", flush=True)
    per = {o: [] for o in OPTS}
    for seed in seeds:
        for o in OPTS:
            acc, *_ = run_logistic_digits(o, seed=seed, noise=0.3, steps=steps)
            per[o].append(acc)
    # higher acc better → invert loss convention
    d = np.asarray(per["sradam"]) - np.asarray(per["adamw"])
    rec = dict(
        dataset="digits_noisy_logistic", model="logistic", noise=0.3,
        adamw=float(np.mean(per["adamw"])), adamw_sd=float(np.std(per["adamw"], ddof=1)),
        sradam=float(np.mean(per["sradam"])), sradam_sd=float(np.std(per["sradam"], ddof=1)),
        paired_diff=float(d.mean()),  # positive = SR higher acc
        t_stat=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else 0.0,
        win_sradam=bool(d.mean() > 0),
        metric="test_accuracy",
        status="ok",
    )
    rows.append(rec)
    print(f"  AdamW={rec['adamw']:.3f}  SR={rec['sradam']:.3f}  "
          f"Δ={rec['paired_diff']:+.3f}  win={rec['win_sradam']}", flush=True)

    print("\n=== qualitative figure ===", flush=True)
    q = digits_qualitative_figure()
    print(" ", q, flush=True)
    rec["qualitative"] = q

    payload = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             seconds=time.time() - t0,
                             seeds=seeds, steps=steps,
                             note="regression: positive Δ = lower MSE for SR; "
                                  "classification: positive Δ = higher acc for SR"),
                   rows=rows)
    (OUT / "path6_year_blog_clf.json").write_text(json.dumps(payload, indent=2))

    ok = [r for r in rows if r.get("status") == "ok"]
    wins = sum(1 for r in ok if r["win_sradam"])
    print(f"\n{wins}/{len(ok)} cells: SR-Adam better", flush=True)
    print(f"wrote {OUT / 'path6_year_blog_clf.json'}", flush=True)
    print(f"figure: {FIGS / 'clf_digits_noise_qualitative.png'}", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
