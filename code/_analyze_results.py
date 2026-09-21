import csv
import json
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path

csv_path = Path("results.csv")
rows = list(csv.DictReader(csv_path.open()))
print("=== results.csv ===")
print("rows", len(rows))
print("columns", list(rows[0].keys()) if rows else None)

for col in ["dataset", "noise", "optimizer", "batch_size", "seed", "epoch"]:
    vals = sorted({r[col] for r in rows}, key=lambda x: (len(str(x)), str(x)))
    print(f"{col}: n_unique={len(vals)} values={vals}")

print("\nrows by dataset:", Counter(r["dataset"] for r in rows))
print("rows by optimizer:", Counter(r["optimizer"] for r in rows))
print("rows by (dataset, optimizer):")
for k, v in sorted(Counter((r["dataset"], r["optimizer"]) for r in rows).items()):
    print(" ", k, v)

print("\n--- run completeness ---")
runs = defaultdict(list)
for r in rows:
    key = (r["dataset"], r["noise"], r["optimizer"], r["batch_size"], r["seed"])
    runs[key].append(int(float(r["epoch"])))

for key, eps in sorted(runs.items()):
    eps = sorted(eps)
    ok = len(eps) == max(eps)
    print(key, "epochs", eps[0], "..", eps[-1], "n=", len(eps), "max", max(eps), "OK" if ok else "INCOMPLETE")

print("\n=== final-epoch test_acc mean±std ===")
final_acc = defaultdict(list)
final_loss = defaultdict(list)
best_acc = defaultdict(list)
for key, eps in runs.items():
    max_ep = max(eps)
    for r in rows:
        k2 = (r["dataset"], r["noise"], r["optimizer"], r["batch_size"], r["seed"])
        if k2 != key:
            continue
        ep = int(float(r["epoch"]))
        if ep == max_ep:
            final_acc[(r["dataset"], r["noise"], r["optimizer"], r["batch_size"])].append(float(r["test_acc"]))
            final_loss[(r["dataset"], r["noise"], r["optimizer"], r["batch_size"])].append(float(r["test_loss"]))

for r in rows:
    best_acc[(r["dataset"], r["noise"], r["optimizer"], r["batch_size"], r["seed"])].append(float(r["test_acc"]))

best_mean = defaultdict(list)
for (ds, noise, opt, bs, seed), accs in best_acc.items():
    best_mean[(ds, noise, opt, bs)].append(max(accs))

for key in sorted(final_acc):
    accs = final_acc[key]
    losses = final_loss[key]
    mu = sum(accs) / len(accs)
    sd = stats.stdev(accs) if len(accs) > 1 else 0.0
    lm = sum(losses) / len(losses)
    bm = sum(best_mean[key]) / len(best_mean[key])
    print(
        f"{key[0]:9s} noise={key[1]:5s} {key[2]:14s} B={key[3]:5s} "
        f"n={len(accs)}  final_acc={mu:6.2f}±{sd:.2f}  best_acc={bm:6.2f}  loss={lm:.4f}"
    )

print("\n=== Adam vs SR-Adam final-acc (paper idea check) ===")
pairs = defaultdict(dict)
for (ds, noise, opt, bs), accs in final_acc.items():
    mu = sum(accs) / len(accs)
    pairs[(ds, noise, bs)][opt] = mu
for key in sorted(pairs):
    d = pairs[key]
    if "adam" in d and "sradam" in d:
        delta = d["sradam"] - d["adam"]
        mark = "SR better" if delta > 0 else "Adam better"
        print(f"{key[0]:9s} noise={key[1]:5s} B={key[2]:5s}  Adam={d['adam']:6.2f}  SR-Adam={d['sradam']:6.2f}  Δ={delta:+.2f}  {mark}")
        for other in ["adamw", "amsgrad"]:
            if other in d:
                print(f"{'':9s} {'':5s} {'':5s}  {other}={d[other]:6.2f}")

print("\n=== epoch_time by optimizer ===")
et = defaultdict(list)
for r in rows:
    et[r["optimizer"]].append(float(r["epoch_time"]))
for opt, vals in sorted(et.items()):
    print(f"{opt:14s} n={len(vals):4d} mean={sum(vals)/len(vals):.2f}s min={min(vals):.2f} max={max(vals):.2f}")

print("\n=== loss trajectory sanity (should generally decrease) ===")
traj = defaultdict(list)
for r in rows:
    traj[(r["dataset"], r["noise"], r["optimizer"], r["seed"])].append(
        (int(float(r["epoch"])), float(r["test_loss"]), float(r["test_acc"]))
    )
bad = 0
for key, series in sorted(traj.items()):
    series = sorted(series)
    if len(series) < 2:
        continue
    first_l, last_l = series[0][1], series[-1][1]
    first_a, last_a = series[0][2], series[-1][2]
    worse = last_l > first_l + 0.5
    if worse:
        bad += 1
        print("  LOSS WORSENED", key, "first", first_l, "last", last_l)
print(f"runs with large loss increase: {bad} / {len(traj)}")
# print one CIFAR10 sradam trajectory
for key, series in sorted(traj.items()):
    if key[0] == "cifar10" and key[2] == "sradam" and key[1] == "0.05":
        series = sorted(series)
        print("  sample traj", key)
        for ep, l, a in series:
            print(f"    ep={ep:2d} loss={l:.4f} acc={a:.2f}")

print("\n=== diagnostics.json ===")
diag = json.loads(Path("diagnostics.json").read_text())
print("type", type(diag).__name__, "n", len(diag) if hasattr(diag, "__len__") else None)
if isinstance(diag, list) and diag:
    print("keys sample", list(diag[0].keys()))
    print("first record:")
    print(" ", diag[0])
    by_param = defaultdict(list)
    for d in diag:
        by_param[d.get("param", "?")].append(d)
    print("\nparams:", list(by_param.keys()))
    print("\nper-param diagnostics:")
    for p, recs in by_param.items():
        n = len(recs)
        k = [r["shrink_weight"] for r in recs if r.get("shrink_weight") == r.get("shrink_weight")]
        peff = [r["peff"] for r in recs if r.get("peff") == r.get("peff")]
        s2 = [r["sigma2"] for r in recs if r.get("sigma2") == r.get("sigma2")]
        gate = [r for r in recs if r.get("peff") == r.get("peff") and r["peff"] > 3.0]
        numel = recs[0].get("numel")
        adam_w = recs[0].get("adam_weight")

        def m(x):
            return sum(x) / len(x) if x else float("nan")

        frac = (len(gate) / n) if n else 0.0
        print(
            f"  {p:12s} numel={numel} n={n} mean_k={m(k):.3f} "
            f"mean_peff={m(peff):.2f} mean_s2={m(s2):.4f} "
            f"gate_open_frac={frac:.2f} adam_w={adam_w}"
        )

    # gate open by layer type
    print("\n=== gate-open fraction vs paper claim ===")
    for p, recs in by_param.items():
        peffs = [r["peff"] for r in recs if r.get("peff") == r.get("peff")]
        if not peffs:
            print(f"  {p}: no peff (gate never computed / always closed early)")
            continue
        open_n = sum(1 for x in peffs if x > 3.0)
        print(f"  {p}: gate open {100*open_n/len(peffs):.1f}%  peff range [{min(peffs):.2f}, {max(peffs):.2f}]")
