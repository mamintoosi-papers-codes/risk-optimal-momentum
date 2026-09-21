import csv
import json
import math
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(r"C:\git\mamintoosi-papers-codes\risk-optimal-momentum\results\runs\main_20260921-085656")
rows = list(csv.DictReader((RUN / "results.csv").open()))
meta = json.loads((RUN / "run_meta.json").read_text())

print("=== run_meta ===")
print(json.dumps(meta, indent=2))

print("\n=== results.csv overview ===")
print("rows", len(rows))
for col in ["dataset", "noise", "optimizer", "batch_size", "seed", "epoch"]:
    vals = sorted({r[col] for r in rows}, key=lambda x: (len(str(x)), str(x)))
    print(f"{col}: {vals}")

print("\nrows by dataset:", Counter(r["dataset"] for r in rows))
print("rows by optimizer:", Counter(r["optimizer"] for r in rows))
print("rows by (dataset, optimizer):")
for k, v in sorted(Counter((r["dataset"], r["optimizer"]) for r in rows).items()):
    print(" ", k, v)

# completeness
runs = defaultdict(list)
for r in rows:
    key = (r["dataset"], r["noise"], r["optimizer"], r["batch_size"], r["seed"])
    runs[key].append(int(float(r["epoch"])))

print("\n=== run completeness ===")
n_ok = n_bad = 0
missing = []
expected_seeds = set(str(i) for i in range(int(meta.get("seeds", 5))))
expected_opts = ["sgd", "momentum", "adam", "adamw", "amsgrad", "sradam"]
for ds in ["cifar10", "cifar100"]:
    for noise in ["0.0", "0.05", "0.1"]:
        for opt in expected_opts:
            for seed in sorted(expected_seeds):
                key = (ds, noise, opt, "512", seed)
                eps = runs.get(key, [])
                if not eps:
                    missing.append(key)
                    n_bad += 1
                    continue
                if max(eps) == 20 and len(set(eps)) == 20:
                    n_ok += 1
                else:
                    n_bad += 1
                    missing.append(key + (f"epochs={sorted(set(eps))}",))
print(f"complete runs (20 unique epochs): {n_ok}")
print(f"incomplete/missing: {n_bad}")
if missing:
    print("missing/incomplete sample:")
    for m in missing[:30]:
        print(" ", m)

# final-epoch accuracy
final_acc = defaultdict(list)
final_loss = defaultdict(list)
best_acc = defaultdict(list)
time_sum = defaultdict(list)
for key, eps in runs.items():
    max_ep = max(eps)
    # if duplicate epochs, take last occurrence in file order
    last = {}
    for r in rows:
        k2 = (r["dataset"], r["noise"], r["optimizer"], r["batch_size"], r["seed"])
        if k2 != key:
            continue
        ep = int(float(r["epoch"]))
        last[ep] = r
    for ep, r in last.items():
        acc_key = (r["dataset"], r["noise"], r["optimizer"], r["batch_size"])
        best_acc[(r["dataset"], r["noise"], r["optimizer"], r["batch_size"], r["seed"])].append(float(r["test_acc"]))
        time_sum[acc_key].append(float(r["epoch_time"]))
        if ep == max_ep:
            final_acc[acc_key].append(float(r["test_acc"]))
            final_loss[acc_key].append(float(r["test_loss"]))

best_mean = defaultdict(list)
for (ds, noise, opt, bs, seed), accs in best_acc.items():
    best_mean[(ds, noise, opt, bs)].append(max(accs))

print("\n=== FINAL-EPOCH test accuracy (mean ± std over seeds) — paper metric ===")
header = f"{'dataset':9s} {'noise':6s} {'optimizer':10s} {'n':>3s} {'final_acc':>16s} {'best_acc':>8s} {'final_loss':>10s} {'epoch_s':>8s}"
print(header)
order = ["sgd", "momentum", "adam", "adamw", "amsgrad", "sradam"]
table = {}
for ds in ["cifar10", "cifar100"]:
    for noise in ["0.0", "0.05", "0.1"]:
        print("-" * len(header))
        for opt in order:
            key = (ds, noise, opt, "512")
            if key not in final_acc:
                print(f"{ds:9s} {noise:6s} {opt:10s}   -- MISSING --")
                continue
            accs = final_acc[key]
            losses = final_loss[key]
            mu = sum(accs) / len(accs)
            sd = stats.stdev(accs) if len(accs) > 1 else 0.0
            lm = sum(losses) / len(losses)
            bm = sum(best_mean[key]) / len(best_mean[key])
            et = sum(time_sum[key]) / len(time_sum[key]) if time_sum[key] else float("nan")
            table[(ds, noise, opt)] = dict(final=mu, final_sd=sd, best=bm, loss=lm, n=len(accs), epoch=et)
            print(f"{ds:9s} {noise:6s} {opt:10s} {len(accs):3d} {mu:8.2f}±{sd:5.2f} {bm:8.2f} {lm:10.4f} {et:8.2f}")

print("\n=== SR-Adam vs baselines (final-epoch acc) ===")
for ds in ["cifar10", "cifar100"]:
    for noise in ["0.0", "0.05", "0.1"]:
        sr = table.get((ds, noise, "sradam"))
        ad = table.get((ds, noise, "adam"))
        aw = table.get((ds, noise, "adamw"))
        am = table.get((ds, noise, "amsgrad"))
        if not sr:
            continue
        print(f"\n{ds} noise={noise}")
        for name, base in [("Adam", ad), ("AdamW", aw), ("AMSGrad", am)]:
            if not base:
                print(f"  vs {name}: missing")
                continue
            d = sr["final"] - base["final"]
            mark = "SR better" if d > 0 else "baseline better"
            # rough effect size
            pooled = math.sqrt((sr["final_sd"]**2 + base["final_sd"]**2) / 2 + 1e-12)
            print(f"  vs {name:8s}: SR={sr['final']:6.2f}±{sr['final_sd']:.2f}  {name}={base['final']:6.2f}±{base['final_sd']:.2f}  Δ={d:+.2f}  ({mark})")
        if ad:
            # best-acc comparison like old paper
            db = sr["best"] - ad["best"]
            print(f"  best-acc (diagnostic only): SR={sr['best']:.2f} Adam={ad['best']:.2f} Δ={db:+.2f}")

print("\n=== Ranking per (dataset, noise) by final_acc ===")
for ds in ["cifar10", "cifar100"]:
    for noise in ["0.0", "0.05", "0.1"]:
        items = []
        for opt in order:
            k = (ds, noise, opt, "512")
            if k in final_acc:
                items.append((opt, sum(final_acc[k])/len(final_acc[k])))
        items.sort(key=lambda x: -x[1])
        print(f"{ds:9s} noise={noise:5s}: " + "  ".join(f"{o}={a:.2f}" for o, a in items))

print("\n=== loss sanity (final vs first epoch, last written) ===")
worse = 0
for key, eps in runs.items():
    last = {}
    first = {}
    for r in rows:
        k2 = (r["dataset"], r["noise"], r["optimizer"], r["batch_size"], r["seed"])
        if k2 != key:
            continue
        ep = int(float(r["epoch"]))
        if ep not in first:
            first[ep] = r
        last[ep] = r
    if 1 in last and max(last) in last:
        if float(last[max(last)]["test_loss"]) > float(last[1]["test_loss"]) + 0.3:
            worse += 1
            print("  WORSE", key, last[1]["test_loss"], "->", last[max(last)]["test_loss"])
print(f"runs with large final loss increase: {worse}")

print("\n=== epoch time total estimate ===")
total_sec = sum(float(r["epoch_time"]) for r in rows)
# unique epochs only roughly
print(f"sum of epoch_time in file: {total_sec/3600:.2f} hours")
print(f"meta says: {meta}")

print("\n=== paper expectations checklist ===")
# Large batch 512: paper claims SR-Adam gains over Adam especially large-batch/noisy
for ds in ["cifar10", "cifar100"]:
    for noise in ["0.0", "0.05", "0.1"]:
        sr = table.get((ds, noise, "sradam"))
        ad = table.get((ds, noise, "adam"))
        if not sr or not ad:
            continue
        d = sr["final"] - ad["final"]
        print(f"  {ds} noise={noise}: Δ(SR-Adam - Adam) final-acc = {d:+.2f}  -> {'OK' if d>0 else 'NOT as paper claims'}")
