import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:\git\mamintoosi-papers-codes\risk-optimal-momentum\results\runs")

for run in ["grouping_20260928-124556", "grouping_20260928-140120"]:
    p = ROOT / run
    print("=" * 70)
    print(run)
    meta = json.loads((p / "run_meta.json").read_text())
    print(json.dumps(meta, indent=2))
    rows = list(csv.DictReader((p / "results.csv").open()))
    print("rows", len(rows), "cols", list(rows[0].keys()) if rows else None)
    if not rows:
        continue
    print("datasets", sorted(set(r["dataset"] for r in rows)))
    print("optimizers", sorted(set(r["optimizer"] for r in rows)))
    print("seeds", sorted(set(r["seed"] for r in rows)))
    print("epochs", sorted(set(int(float(r["epoch"])) for r in rows)))

    # final-epoch accuracy mean±std
    last = {}
    for r in rows:
        k = (r["dataset"], r["optimizer"], r["grouping"], r.get("k_min", "0"), r["seed"])
        ep = int(float(r["epoch"]))
        if k not in last or ep >= last[k][0]:
            last[k] = (ep, float(r["test_acc"]), float(r["test_loss"]))

    acc = defaultdict(list)
    loss = defaultdict(list)
    for (ds, opt, grp, km, seed), (ep, a, l) in last.items():
        acc[(ds, opt, grp, km)].append(a)
        loss[(ds, opt, grp, km)].append(l)

    print("\n=== final-epoch test acc ===")
    for ds in sorted(set(k[0] for k in acc)):
        print(f"-- {ds} --")
        for key in sorted(k for k in acc if k[0] == ds):
            v = acc[key]
            mu = sum(v) / len(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            print(f"  {key[1]:22s} grouping={key[2]:8s} kmin={key[3]:4s} "
                  f"n={len(v)} acc={mu:6.2f}±{sd:.2f}  loss={sum(loss[key])/len(loss[key]):.4f}")

    # compare to adamw
    print("\n=== Δ acc (variant - adamw), higher = better ===")
    for ds in sorted(set(k[0] for k in acc)):
        base = acc.get((ds, "adamw", "adamw", "0"))
        if not base:
            # grouping field for adamw
            for k, v in acc.items():
                if k[0] == ds and k[1] == "adamw":
                    base = v
                    base_key = k
                    break
        else:
            base_key = (ds, "adamw", "adamw", "0")
        if not base:
            print(f"  {ds}: no adamw")
            continue
        bmu = sum(base) / len(base)
        print(f"  {ds}: adamw={bmu:.2f}")
        for key in sorted(k for k in acc if k[0] == ds and k[1] != "adamw"):
            mu = sum(acc[key]) / len(acc[key])
            print(f"    {key[1]:22s} grp={key[2]:8s} kmin={key[3]:4s}  "
                  f"Δ={mu - bmu:+.2f}  acc={mu:.2f}")
