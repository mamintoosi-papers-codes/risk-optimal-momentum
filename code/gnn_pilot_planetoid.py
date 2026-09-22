"""
Extra Planetoid / PyG node-classification datasets (CiteSeer, PubMed).

Same protocol as gnn_pilot.py (Cora): GCN, AdamW vs SR-Adam, optional label
noise on training labels. Downloads to data/<Name>/ if not present (gitignored).

Run:
  python -u gnn_pilot_planetoid.py
  python -u gnn_pilot_planetoid.py --seeds 3 --epochs 120
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import Planetoid

from gnn_pilot import GCN, build_opt, paired_summary

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "data"
OUT = HERE.parent / "out"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def run_once(data, opt_name, seed, epochs=120, lr=0.01, label_noise=0.0, hid=16):
    torch.manual_seed(seed)
    y = data.y.clone()
    if label_noise > 0:
        n = y.size(0)
        mask = torch.rand(n) < label_noise
        y[mask] = torch.randint(0, int(data.y.max()) + 1, (int(mask.sum()),))
    model = GCN(data.num_features, hid, int(data.y.max().item() + 1))
    opt = build_opt(opt_name, model.parameters(), lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = torch.nn.functional.cross_entropy(out[data.train_mask], y[data.train_mask])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(1)
        acc_te = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return acc_te, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--names", nargs="+", default=["CiteSeer", "PubMed"])
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    for name in args.names:
        print(f"\n=== {name} ===", flush=True)
        root = DATA_ROOT / name
        dataset = Planetoid(root=str(root), name=name)
        data = dataset[0]
        print(data, flush=True)

        for noise in (0.0, 0.3):
            per_te = {"adamw": [], "sradam": []}
            preds = {}
            for seed in range(1, args.seeds + 1):
                for o in ("adamw", "sradam"):
                    acc_te, pred = run_once(
                        data, o, seed, epochs=args.epochs, label_noise=noise)
                    per_te[o].append(acc_te)
                    if seed == 1:
                        preds[o] = pred
            rec = paired_summary(per_te)
            rec.update(dataset=name, task="node_classification",
                       epochs=args.epochs, label_noise=noise)
            rows.append(rec)
            print(f"noise={noise:.1f}  AdamW={rec['adamw']:.3f}±{rec['adamw_sd']:.3f}  "
                  f"SR={rec['sradam']:.3f}±{rec['sradam_sd']:.3f}  "
                  f"Δ={rec['paired_diff']:+.3f}  win={rec['win_sradam']}", flush=True)
            if noise == 0.0:
                pt, ps = preds["adamw"], preds["sradam"]
                te = data.test_mask
                rec["agreement"] = dict(
                    both_same=int(((pt == ps) & te).sum()),
                    adam_only_correct=int(((pt != ps) & (pt == data.y) & te).sum()),
                    sradam_only_correct=int(((pt != ps) & (ps == data.y) & te).sum()),
                    n_test=int(te.sum()),
                )
                print("  agreement:", rec["agreement"], flush=True)

    payload = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  seconds=time.time() - t0, seeds=args.seeds, epochs=args.epochs,
                  note="positive Δ = higher test acc for SR-Adam"),
        rows=rows,
    )
    (OUT / "gnn_pilot_planetoid.json").write_text(json.dumps(payload, indent=2))
    wins = sum(1 for r in rows if r["win_sradam"])
    print(f"\n{wins}/{len(rows)} cells: SR-Adam higher test acc", flush=True)
    print(f"wrote {OUT / 'gnn_pilot_planetoid.json'}", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
