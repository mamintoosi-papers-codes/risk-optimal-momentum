"""
GNN pilot — node classification on Cora (Planetoid / PyG).

Compares AdamW vs SR-Adam on the same GCN, with optional label noise on the
training labels (a regime where shrinkage is expected to help).

Data (not in git): data/Cora/  (copied from a local PyG cache)

Run:
  python -u gnn_pilot.py
  python -u gnn_pilot.py --seeds 5 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from sr_adam import SRAdam

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "Cora"
OUT = HERE.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
if not DATA.exists():
    DATA = HERE.parent / "data" / "Cora"


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid, out_dim, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, e = data.x, data.edge_index
        x = self.c1(x, e).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.c2(x, e)


def build_opt(name, params, lr):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)
    if name == "amsgrad":
        return torch.optim.AdamW(params, lr=lr, weight_decay=5e-4, amsgrad=True)
    return SRAdam(params, lr=lr, weight_decay=5e-4, warmup=5)


def run_once(data, opt_name, seed, epochs=150, lr=0.01, label_noise=0.0, hid=16):
    torch.manual_seed(seed)
    y = data.y.clone()
    if label_noise > 0:
        n = y.size(0)
        mask = torch.rand(n) < label_noise
        y[mask] = torch.randint(0, int(y.max()) + 1, (int(mask.sum()),))
    model = GCN(data.num_features, hid, int(data.y.max().item() + 1))
    opt = build_opt(opt_name, model.parameters(), lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], y[data.train_mask])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(1)
        acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        acc_te = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return acc, acc_te, pred


def paired_summary(per):
    a = np.asarray(per["adamw"])
    b = np.asarray(per["sradam"])
    d = a - b  # accuracy higher better → AdamW - SR < 0 means SR better
    return dict(
        adamw=float(a.mean()), adamw_sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        sradam=float(b.mean()), sradam_sd=float(b.std(ddof=1)) if len(b) > 1 else 0.0,
        paired_diff=float(-d.mean()),  # positive = SR-Adam higher accuracy
        win_sradam=bool(b.mean() > a.mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()

    print(f"Loading Cora from {DATA}", flush=True)
    dataset = Planetoid(root=str(DATA), name="Cora")
    data = dataset[0]
    print(data, flush=True)

    rows = []
    t0 = time.time()
    for noise in (0.0, 0.3, 0.5):
        per_te = {"adamw": [], "sradam": []}
        preds = {}
        for seed in range(1, args.seeds + 1):
            for o in ("adamw", "sradam"):
                acc_val, acc_te, pred = run_once(
                    data, o, seed, epochs=args.epochs, label_noise=noise)
                per_te[o].append(acc_te)
                if seed == 1:
                    preds[o] = pred
        rec = paired_summary(per_te)
        rec.update(dataset="Cora", task="node_classification",
                   epochs=args.epochs, label_noise=noise)
        rows.append(rec)
        print(f"label_noise={noise:.1f}  AdamW={rec['adamw']:.3f}±{rec['adamw_sd']:.3f}  "
              f"SR-Adam={rec['sradam']:.3f}±{rec['sradam_sd']:.3f}  "
              f"Δ={rec['paired_diff']:+.3f}  win={rec['win_sradam']}", flush=True)

        # qualitative agreement table (seed 1)
        if noise == 0.0:
            pt, ps = preds["adamw"], preds["sradam"]
            te = data.test_mask
            both = int(((pt == ps) & te).sum())
            only_a = int(((pt != ps) & (pt == data.y) & te).sum())
            only_s = int(((pt != ps) & (ps == data.y) & te).sum())
            both_wrong = int(((pt != ps) & (pt != data.y) & (ps != data.y) & te).sum())
            rec["agreement"] = dict(both_same=both, adam_only_correct=only_a,
                                    sradam_only_correct=only_s, both_wrong=both_wrong,
                                    n_test=int(te.sum()))
            print("  test agreement (seed1):", rec["agreement"], flush=True)

    payload = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             seconds=time.time() - t0,
                             note="positive Δ = higher test acc for SR-Adam"),
                   rows=rows)
    (OUT / "gnn_pilot_cora.json").write_text(json.dumps(payload, indent=2))
    wins = sum(1 for r in rows if r["win_sradam"])
    print(f"\n{wins}/{len(rows)} label-noise settings: SR-Adam higher test acc",
          flush=True)
    print(f"wrote {OUT / 'gnn_pilot_cora.json'}", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
