"""
MNIST CNN pilot — domain coverage (vision / CNN).

Small ConvNet, AdamW vs SR-Adam, optional train-label noise.
Complements CIFAR (negative) and non-vision wins.

Run:
  python -u mnist_cnn_pilot.py
  python -u mnist_cnn_pilot.py --seeds 3 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from sr_adam import SRAdam

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "mnist"
OUT = HERE.parent / "out"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)   # 28→14
        x = F.max_pool2d(F.relu(self.c2(x)), 2)   # 14→7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_opt(name, params, lr):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    return SRAdam(params, lr=lr, weight_decay=1e-2, warmup=5)


def load_data(batch_size=256):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    tr = datasets.MNIST(str(DATA), train=True, download=True, transform=tf)
    te = datasets.MNIST(str(DATA), train=False, download=True, transform=tf)
    # keep a small subset on CPU for speed
    n_tr = 20000
    idx = torch.randperm(len(tr))[:n_tr]
    tr = torch.utils.data.Subset(tr, idx)
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True)
    te_loader = torch.utils.data.DataLoader(te, batch_size=1000, shuffle=False)
    return tr, tr_loader, te_loader


def flip_labels(y, frac, gen):
    if frac <= 0:
        return y
    mask = torch.rand(y.shape[0], generator=gen) < frac
    y = y.clone()
    y[mask] = torch.randint(0, 10, (int(mask.sum()),), generator=gen)
    return y


def run_one(tr, tr_loader_factory, te_loader, opt_name, seed, epochs=5, lr=1e-3,
            label_noise=0.0):
    torch.manual_seed(seed)
    net = SmallCNN().to(DEVICE)
    opt = build_opt(opt_name, net.parameters(), lr)

    # flip labels in the subset's targets (aligned with __getitem__)
    ys = torch.tensor([int(tr[i][1]) for i in range(len(tr))], dtype=torch.long)
    if label_noise > 0:
        gen = torch.Generator().manual_seed(900 + seed)
        mask = torch.rand(len(ys), generator=gen) < label_noise
        ys[mask] = torch.randint(0, 10, (int(mask.sum()),), generator=gen)

    class LabeledSubset(torch.utils.data.Dataset):
        def __init__(self, base, labels):
            self.base = base
            self.labels = labels

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            x, _ = self.base[i]
            return x, int(self.labels[i])

    train_ds = LabeledSubset(tr, ys)
    tr_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))

    for _ep in range(epochs):
        net.train()
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(net(xb), yb).backward()
            opt.step()

    net.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = net(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=12000)
    args = ap.parse_args()

    print(f"device={DEVICE}", flush=True)
    tr, _tr_loader, te_loader = load_data()
    print(f"train subset {len(tr)}  test 10000", flush=True)

    rows = []
    t0 = time.time()
    for noise in (0.0, 0.3):
        per = {"adamw": [], "sradam": []}
        for seed in range(1, args.seeds + 1):
            for o in ("adamw", "sradam"):
                acc = run_one(tr, None, te_loader, o, seed,
                              epochs=args.epochs, label_noise=noise)
                per[o].append(acc)
                print(f"  noise={noise} {o} seed={seed} acc={acc:.2f}", flush=True)
        a = np.asarray(per["adamw"])
        s = np.asarray(per["sradam"])
        d = s - a
        rec = dict(
            dataset="mnist_cnn", model="SmallCNN", label_noise=noise,
            epochs=args.epochs, seeds=args.seeds, device=DEVICE,
            adamw=float(a.mean()), adamw_sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            sradam=float(s.mean()), sradam_sd=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            paired_diff=float(d.mean()),  # + = SR higher acc
            win_sradam=bool(d.mean() > 0),
            metric="test_accuracy_pct",
        )
        rows.append(rec)
        print(f"noise={noise}  AdamW={rec['adamw']:.2f}  SR={rec['sradam']:.2f}  "
              f"Δ={rec['paired_diff']:+.2f}  win={rec['win_sradam']}", flush=True)

    payload = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             seconds=time.time() - t0,
                             note="positive Δ = higher test acc for SR-Adam"),
                   rows=rows)
    (OUT / "mnist_cnn_pilot.json").write_text(json.dumps(payload, indent=2))
    wins = sum(1 for r in rows if r["win_sradam"])
    print(f"\n{wins}/{len(rows)} noise settings: SR-Adam higher test acc", flush=True)
    print(f"wrote {OUT / 'mnist_cnn_pilot.json'}", flush=True)
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
