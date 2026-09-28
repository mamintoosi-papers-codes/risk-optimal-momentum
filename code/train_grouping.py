"""
Grouping ablation on vision CNNs: block vs filter vs channel Stein weights.

Datasets: MNIST, CIFAR-10.  Optimiser variants:
  adamw
  sradam_block    one k_t per parameter tensor (Algorithm 1 default)
  sradam_filter   one k_t per conv filter / dense output row (dim 0)
  sradam_channel  alias of filter for conv/linear weights
  sradam_channel_kmin  channel-wise with k_t floored at 0.3

Run (GPU machine):
  python -u train_grouping.py --dataset both --seeds 3 --epochs 5 --out results_grouping.csv

Smoke:
  python -u train_grouping.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sr_adam import SRAdam

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "data"
OUT_ROOT = HERE.parent / "results" / "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIELDS = ["dataset", "optimizer", "grouping", "k_min", "seed", "epoch",
          "test_acc", "test_loss", "epoch_time"]


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10, in_ch=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def make_optimizer(name, params, lr=1e-3):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    grouping = name.replace("sradam_", "")
    k_min = 0.0
    if grouping.endswith("kmin"):
        k_min = 0.3
        grouping = grouping.replace("_kmin", "")
    if grouping not in ("block", "filter", "channel"):
        grouping = "block"
    return SRAdam(params, lr=lr, weight_decay=1e-2, warmup=5,
                  grouping=grouping, k_min=k_min)


def loaders(dataset, batch_size, subset=None):
    if dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        tr = datasets.MNIST(str(DATA_ROOT / "mnist"), train=True, download=True, transform=tf)
        te = datasets.MNIST(str(DATA_ROOT / "mnist"), train=False, download=True, transform=tf)
        net_in = 1
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        tfe = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        tr = datasets.CIFAR10(str(DATA_ROOT / "cifar10"), train=True, download=True, transform=tf)
        te = datasets.CIFAR10(str(DATA_ROOT / "cifar10"), train=False, download=True, transform=tfe)
        net_in = 3
    if subset is not None:
        tr = Subset(tr, list(range(min(subset, len(tr)))))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(te, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True),
            net_in)


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    tot = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        out = net(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        tot += y.numel()
    return 100.0 * correct / tot, loss_sum / tot


def train_one(dataset, opt_name, seed, epochs, batch_size, subset, writer):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, te, in_ch = loaders(dataset, batch_size, subset=subset)
    net = SmallCNN(n_classes=10, in_ch=in_ch).to(DEVICE)
    opt = make_optimizer(opt_name, net.parameters())
    grouping = "adamw" if opt_name == "adamw" else opt_name.replace("sradam_", "")
    k_min = 0.3 if opt_name.endswith("kmin") else 0.0
    for ep in range(1, epochs + 1):
        net.train()
        t0 = time.time()
        for x, y in tr:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(x), y).backward()
            opt.step()
        acc, loss = evaluate(net, te)
        writer.writerow(dict(dataset=dataset, optimizer=opt_name, grouping=grouping,
                             k_min=k_min, seed=seed, epoch=ep, test_acc=acc,
                             test_loss=loss, epoch_time=time.time() - t0))
        print(f"  {dataset} {opt_name} seed={seed} ep={ep:2d} acc={acc:6.2f} loss={loss:.4f}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both", choices=["mnist", "cifar10", "both"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--subset", type=int, default=None, help="limit train set (smoke)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--groupings", nargs="+",
                    default=["adamw", "sradam_block", "sradam_filter",
                             "sradam_channel", "sradam_channel_kmin"])
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
        args.seeds = 1
        args.subset = 256
        args.groupings = ["adamw", "sradam_block", "sradam_channel"]
        args.dataset = "mnist"

    dss = ["mnist", "cifar10"] if args.dataset == "both" else [args.dataset]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_ROOT / f"grouping_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out if args.out else str(run_dir / "results.csv")
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), device=DEVICE,
                torch=torch.__version__, command=" ".join(sys.argv),
                datasets=dss, groupings=args.groupings, epochs=args.epochs,
                seeds=args.seeds, subset=args.subset, smoke=args.smoke)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("Run dir:", run_dir, "device:", DEVICE, flush=True)

    new = not os.path.exists(csv_path)
    with open(csv_path, "a" if not args.smoke else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new or args.smoke:
            w.writeheader()
        for ds in dss:
            for opt_name in args.groupings:
                for seed in range(1, args.seeds + 1):
                    print(f"=== {ds} {opt_name} seed={seed} ===", flush=True)
                    train_one(ds, opt_name, seed, args.epochs, args.batch_size,
                              args.subset, w)
                    fh.flush()
    print("Done.", csv_path, flush=True)


if __name__ == "__main__":
    main()
