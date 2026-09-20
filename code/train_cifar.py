"""
CIFAR-10 / CIFAR-100 training and ablation driver for SR-Adam.

The script reproduces the full experimental grid reported in the paper.  It is written to
be run on a single GPU; the complete grid used for the paper is

    python3 train_cifar.py --grid main        # optimiser comparison, batch size 512
    python3 train_cifar.py --grid batch       # batch-size ablation
    python3 train_cifar.py --grid scope       # shrinkage-scope ablation
    python3 train_cifar.py --grid diagnostics # per-layer diagnostics, one run

Results are appended to a CSV, one row per (dataset, noise, optimiser, batch size, seed,
epoch), so that every table and figure in the paper can be regenerated from a single file.

Only the optimiser changes between arms.  Backbone, preprocessing, epoch count, batch
size, input-noise injection, evaluation procedure and seed schedule are held fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sr_adam import SRAdam, conv_only_filter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------ model
class SimpleCNN(nn.Module):
    """Two convolution blocks followed by two fully connected layers."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


class GaussianNoise:
    """Additive Gaussian input noise, applied after normalisation."""

    def __init__(self, sd):
        self.sd = sd

    def __call__(self, x):
        return x if self.sd == 0 else x + self.sd * torch.randn_like(x)


def loaders(dataset, noise_sd, batch_size, root="./data", workers=4):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        GaussianNoise(noise_sd),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        GaussianNoise(noise_sd),
    ])
    cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    tr = cls(root, train=True, download=True, transform=tf_train)
    te = cls(root, train=False, download=True, transform=tf_test)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=workers,
                       pin_memory=True, drop_last=True),
            DataLoader(te, batch_size=1024, shuffle=False, num_workers=workers,
                       pin_memory=True))


# -------------------------------------------------------------------------- optimisers
def make_optimizer(name, params, lr=None, wd=0.0):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr or 0.01)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr or 0.01, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr or 1e-3)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr or 1e-3, weight_decay=wd or 1e-2)
    if name == "amsgrad":
        return torch.optim.AdamW(params, lr=lr or 1e-3, weight_decay=wd, amsgrad=True)
    if name == "sradam":
        return SRAdam(params, lr=lr or 1e-3, weight_decay=wd, track_stats=False)
    if name == "sradam_conv":                      # layer-type rule of the first version
        return SRAdam(params, lr=lr or 1e-3, weight_decay=wd,
                      shrink_filter=conv_only_filter)
    if name == "sradam_peff":                      # conservative constant
        return SRAdam(params, lr=lr or 1e-3, weight_decay=wd, constant="peff")
    if name == "sradam_nogate":                    # gate disabled
        return SRAdam(params, lr=lr or 1e-3, weight_decay=wd, peff_margin=-1.0)
    if name == "sradam_nodebias":                  # variance correction switched off
        opt = SRAdam(params, lr=lr or 1e-3, weight_decay=wd)
        import sr_adam
        sr_adam.ema_variance_correction = lambda t, b: 1.0
        return opt
    raise ValueError(name)


# ------------------------------------------------------------------------------ train
@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        out = net(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        tot += y.numel()
    return 100.0 * correct / tot, loss_sum / tot


def train_one(dataset, noise_sd, opt_name, batch_size, seed, epochs, writer,
              diagnostics=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_classes = 10 if dataset == "cifar10" else 100
    tr, te = loaders(dataset, noise_sd, batch_size)
    net = SimpleCNN(n_classes).to(DEVICE)
    opt = make_optimizer(opt_name, net.parameters())
    if diagnostics and isinstance(opt, SRAdam):
        opt.track_stats = True
    diag_rows = []

    for ep in range(1, epochs + 1):
        net.train()
        t0 = time.time()
        for it, (x, y) in enumerate(tr):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(x), y).backward()
            opt.step()
            if diagnostics and isinstance(opt, SRAdam) and it % 20 == 0 and opt.stats:
                for name, st in zip([n for n, _ in net.named_parameters()], opt.stats):
                    diag_rows.append(dict(epoch=ep, iter=it, param=name, **st))
        acc, loss = evaluate(net, te)
        writer.writerow(dict(dataset=dataset, noise=noise_sd, optimizer=opt_name,
                             batch_size=batch_size, seed=seed, epoch=ep,
                             test_acc=acc, test_loss=loss,
                             epoch_time=time.time() - t0))
        print(f"  {dataset} noise={noise_sd} {opt_name} B={batch_size} seed={seed} "
              f"epoch {ep:3d}  acc={acc:6.2f}  loss={loss:.4f}")
    return diag_rows


# ------------------------------------------------------------------------------- grids
FIELDS = ["dataset", "noise", "optimizer", "batch_size", "seed", "epoch",
          "test_acc", "test_loss", "epoch_time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main",
                    choices=["main", "batch", "scope", "diagnostics"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()

        if args.grid == "main":
            for ds in ["cifar10", "cifar100"]:
                for noise in [0.0, 0.05, 0.1]:
                    for opt in ["sgd", "momentum", "adam", "adamw", "amsgrad", "sradam"]:
                        for seed in range(args.seeds):
                            train_one(ds, noise, opt, 512, seed, args.epochs, w)
                            fh.flush()

        elif args.grid == "batch":
            for ds in ["cifar10", "cifar100"]:
                for bs in [64, 128, 256, 512, 1024]:
                    for opt in ["adam", "amsgrad", "sradam"]:
                        for seed in range(min(args.seeds, 3)):
                            train_one(ds, 0.05, opt, bs, seed, args.epochs, w)
                            fh.flush()

        elif args.grid == "scope":
            for ds in ["cifar10", "cifar100"]:
                for noise in [0.0, 0.05, 0.1]:
                    for opt in ["sradam", "sradam_conv", "sradam_nogate",
                                "sradam_peff", "sradam_nodebias"]:
                        for seed in range(min(args.seeds, 3)):
                            train_one(ds, noise, opt, 512, seed, args.epochs, w)
                            fh.flush()

        else:
            rows = train_one("cifar10", 0.05, "sradam", 512, 0, args.epochs, w,
                             diagnostics=True)
            with open("diagnostics.json", "w") as g:
                json.dump(rows, g)
            print(f"  wrote {len(rows)} diagnostic records to diagnostics.json")


if __name__ == "__main__":
    main()
