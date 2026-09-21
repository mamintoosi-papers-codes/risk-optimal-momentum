"""
CIFAR-10 / CIFAR-100 training and ablation driver for SR-Adam.

The script reproduces the full experimental grid reported in the paper.  It is written to
be run on a single GPU; the complete grid used for the paper is

    python3 train_cifar.py --grid main        # optimiser comparison, batch size 512
    python3 train_cifar.py --grid batch       # batch-size ablation
    python3 train_cifar.py --grid scope       # shrinkage-scope ablation
    python3 train_cifar.py --grid diagnostics # per-layer diagnostics, one run

Each invocation writes a new run folder under ../results/runs/<grid>_<timestamp>/ with
results.csv and run_meta.json (grid, seeds, device, torch version, command). Re-runs
never overwrite earlier folders unless you pass --out pointing at an existing CSV and
--append.

Only the optimiser changes between arms.  Backbone, preprocessing, epoch count, batch
size, input-noise injection, evaluation procedure and seed schedule are held fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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

DEFAULT_RUN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "results", "runs")


def resolve_run_dir(grid, out):
    """One folder per invocation. --out may be a CSV path or a directory."""
    if out is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(DEFAULT_RUN_ROOT, f"{grid}_{stamp}")
    else:
        out_abs = os.path.abspath(out)
        if str(out).lower().endswith(".csv"):
            run_dir = os.path.dirname(out_abs) or "."
        else:
            run_dir = out_abs
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_run_meta(run_dir, args, csv_path):
    meta = dict(
        grid=args.grid,
        epochs=args.epochs,
        seeds=args.seeds,
        csv=os.path.abspath(csv_path),
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        device=DEVICE,
        torch_version=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        command=" ".join(sys.argv),
        cwd=os.getcwd(),
        note="Only the optimizer changes across arms; paper tables use final-epoch metrics.",
    )
    path = os.path.join(run_dir, "run_meta.json")
    with open(path, "w") as g:
        json.dump(meta, g, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main",
                    choices=["main", "batch", "scope", "diagnostics"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--out",
        default=None,
        help="CSV path or run directory. Default: ../results/runs/<grid>_<timestamp>/",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to the chosen CSV. Default is a fresh run folder (old runs kept).",
    )
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.grid, args.out)
    if args.out and str(args.out).lower().endswith(".csv"):
        csv_path = os.path.abspath(args.out)
    else:
        csv_path = os.path.join(run_dir, "results.csv")
    meta_path = write_run_meta(run_dir, args, csv_path)
    print(f"Run dir : {os.path.abspath(run_dir)}")
    print(f"CSV     : {csv_path}")
    print(f"Meta    : {os.path.abspath(meta_path)}")

    write_header = args.append and os.path.exists(csv_path)
    mode = "a" if args.append else "w"
    with open(csv_path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not write_header:
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
            diag_path = os.path.join(run_dir, "diagnostics.json")
            with open(diag_path, "w") as g:
                json.dump(rows, g)
            print(f"  wrote {len(rows)} diagnostic records to {diag_path}")

    print(f"Finished grid={args.grid}. Results kept in {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()
