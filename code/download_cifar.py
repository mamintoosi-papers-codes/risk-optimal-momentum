"""Download CIFAR-10 / CIFAR-100 if missing. Safe to re-run."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from torchvision import datasets


def download(root: str) -> None:
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    for name, cls in (("CIFAR-10", datasets.CIFAR10), ("CIFAR-100", datasets.CIFAR100)):
        print(f"Checking {name} under {root_path} ...")
        cls(root=str(root_path), train=True, download=True)
        cls(root=str(root_path), train=False, download=True)
        print(f"  {name}: OK")
    print(f"Done. Data root: {root_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="CIFAR root (default: <repo>/code/data, same as train_cifar.py)",
    )
    download(ap.parse_args().root)
