#!/usr/bin/env bash
# Run the CIFAR experimental grids for the revision paper.
# Intended for a Linux machine with a CUDA GPU (e.g. RTX 3090).
#
# Usage (from repo root):
#   bash scripts/run_cifar_grids.sh              # full paper grids
#   bash scripts/run_cifar_grids.sh smoke        # tiny smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/code"

PY="${PYTHON:-python}"
OUT="${OUT:-results.csv}"
MODE="${1:-full}"

echo "Using Python: $($PY -c 'import sys; print(sys.executable)')"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ "$MODE" == "smoke" ]]; then
  echo "=== smoke: 1 seed, 2 epochs, diagnostics only ==="
  $PY train_cifar.py --grid diagnostics --epochs 2 --seeds 1 --out "$OUT"
  exit 0
fi

echo "=== diagnostics (gate / peff per layer) ==="
$PY train_cifar.py --grid diagnostics --epochs 20 --seeds 1 --out "$OUT"

echo "=== component ablation (scope) ==="
$PY train_cifar.py --grid scope --epochs 20 --seeds 3 --out "$OUT"

echo "=== batch-size ablation ==="
$PY train_cifar.py --grid batch --epochs 20 --seeds 3 --out "$OUT"

echo "=== main optimiser comparison ==="
$PY train_cifar.py --grid main --epochs 20 --seeds 5 --out "$OUT"

echo "Done. CSV: $ROOT/code/$OUT"
echo "Aggregate final-epoch mean/std, then fill \\torerun{} in paper/sn-article.tex"
