#!/usr/bin/env bash
# Grouping ablation on MNIST + CIFAR-10 (block / filter / channel / channel+kmin).
# Run on a Linux GPU machine from the repository root:
#   bash scripts/run_grouping_grid.sh          # full grid
#   bash scripts/run_grouping_grid.sh smoke    # tiny CPU/GPU smoke
#   bash scripts/run_grouping_grid.sh mnist    # MNIST only
#   bash scripts/run_grouping_grid.sh cifar10  # CIFAR-10 only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/code"
PY="${PYTHON:-python}"
MODE="${1:-full}"

echo "Python: $($PY -c 'import sys; print(sys.executable)')"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ensure data (gitignored)
$PY download_cifar.py --root ./data || true

case "$MODE" in
  smoke)
    $PY -u train_grouping.py --smoke
    ;;
  mnist)
    $PY -u train_grouping.py --dataset mnist --seeds 3 --epochs 5
    ;;
  cifar10)
    $PY -u train_grouping.py --dataset cifar10 --seeds 3 --epochs 5
    ;;
  full)
    $PY -u train_grouping.py --dataset both --seeds 3 --epochs 5
    ;;
  *)
    echo "Usage: $0 [smoke|mnist|cifar10|full]" >&2
    exit 1
    ;;
esac

echo ""
echo "Results under $ROOT/results/runs/grouping_*"
echo "Groupings: sradam_block | sradam_filter | sradam_channel | sradam_channel_kmin | adamw"
