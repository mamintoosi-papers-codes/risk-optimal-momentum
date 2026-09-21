#!/usr/bin/env bash
# Run the CIFAR experimental grids for the revision paper.
# Intended for a Linux machine with a CUDA GPU (e.g. RTX 3090).
#
# Usage (from repo root):
#   bash scripts/run_cifar_grids.sh download  # CIFAR-10/100 only
#   bash scripts/run_cifar_grids.sh smoke     # download + tiny train/eval
#   bash scripts/run_cifar_grids.sh main      # main optimiser comparison (both datasets)
#   bash scripts/run_cifar_grids.sh batch     # batch-size ablation
#   bash scripts/run_cifar_grids.sh scope     # component ablation
#   bash scripts/run_cifar_grids.sh diag      # per-layer diagnostics
#   bash scripts/run_cifar_grids.sh full      # all of the above, in order
#   bash scripts/run_cifar_grids.sh           # same as full
#
# Each grid writes a NEW folder:
#   code/../results/runs/<grid>_<YYYYmmdd-HHMMSS>/{results.csv,run_meta.json}
# Old runs are never deleted by this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/code"

PY="${PYTHON:-python}"
MODE="${1:-full}"
RUNS="$ROOT/results/runs"
mkdir -p "$RUNS"

echo "Using Python: $($PY -c 'import sys; print(sys.executable)')"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== download CIFAR-10 / CIFAR-100 (skip if already present) ==="
$PY download_cifar.py --root ./data

if [[ "$MODE" == "download" ]]; then
  echo "Download-only mode finished."
  exit 0
fi

run_grid() {
  local grid="$1"; shift
  echo ""
  echo "=== grid: $grid $* ==="
  $PY train_cifar.py --grid "$grid" "$@"
}

if [[ "$MODE" == "smoke" ]]; then
  run_grid diagnostics --epochs 2 --seeds 1
  echo "Smoke finished. See $RUNS/diagnostics_*"
  exit 0
fi

if [[ "$MODE" == "diag" || "$MODE" == "diagnostics" ]]; then
  run_grid diagnostics --epochs 20 --seeds 1
elif [[ "$MODE" == "scope" ]]; then
  run_grid scope --epochs 20 --seeds 3
elif [[ "$MODE" == "batch" ]]; then
  run_grid batch --epochs 20 --seeds 3
elif [[ "$MODE" == "main" ]]; then
  run_grid main --epochs 20 --seeds 5
elif [[ "$MODE" == "full" ]]; then
  # Priority for the paper: main first (needs Adam/AdamW baselines + both datasets)
  run_grid main --epochs 20 --seeds 5
  run_grid batch --epochs 20 --seeds 3
  run_grid scope --epochs 20 --seeds 3
  run_grid diagnostics --epochs 20 --seeds 1
else
  echo "Unknown mode: $MODE" >&2
  echo "Use: download | smoke | main | batch | scope | diag | full" >&2
  exit 1
fi

echo ""
echo "Done. Run folders:"
ls -1dt "$RUNS"/* 2>/dev/null | head -n 20 || true
echo "Aggregate final-epoch mean/std from each results.csv, then fill \\torerun{} in paper/sn-article.tex"
