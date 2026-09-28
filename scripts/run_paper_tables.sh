#!/usr/bin/env bash
# Paper tables: regression + classification (logistic / MLP / GCN).
# Linux + GPU (or CPU) from repo root:
#   bash scripts/run_paper_tables.sh          # full
#   bash scripts/run_paper_tables.sh quick    # short smoke
#   bash scripts/run_paper_tables.sh seeds5   # 5 seeds, longer
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/code"
PY="${PYTHON:-python}"
MODE="${1:-full}"

echo "Python: $($PY -c 'import sys; print(sys.executable)')"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# CIFAR not required; MNIST/GNN data under ../data if present
case "$MODE" in
  quick)
    $PY -u paper_tables.py --quick
    ;;
  seeds5)
    $PY -u paper_tables.py --seeds 5 --steps 800
    ;;
  full)
    $PY -u paper_tables.py --seeds 5 --steps 800
    ;;
  *)
    echo "Usage: $0 [quick|seeds5|full]" >&2
    exit 1
    ;;
esac

echo ""
echo "LaTeX tables:"
echo "  $ROOT/paper/tab_regression.tex"
echo "  $ROOT/paper/tab_classification.tex"
echo "JSON: $ROOT/out/paper_tables.json"
