#!/usr/bin/env bash
# Phase D - ordered re-run of every paper artifact (Linux/macOS server box).
# Mirrors scripts/phaseD.bat.  First failure aborts with its exit status.
#
# Usage:
#   bash scripts/phaseD.sh                # full (paper numbers)
#   bash scripts/phaseD.sh smoke          # reduced sizes, pipeline check
#   PY=/path/to/python bash scripts/phaseD.sh   # other interpreter
#   SKIP_GNN=1 bash scripts/phaseD.sh     # no torch_geometric installed
set -euo pipefail

PY="${PY:-python}"
MODE="${1:-full}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/code"

QS=(--seeds 5 --steps 800)
P4=(--ps 10,20,50,100,200,400,500 --tau 1.0 --steps 800 --seeds 5)
SN=()
if [[ "$MODE" == "smoke" ]]; then
  QS=(--quick)
  P4=(--smoke)
  SN=(--smoke)
fi

echo "[0] test_sr_adam.py"
"$PY" test_sr_adam.py

echo "[1] verify_theory.py / verify_theory2.py"
"$PY" verify_theory.py
"$PY" verify_theory2.py

echo "[2] simulation_study.py / simulation_study3.py"
"$PY" simulation_study.py
"$PY" simulation_study3.py

echo "[3] simulation_noise.py (Table tab:opt_noise)"
"$PY" -u simulation_noise.py ${SN[@]+"${SN[@]}"} --tune --timing --gate-log

echo "[4] simulation_batch.py (Table tab:batch)"
"$PY" simulation_batch.py

echo "[5] path4_dim_sweep.py (tab:dim_sweep, unified config = a8)"
"$PY" -u path4_dim_sweep.py "${P4[@]}" --tune --timing

echo "[6] paper_tables.py (tab_regression / tab_classification)"
"$PY" -u paper_tables.py "${QS[@]}" --tune --timing --gate-log

echo "[7] noise_sweep.py (tab_noise + fig)"
"$PY" -u noise_sweep.py "${QS[@]}" --tune --early-stop 200 --timing

if [[ "${SKIP_GNN:-0}" != "1" ]]; then
  echo "[8] gnn pilots"
  "$PY" -u gnn_pilot.py --seeds 5
  "$PY" -u gnn_pilot_planetoid.py --seeds 3
else
  echo "[8] gnn pilots SKIPPED (SKIP_GNN=1)"
fi

echo "[9] figures"
"$PY" make_figures.py
"$PY" make_paper_barfigs.py

echo "[10] stats_tests.py (Wilcoxon + Holm on the fresh json)"
"$PY" stats_tests.py

echo
echo "PHASE D OK -> out/*.json, paper/tab_*.tex, figs/*.pdf, out/stats_tests.md"
echo "Next: latex build + torerun audit (scripts/phaseE.bat on Windows)"