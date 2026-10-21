@echo off
setlocal
rem =====================================================================
rem Phase D - THE ordered re-run of everything with the FINAL configuration.
rem Run only after Phases B and C are frozen (estimator + baseline settings).
rem Order mirrors TODO section D (test first, tables, figures, statistics).
rem
rem Usage:
rem   scripts\phaseD.bat                 full (paper numbers)
rem   scripts\phaseD.bat smoke           reduced sizes, pipeline check
rem   scripts\phaseD.bat full "C:\path\to\python.exe"
rem   set SKIP_GNN=1  -  skip the PyG graph pilots (no torch_geometric)
rem
rem Every command preserves its exit status; the script fails at the first
rem failing step.
rem =====================================================================
set PY=C:\programs\anaconda3\envs\pth\python.exe
set MODE=%1
if "%MODE%"=="" set MODE=full
if not "%~2"=="" set PY=%~2

set SMK=
set QS=--seeds 5 --steps 800
set P4=--ps 10,20,50,100,200,400,500 --tau 1.0 --steps 800 --seeds 5
set SN=
if /i "%MODE%"=="smoke" (
    set SMK=--smoke
    set QS=--quick
    set P4=--smoke
    set SN=--smoke
)

pushd "%~dp0..\code"

echo [0] test_sr_adam.py  (mandatory after the sr_adam var_window edit)
"%PY%" test_sr_adam.py
if errorlevel 1 goto fail

echo [1a] verify_theory.py
"%PY%" verify_theory.py
if errorlevel 1 goto fail
echo [1b] verify_theory2.py
"%PY%" verify_theory2.py
if errorlevel 1 goto fail

echo [2a] simulation_study.py
"%PY%" simulation_study.py
if errorlevel 1 goto fail
echo [2b] simulation_study3.py
"%PY%" simulation_study3.py
if errorlevel 1 goto fail

echo [3] simulation_noise.py  (Table tab:opt_noise, tuned arms + timing)
"%PY%" -u simulation_noise.py %SN% --tune --timing --gate-log
if errorlevel 1 goto fail

echo [4] simulation_batch.py  (Table tab:batch)
"%PY%" simulation_batch.py
if errorlevel 1 goto fail

echo [5] path4_dim_sweep.py  (tab:dim_sweep + fig; unified config = item a8)
"%PY%" -u path4_dim_sweep.py %P4% --tune --timing
if errorlevel 1 goto fail

echo [6] paper_tables.py  (tab_regression / tab_classification)
"%PY%" -u paper_tables.py %QS% --tune --timing --gate-log
if errorlevel 1 goto fail

echo [7] noise_sweep.py  (tab_noise + fig_noise_sweep)
"%PY%" -u noise_sweep.py %QS% --tune --early-stop 200 --timing
if errorlevel 1 goto fail

if "%SKIP_GNN%"=="1" (
    echo [8] gnn pilots SKIPPED by SKIP_GNN=1
    goto skip_gnn
)
rem gnn_pilot prints a Greek delta; force UTF-8 so the runs do not die on print
set PYTHONIOENCODING=utf-8
echo [8a] gnn_pilot.py
"%PY%" -u gnn_pilot.py --seeds 5 --tune
if errorlevel 1 goto fail
echo [8b] gnn_pilot_planetoid.py
"%PY%" -u gnn_pilot_planetoid.py --seeds 3 --tune
if errorlevel 1 goto fail
echo [8c] refresh GCN rows of paper_tables.json + tab_classification.tex
"%PY%" -u _refresh_gcn_rows.py
if errorlevel 1 goto fail
:skip_gnn

echo [9a] make_figures.py
"%PY%" make_figures.py
if errorlevel 1 goto fail
echo [9b] make_paper_barfigs.py  (data-driven since Phase A)
"%PY%" make_paper_barfigs.py
if errorlevel 1 goto fail

echo [10] stats_tests.py  (Wilcoxon + Holm on the FRESH json)
"%PY%" stats_tests.py
if errorlevel 1 goto fail

popd
echo.
echo PHASE D OK - see out\*.json, paper\tab_*.tex, figs\*.pdf, out\stats_tests.md
echo Next: scripts\phaseE.bat  (recompile the manuscript and grep torerun)
exit /b 0

:fail
echo.
echo PHASE D FAILED at the step shown above; outputs before it are complete.
popd
exit /b 1