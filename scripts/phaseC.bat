@echo off
setlocal
rem =====================================================================
rem Phase C - fair-baseline infrastructure RUNS (item a7 + a8 + complexity c)
rem   * --tune           identical lr (and optional beta1 / wd) grid per optimiser
rem   * --early-stop N   regularisation control arm
rem   * --timing         per-step wall-clock column
rem   * --gate-log       SR-Adam gate-open frequencies (replaces the dropped CIFAR run)
rem
rem Usage:
rem   scripts\phaseC.bat smoke     default: --quick/--smoke pilots, seconds-minutes
rem   scripts\phaseC.bat full      full grid runs (writes the paper tables too)
rem   scripts\phaseC.bat full "C:\path\to\python.exe"
rem
rem NOTE: full mode overwrites out/paper_tables.json, out/noise_sweep.json,
rem out/simulation_noise.json, out/path4_dim_sweep.json and paper/tab_*.tex.
rem Run it only after the estimator defaults are frozen (TODO phase order).
rem =====================================================================
set PY=C:\programs\anaconda3\envs\pth\python.exe
set MODE=%1
if "%MODE%"=="" set MODE=smoke
if not "%~2"=="" set PY=%~2

pushd "%~dp0..\code"

if /i "%MODE%"=="full" goto full

echo --- Phase C smoke (pipeline check; numbers are NOT paper numbers) ---
"%PY%" paper_tables.py --quick --tune --timing
if errorlevel 1 goto fail
"%PY%" noise_sweep.py --quick --tune --timing
if errorlevel 1 goto fail
"%PY%" simulation_noise.py --smoke --tune --timing
if errorlevel 1 goto fail
"%PY%" path4_dim_sweep.py --smoke --tune --timing
if errorlevel 1 goto fail
goto done

:full
echo --- Phase C full (tuned baselines, controls, timings) ---
"%PY%" -u paper_tables.py --seeds 5 --steps 800 --tune --timing --gate-log
if errorlevel 1 goto fail
"%PY%" -u noise_sweep.py --seeds 5 --steps 800 --tune --early-stop 200 --timing
if errorlevel 1 goto fail
"%PY%" -u simulation_noise.py --tune --timing --gate-log --early-stop 200
if errorlevel 1 goto fail
rem item a8: ONE configuration shared with Table 5 (p=400 exists in both)
"%PY%" -u path4_dim_sweep.py --ps 10,20,50,100,200,400,500 --tau 1.0 ^
    --steps 800 --seeds 5 --tune --timing
if errorlevel 1 goto fail

:done
popd
echo.
echo PHASE C OK - see out\*_tune.json and the timing and gate columns in out\*.json
exit /b 0

:fail
echo.
echo PHASE C FAILED  (see message above)
popd
exit /b 1