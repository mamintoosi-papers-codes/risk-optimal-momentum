@echo off
setlocal
rem =====================================================================
rem Phase B - NEW experiment programs (items a3, a4, a5, b1, b2)
rem Order matters only for reporting: unit test first, then the MC scripts.
rem
rem Usage:
rem   scripts\phaseB.bat            full runs (minutes on CPU)
rem   scripts\phaseB.bat smoke      tiny grids, seconds each (pipeline test)
rem   scripts\phaseB.bat full "C:\path\to\python.exe"
rem =====================================================================
set PY=C:\programs\anaconda3\envs\pth\python.exe
set MODE=%1
if "%MODE%"=="" set MODE=full
if not "%~2"=="" set PY=%~2

set SM=
if /i "%MODE%"=="smoke" set SM=--smoke

pushd "%~dp0..\code"

rem MUST pass before anything else: sr_adam.py gained var_window="matched"
echo [check] test_sr_adam.py
"%PY%" test_sr_adam.py
if errorlevel 1 goto fail

rem B1 / a4: positive-part dominance under the plug-in scale spread
echo [B1/a4] verify_plugin_pospart.py %SM%
"%PY%" verify_plugin_pospart.py %SM%
if errorlevel 1 goto fail

rem B2 / a3: covariance of coordinatewise variance estimates under a shared W
echo [B2/a3] sim_a5_covariance.py %SM%
"%PY%" sim_a5_covariance.py %SM%
if errorlevel 1 goto fail

rem B3 / a5: sigma_hat^2 / sigma_true^2 along real runs (both windows)
echo [B3/a5] log_variance_ratio.py %SM%
"%PY%" log_variance_ratio.py %SM%
if errorlevel 1 goto fail

rem B4 / b2: Wbar(mu) sweep + heavy-tail risk by signal level
echo [B4/b2] sim_window_signal.py %SM%
"%PY%" sim_window_signal.py %SM%
if errorlevel 1 goto fail

rem B5 / b1: numeric constants of the convergence bound
echo [B5/b1] conv_constants.py %SM%
"%PY%" conv_constants.py %SM%
if errorlevel 1 goto fail

popd
echo.
echo PHASE B OK - see out\verify_plugin_pospart.json, out\sim_a5_covariance.json,
echo                out\log_variance_ratio.json, out\sim_window_signal.json,
echo                out\conv_constants.json
exit /b 0

:fail
echo.
echo PHASE B FAILED  (see message above)
popd
exit /b 1