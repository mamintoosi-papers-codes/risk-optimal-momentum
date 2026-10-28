@echo off
setlocal
rem =====================================================================
rem Phase A - statistics on the STORED results (no new simulation)
rem Covers diagnostic b3 + TODO items A1-A4.
rem
rem Usage:
rem   scripts\phaseA.bat                 full (rebuild figures from out/*.json)
rem   scripts\phaseA.bat smoke           numbers only, no PDFs
rem   scripts\phaseA.bat full "C:\path\to\python.exe"   other interpreter
rem =====================================================================
set PY=C:\programs\anaconda3\envs\pth\python.exe
set MODE=%1
if "%MODE%"=="" set MODE=full
if not "%~2"=="" set PY=%~2

set BARFIGARGS=
if /i "%MODE%"=="smoke" set BARFIGARGS=--check

pushd "%~dp0..\code"

echo [A1/A4] make_paper_barfigs.py %BARFIGARGS%
"%PY%" make_paper_barfigs.py %BARFIGARGS%
if errorlevel 1 goto fail

echo [A2/A3/A4] stats_tests.py
"%PY%" stats_tests.py
if errorlevel 1 goto fail

popd
echo.
echo PHASE A OK - see out\winrate_summary.json and out\stats_tests.md
exit /b 0

:fail
echo.
echo PHASE A FAILED  (see message above)
popd
exit /b 1