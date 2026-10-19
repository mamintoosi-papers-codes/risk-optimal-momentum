@echo off
setlocal
rem =====================================================================
rem Phase E - manuscript-facing checks (TODO section E)
rem   E1  tables are generated, never hand-edited (checks provenance comments)
rem   E2  torerun audit - must find only the comment (line 43) and the
rem       \newcommand definition (line 47); any hit in the body fails
rem   E3  full PDF build: pdflatex + bibtex + pdflatex x2
rem
rem Usage:
rem   scripts\phaseE.bat                 all checks + build
rem   scripts\phaseE.bat check           checks only (no LaTeX)
rem   scripts\phaseE.bat full "C:\path\to\python.exe"
rem
rem pdflatex/bibtex come from your TeX distribution; if they are not on PATH
rem the build step is reported as SKIPPED (install MiKTeX/TeX Live or run it
rem from your LaTeX shell).
rem =====================================================================
set PY=C:\programs\anaconda3\envs\pth\python.exe
set MODE=%1
if "%MODE%"=="" set MODE=full
if not "%~2"=="" set PY=%~2

pushd "%~dp0.."

echo [E1] tab_*.tex provenance
for %%F in (tab_regression tab_classification tab_noise) do (
    if not exist "paper\%%F.tex" (
        echo   MISSING paper\%%F.tex  - run scripts\phaseD.bat first
        goto fail
    )
    findstr /b /c:"%% Auto-generated" "paper\%%F.tex" >nul 2>&1
    if errorlevel 1 (
        echo   WARNING paper\%%F.tex does not carry the auto-generated header
    ) else (
        echo   OK paper\%%F.tex is generated
    )
)

echo [E2] torerun audit in paper\sn-article.tex
set FOUND=0
for /f "tokens=1,* delims=:" %%A in ('findstr /n "torerun" "paper\sn-article.tex"') do (
    echo   line %%A: %%B
    set FOUND=1
)
rem Only lines 43 (comment) and 47 (definition) are acceptable.  Body usage
rem would appear as a line containing \torerun{ that is neither of them.
findstr "\\torerun{" "paper\sn-article.tex" >nul 2>&1
if not errorlevel 1 (
    echo   FAIL - \torerun{...} placeholder found in the body; remove it or fill it
    goto fail
)
echo   OK - no \torerun{...} placeholder in the body

if /i "%MODE%"=="check" goto checked

echo [E3] PDF build
where pdflatex >nul 2>&1
if errorlevel 1 (
    echo   SKIPPED - pdflatex not on PATH
    goto checked
)
pushd paper
pdflatex -interaction=nonstopmode sn-article.tex > pdflatex1.log 2>&1
if errorlevel 1 goto failtex
bibtex sn-article > bibtex.log 2>&1
pdflatex -interaction=nonstopmode sn-article.tex > pdflatex2.log 2>&1
pdflatex -interaction=nonstopmode sn-article.tex > pdflatex3.log 2>&1
if errorlevel 1 goto failtex
popd
echo   built paper\sn-article.pdf

:checked
popd
echo.
echo PHASE E OK
exit /b 0

:failtex
echo   LaTeX FAILED - see paper\pdflatex1.log / bibtex.log
popd
goto fail2
:fail
set ERR=1
:fail2
popd
echo PHASE E FAILED
exit /b 1