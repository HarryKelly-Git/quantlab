@echo off
REM QuantLab PAPER runner with automatic restart. Paper only - there is no live mode.
REM
REM Start it and leave it: double-click, or from a terminal
REM     scripts\run_paper.cmd
REM Stop it: close this window, or from another terminal
REM     .venv\Scripts\python -m quantlab.cli paper stop
REM (a clean stop is recorded; this wrapper then exits instead of restarting).
REM
REM The runner is single-instance: a second copy refuses to start, so running this twice is safe.
setlocal
cd /d "%~dp0.."

REM The runner refuses to start unless PAPER is declared explicitly.
set TRADING_MODE=PAPER
set LIVE_TRADING=false

if not exist "var\logs" mkdir "var\logs"

:loop
echo [%date% %time%] starting QuantLab paper runner >> var\logs\runner-supervisor.log
.venv\Scripts\python.exe -m quantlab.cli paper start
set RC=%ERRORLEVEL%
echo [%date% %time%] runner exited with code %RC% >> var\logs\runner-supervisor.log

REM 0 = clean stop (quantlab paper stop / Ctrl+C): do not restart.
if "%RC%"=="0" (
  echo [%date% %time%] clean stop: supervisor exiting >> var\logs\runner-supervisor.log
  goto :eof
)

echo [%date% %time%] restarting in 60s >> var\logs\runner-supervisor.log
timeout /t 60 /nobreak > nul
goto loop
