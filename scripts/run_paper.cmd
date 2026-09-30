@echo off
REM QuantLab PAPER runner with automatic restart. Paper only - there is no live mode.
REM
REM Start it and leave it: double-click, or from a terminal
REM     scripts\run_paper.cmd
REM Stop it: close this window, or from another terminal
REM     .venv\Scripts\python -m quantlab.cli paper stop
REM (a clean stop is recorded; this wrapper then exits instead of restarting).
REM
REM The runner is single-instance: a second copy refuses to start. This wrapper also checks for a live
REM runner process first (scripts\runner_alive.py) and exits if one exists, so a second supervisor --
REM the Windows Startup entry plus a manual double-click -- does not retry a refused start forever.
REM Started automatically at logon by the "QuantLab paper runner" shortcut in the user Startup folder.
setlocal
cd /d "%~dp0.."

REM The runner refuses to start unless PAPER is declared explicitly.
set TRADING_MODE=PAPER
set LIVE_TRADING=false

if not exist "var\logs" mkdir "var\logs"

:loop
REM A live runner that is not ours (our own runner has exited by the time we loop): leave it alone.
.venv\Scripts\python.exe scripts\runner_alive.py >> var\logs\runner-supervisor.log 2>&1
if "%ERRORLEVEL%"=="0" (
  echo [%date% %time%] another paper runner is alive: this supervisor exits >> var\logs\runner-supervisor.log
  goto :eof
)
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
REM ping, not timeout: timeout needs console input and returns at once when the supervisor
REM has none (Startup launch), which turned this 60s back-off into a ~3s preflight loop.
ping -n 61 127.0.0.1 > nul
goto loop
