@echo off
REM QuantLab dashboard (read-only, localhost only): http://127.0.0.1:8765/live
REM Double-click to start; close this window to stop. Safe to run alongside the paper runner.
cd /d "%~dp0.."
.venv\Scripts\python.exe -m quantlab.cli dashboard
