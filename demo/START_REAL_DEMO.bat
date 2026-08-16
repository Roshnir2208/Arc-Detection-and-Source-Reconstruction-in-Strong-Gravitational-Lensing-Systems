@echo off
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo\run_live_demo.py --sample real
) else (
  python demo\run_live_demo.py --sample real
)
pause
