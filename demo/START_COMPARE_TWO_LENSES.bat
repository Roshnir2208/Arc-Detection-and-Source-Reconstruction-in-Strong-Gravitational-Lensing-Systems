@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Local demo environment not found.
  echo Run demo\SETUP_LOCAL_DEMO_ENV.bat once before the presentation.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" demo\run_live_demo.py --compare
if exist "demo_outputs\synthetic_comparison\lens_00695_vs_lens_00932_comparison.png" (
  start "" "demo_outputs\synthetic_comparison\lens_00695_vs_lens_00932_comparison.png"
)
pause
