@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Local demo environment not found.
  echo Run demo\SETUP_LOCAL_DEMO_ENV.bat once before the presentation.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\run_classification.py
".venv\Scripts\python.exe" scripts\run_cnn_ensemble.py
if exist "results_summary\morphology_classification_results.png" (
  start "" "results_summary\morphology_classification_results.png"
)
pause
