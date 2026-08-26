@echo off
cd /d "%~dp0.."
echo ============================================================
echo RELENSING / FORWARD VALIDATION DEMO
echo ============================================================
echo.
".venv\Scripts\python.exe" "demo\run_relensing_demo.py"
if exist "demo_outputs\relensing\Relensing_Forward_Validation_Demo.png" start "" "demo_outputs\relensing\Relensing_Forward_Validation_Demo.png"
echo.
pause
