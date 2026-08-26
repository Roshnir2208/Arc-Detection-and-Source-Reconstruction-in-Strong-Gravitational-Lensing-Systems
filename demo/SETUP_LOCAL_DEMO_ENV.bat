@echo off
cd /d "%~dp0.."
set SOURCE_VENV=D:\OneDrive Data\Masters Study\Summer\Masters Project\Codex\Project\outputs\lensing_pipeline_starter\.venv
set TARGET_VENV=%CD%\.venv

if exist "%TARGET_VENV%\Scripts\python.exe" (
  echo Local demo environment already exists:
  echo %TARGET_VENV%
  pause
  exit /b 0
)

if not exist "%SOURCE_VENV%\Scripts\python.exe" (
  echo Source project environment was not found:
  echo %SOURCE_VENV%
  echo Please create a local .venv manually or run the demo with the known project environment.
  pause
  exit /b 1
)

echo Copying known working Python environment into the demo folder...
echo This may take a few minutes the first time.
robocopy "%SOURCE_VENV%" "%TARGET_VENV%" /E /XD __pycache__ .pytest_cache /XF *.pyc
if %ERRORLEVEL% LEQ 7 (
  echo.
  echo Local demo environment ready:
  echo %TARGET_VENV%
  pause
  exit /b 0
)

echo Environment copy failed.
pause
exit /b 1
