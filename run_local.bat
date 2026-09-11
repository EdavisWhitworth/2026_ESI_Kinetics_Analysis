@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        exit /b 1
    )
)

echo Installing required dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    exit /b 1
)

set "PYTHONPATH=%CD%\src"
echo Starting ESI Kinetics Analysis...
".venv\Scripts\python.exe" main.py
set "exit_code=%ERRORLEVEL%"

endlocal & exit /b %exit_code%
