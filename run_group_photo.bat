@echo off
REM ============================================================
REM  Team Photo Player ID (group photos) - START THE APP
REM  Double-click this file each time you want to use the app.
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] The app is not set up yet.
    echo         Please double-click  setup.bat  first.
    echo.
    pause
    exit /b 1
)

echo Starting Team Photo Player Identification...
echo (The first start may take a minute while the face model downloads.)
".venv\Scripts\python.exe" -m app.main
if errorlevel 1 (
    echo.
    echo [ERROR] The app closed unexpectedly. See messages above.
    pause
)
