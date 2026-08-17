@echo off
REM ============================================================
REM  QR Photo Sorter (optimised) - START THE APP
REM  Double-click this file each time you want to sort photos.
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

echo Starting QR Photo Sorter...
echo (The first start downloads the face model once - needs internet.)
".venv\Scripts\python.exe" -m sorter_app.main
if errorlevel 1 (
    echo.
    echo [ERROR] The app closed unexpectedly. See messages above.
    pause
)
