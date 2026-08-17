@echo off
REM ============================================================
REM  QR Photo Sorter (optimised) - ONE-TIME SETUP
REM  Double-click this file once before using the app.
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo ==========================================================
echo   Setting up QR Photo Sorter (optimised)
echo   This runs ONCE and needs an internet connection.
echo ==========================================================
echo.

REM --- 1. Find a Python 3.12 interpreter --------------------
REM This app needs Python 3.12 (newer versions like 3.13/3.14 have no
REM prebuilt libraries yet). Prefer the "py -3.12" launcher so it uses 3.12
REM even when a different Python is the system default.
set "PY="
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set "PY=py -3.12"
) else (
    REM No py launcher 3.12 - fall back to "python" only if it IS 3.12.
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
    echo %PYVER% | findstr /b "3.12." >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo [ERROR] Python 3.12 was not found on this PC.
    echo.
    echo   This app requires Python 3.12 specifically.
    echo   Please install Python 3.12.x ^(64-bit^) from:
    echo      https://www.python.org/downloads/release/python-31210/
    echo   IMPORTANT: on the first install screen, TICK the box
    echo   "Add python.exe to PATH", then run this setup again.
    echo.
    pause
    exit /b 1
)

echo Using Python:
%PY% --version
echo.

REM --- 2. Create an isolated environment --------------------
if not exist ".venv\" (
    echo Creating local environment...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the environment. See messages above.
        pause
        exit /b 1
    )
)

REM --- 3. Install the required libraries ---------------------
echo Installing required libraries (this can take a few minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Library installation failed. Check your internet connection
    echo         and run this setup again.
    pause
    exit /b 1
)

REM --- 4. Verify the tag readers actually installed -----------
REM Both readers must be present. If either is missing the app still starts but
REM silently finds fewer students, which is the hardest kind of fault to notice
REM in the results - so check it here, while someone is watching.
echo.
echo Checking the name-tag readers...
".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0,'.'); from sorter.qr_reader import QRReader; e=QRReader().engine; print('   QR reader  :', e); sys.exit(0 if e.startswith('zxing') else 1)"
if errorlevel 1 (
    echo.
    echo [ERROR] The QR reader did not install correctly.
    echo         The app would find very few name tags. Please run setup again.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "from rapidocr_onnxruntime import RapidOCR; print('   Text reader: ready')"
if errorlevel 1 (
    echo.
    echo [ERROR] The text reader did not install correctly.
    echo         The app would miss students whose QR code will not scan.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo   Setup complete - both name-tag readers are working.
echo   You can now double-click  run_sorter.bat  to start.
echo   (The very first start downloads the face model once.)
echo ==========================================================
echo.
pause
