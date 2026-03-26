@echo off
setlocal EnableDelayedExpansion
:: ==============================================================================
:: ED Congestion Forecasting - Auto Setup (Windows)
:: Tested on: Windows 10, Windows 11
:: Double-click this file OR run from Command Prompt / PowerShell
:: ==============================================================================

title ED Congestion Forecasting - Setup

cls
echo.
echo ==============================================
echo   ED Congestion Forecasting - Auto Setup
echo   Graph-Aware Deep Learning ^| 6 Hospitals
echo ==============================================
echo.
echo Steps:
echo   1. Check Python 3.10+
echo   2. Create virtual environment
echo   3. Install dependencies
echo   4. Run tests
echo   5. Launch dashboard at http://localhost:8000
echo.
echo Press any key to continue or close window to cancel...
pause >nul

:: ==============================================================================
:: STEP 1 - Check Python
:: ==============================================================================
echo.
echo --- Step 1/5 - Checking Python ---

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python not found!
    echo.
    echo Install Python 3.10+ from:
    echo https://www.python.org/downloads/windows/
    echo.
    echo IMPORTANT: During install, check "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER% found

python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip not found. Reinstall Python and check "Add to PATH"
    pause
    exit /b 1
)
echo [OK] pip available

python -m venv --help >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] venv not available. Reinstall Python 3.10+
    pause
    exit /b 1
)
echo [OK] venv available

:: ==============================================================================
:: STEP 2 - Virtual Environment
:: ==============================================================================
echo.
echo --- Step 2/5 - Virtual Environment ---

if exist ".venv\" (
    echo [WARN] Virtual environment already exists - reusing .venv\
) else (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause & exit /b 1
    )
    echo [OK] Virtual environment created in .venv\
)

call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    echo Try running as Administrator or check antivirus settings
    pause & exit /b 1
)
echo [OK] Virtual environment activated

python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded

:: ==============================================================================
:: STEP 3 - Install Dependencies
:: ==============================================================================
echo.
echo --- Step 3/5 - Installing Dependencies ---

echo [INFO] Installing core packages...
pip install --quiet "fastapi>=0.109" "uvicorn[standard]>=0.27" "pydantic>=2.5" "pydantic-settings>=2.1" "numpy>=1.26" "scipy>=1.11" "PyYAML>=6.0" "python-dotenv>=1.0" "httpx>=0.26"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install core packages
    echo Check your internet connection and try again
    pause & exit /b 1
)
echo [OK] Core packages installed

echo [INFO] Installing test packages...
pip install --quiet "pytest>=7.4" "pytest-asyncio>=0.23"
if %errorlevel% neq 0 (
    echo [WARN] Test packages failed - continuing anyway
) else (
    echo [OK] Test packages installed
)

python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] PyTorch not installed - optional, only needed for model training
    echo        To install later: pip install torch torch-geometric
) else (
    echo [OK] PyTorch found - model training available
)

:: ==============================================================================
:: STEP 4 - Run Tests
:: ==============================================================================
echo.
echo --- Step 4/5 - Running Tests ---

echo [INFO] Running simulation tests...
python -m pytest tests/unit/test_simulation.py -q --tb=short
if %errorlevel% neq 0 (
    echo [WARN] Some simulation tests failed - continuing anyway
) else (
    echo [OK] Simulation tests passed
)

echo [INFO] Running API tests...
python -m pytest tests/integration/test_api.py -q --tb=short
if %errorlevel% neq 0 (
    echo [WARN] Some API tests failed - continuing anyway
) else (
    echo [OK] API tests passed
)

:: ==============================================================================
:: STEP 5 - Launch
:: ==============================================================================
echo.
echo --- Step 5/5 - Launching Dashboard ---
echo.
echo [OK] Setup complete!
echo.
echo ==============================================
echo   Dashboard  -^>  http://localhost:8000
echo   API Docs   -^>  http://localhost:8000/docs
echo   Health     -^>  http://localhost:8000/health
echo   Press Ctrl+C to stop
echo ==============================================
echo.

:: Open browser after 3 seconds in background
start "" cmd /c "timeout /t 3 >nul && start http://localhost:8000"

:: Start server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

pause
