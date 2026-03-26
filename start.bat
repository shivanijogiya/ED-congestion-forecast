@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM ED Congestion Forecasting — One-command local startup (Windows)
REM Usage: Double-click start.bat  OR  run it from Command Prompt
REM ─────────────────────────────────────────────────────────────────────────────

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║   ED Congestion Forecasting System                      ║
echo ║   Graph-Aware Deep Learning · Real-Time · 6 Hospitals   ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM ── Check Python ──────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+ from https://python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version') do echo [OK] %%v found

REM ── Install dependencies ──────────────────────────────────────────────────────
echo.
echo [INSTALLING] Installing dependencies...
pip install --quiet fastapi uvicorn pydantic pydantic-settings httpx pytest pytest-asyncio numpy scipy
echo [OK] Dependencies ready

REM ── Run tests ─────────────────────────────────────────────────────────────────
echo.
echo [TESTING] Running tests...
python -m pytest tests/unit/test_simulation.py tests/integration/test_api.py -q --tb=short
if errorlevel 1 (
    echo [WARNING] Some tests failed. Continuing anyway...
)
echo [OK] Tests complete

REM ── Open browser and start server ─────────────────────────────────────────────
echo.
echo [STARTING] Launching server...
echo.
echo ┌────────────────────────────────────────────────────────┐
echo │  Dashboard  ^>  http://localhost:8000                  │
echo │  API Docs   ^>  http://localhost:8000/docs             │
echo │  Health     ^>  http://localhost:8000/health           │
echo │                                                        │
echo │  Press Ctrl+C to stop                                  │
echo └────────────────────────────────────────────────────────┘
echo.

REM Open browser after 2 seconds
start "" timeout /t 2 >nul && start "" "http://localhost:8000"

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
pause
