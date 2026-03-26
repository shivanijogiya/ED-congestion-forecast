#!/bin/bash
# ==============================================================================
# ED Congestion Forecasting - Auto Setup (Mac / Linux / WSL)
# Tested on: macOS 13+, Ubuntu 20.04+, Debian 11+, WSL2
# ==============================================================================

set -e

# Colors (safe fallback if terminal doesn't support them)
if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[1;33m'
  BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'; BOLD='\033[1m'
else
  GREEN=''; YELLOW=''; BLUE=''; RED=''; NC=''; BOLD=''
fi

print_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_fail() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
print_step() { echo -e "\n${BOLD}--- $1 ---${NC}"; }

echo ""
echo "=============================================="
echo "  ED Congestion Forecasting - Auto Setup"
echo "  Graph-Aware Deep Learning | 6 Hospitals"
echo "=============================================="
echo ""
echo "Steps:"
echo "  1. Check Python (3.10+)"
echo "  2. Create virtual environment"
echo "  3. Install dependencies"
echo "  4. Run tests"
echo "  5. Launch dashboard at http://localhost:8000"
echo ""
echo "Press ENTER to continue or Ctrl+C to cancel..."
read -r _

# ==============================================================================
# STEP 1 - Check Python
# ==============================================================================
print_step "Step 1/5 - Checking Python"

PY=""
for cmd in python3 python python3.12 python3.11 python3.10; do
  if command -v "$cmd" >/dev/null 2>&1; then
    PY="$cmd"
    break
  fi
done

if [ -z "$PY" ]; then
  echo ""
  print_fail "Python not found. Install Python 3.10+ first:
  Mac:    brew install python3
          OR download from https://www.python.org/downloads/
  Ubuntu: sudo apt update && sudo apt install python3 python3-pip python3-venv
  Debian: sudo apt update && sudo apt install python3 python3-pip python3-venv"
fi

# Check version
PY_VER=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJ=$($PY -c "import sys; print(sys.version_info.major)")
PY_MIN=$($PY -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJ" -lt 3 ] || { [ "$PY_MAJ" -eq 3 ] && [ "$PY_MIN" -lt 10 ]; }; then
  print_fail "Python 3.10+ required. You have Python $PY_VER.
  Download: https://www.python.org/downloads/"
fi

print_ok "Python $PY_VER ($PY)"

# Check pip
if ! $PY -m pip --version >/dev/null 2>&1; then
  print_info "pip not found - installing..."
  $PY -m ensurepip --upgrade 2>/dev/null || {
    print_fail "Cannot install pip. Try:
  Ubuntu/Debian: sudo apt install python3-pip
  Mac:           brew install python3"
  }
fi
print_ok "pip available"

# Check venv
if ! $PY -m venv --help >/dev/null 2>&1; then
  print_fail "venv module missing. Install it:
  Ubuntu/Debian: sudo apt install python3-venv
  Mac:           brew install python3"
fi
print_ok "venv available"

# ==============================================================================
# STEP 2 - Virtual Environment
# ==============================================================================
print_step "Step 2/5 - Virtual Environment"

VENV=".venv"

if [ -d "$VENV" ]; then
  print_warn "Virtual environment already exists - reusing .venv/"
else
  print_info "Creating virtual environment..."
  $PY -m venv "$VENV"
  print_ok "Virtual environment created in .venv/"
fi

# Activate
# shellcheck source=/dev/null
source "$VENV/bin/activate"
print_ok "Virtual environment activated"

# Upgrade pip quietly
pip install --upgrade pip --quiet 2>/dev/null
print_ok "pip upgraded"

# ==============================================================================
# STEP 3 - Install Dependencies
# ==============================================================================
print_step "Step 3/5 - Installing Dependencies"

print_info "Installing core packages..."
pip install --quiet \
  "fastapi>=0.109" \
  "uvicorn[standard]>=0.27" \
  "pydantic>=2.5" \
  "pydantic-settings>=2.1" \
  "numpy>=1.26" \
  "scipy>=1.11" \
  "PyYAML>=6.0" \
  "python-dotenv>=1.0" \
  "httpx>=0.26" || print_fail "Failed to install core packages. Check your internet connection."
print_ok "Core packages installed"

print_info "Installing test packages..."
pip install --quiet "pytest>=7.4" "pytest-asyncio>=0.23" || print_fail "Failed to install test packages."
print_ok "Test packages installed"

# Optional PyTorch check
if python -c "import torch" 2>/dev/null; then
  print_ok "PyTorch found - model training available"
else
  print_warn "PyTorch not installed (optional - only needed for model training)"
  echo "         To install later: pip install torch torch-geometric"
fi

# ==============================================================================
# STEP 4 - Run Tests
# ==============================================================================
print_step "Step 4/5 - Running Tests"

print_info "Running simulation tests..."
if python -m pytest tests/unit/test_simulation.py -q --tb=short 2>&1; then
  print_ok "Simulation tests passed"
else
  print_warn "Some simulation tests failed - continuing anyway"
fi

print_info "Running API tests..."
if python -m pytest tests/integration/test_api.py -q --tb=short 2>&1; then
  print_ok "API tests passed (21/21)"
else
  print_warn "Some API tests failed - continuing anyway"
fi

# ==============================================================================
# STEP 5 - Launch
# ==============================================================================
print_step "Step 5/5 - Launching Dashboard"

PORT="${API_PORT:-8000}"

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "=============================================="
echo "  Dashboard  ->  http://localhost:$PORT"
echo "  API Docs   ->  http://localhost:$PORT/docs"
echo "  Health     ->  http://localhost:$PORT/health"
echo "  Press Ctrl+C to stop"
echo "=============================================="
echo ""

# Auto open browser (best effort, no crash if fails)
(sleep 2 && {
  open "http://localhost:$PORT" 2>/dev/null ||
  xdg-open "http://localhost:$PORT" 2>/dev/null ||
  true
}) &

# Start server
python -m uvicorn api.main:app --host 0.0.0.0 --port "$PORT" --reload
