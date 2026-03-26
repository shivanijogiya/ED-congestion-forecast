#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ED Congestion Forecasting — One-command local startup (Mac / Linux)
# Usage: ./start.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   ED Congestion Forecasting System                      ║${NC}"
echo -e "${BLUE}║   Graph-Aware Deep Learning · Real-Time · 6 Hospitals   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
  echo -e "${RED}✗ Python not found. Install Python 3.11+ from https://python.org${NC}"
  exit 1
fi
PY=$(command -v python3 || command -v python)
echo -e "${GREEN}✓ Python:${NC} $($PY --version)"

# ── Check pip ─────────────────────────────────────────────────────────────────
PIP=$(command -v pip3 || command -v pip)

# ── Install core dependencies ─────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}► Installing dependencies...${NC}"
$PIP install --quiet fastapi uvicorn pydantic pydantic-settings \
  httpx pytest pytest-asyncio numpy scipy 2>&1 | tail -2
echo -e "${GREEN}✓ Dependencies installed${NC}"

# ── Run tests ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}► Running tests...${NC}"
$PY -m pytest tests/unit/test_simulation.py tests/integration/test_api.py \
  -q --tb=short 2>&1
echo -e "${GREEN}✓ All tests passed${NC}"

# ── Start server ──────────────────────────────────────────────────────────────
PORT=${API_PORT:-8000}
echo ""
echo -e "${GREEN}✓ Starting server on port ${PORT}...${NC}"
echo ""
echo -e "${BLUE}┌─────────────────────────────────────────────────────┐${NC}"
echo -e "${BLUE}│  Dashboard  →  http://localhost:${PORT}              │${NC}"
echo -e "${BLUE}│  API Docs   →  http://localhost:${PORT}/docs         │${NC}"
echo -e "${BLUE}│  Health     →  http://localhost:${PORT}/health       │${NC}"
echo -e "${BLUE}│                                                      │${NC}"
echo -e "${BLUE}│  Press Ctrl+C to stop                                │${NC}"
echo -e "${BLUE}└─────────────────────────────────────────────────────┘${NC}"
echo ""

# Open browser after 2 seconds (background)
(sleep 2 && (open "http://localhost:${PORT}" 2>/dev/null || xdg-open "http://localhost:${PORT}" 2>/dev/null)) &

$PY -m uvicorn api.main:app --host 0.0.0.0 --port $PORT --reload
