#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Night Notes — setup script
# Installs all dependencies and checks Ollama + model status
# ─────────────────────────────────────────────────────────────

set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
RESET="\033[0m"

ok()   { echo -e "  ${GREEN}✓${RESET}  $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
err()  { echo -e "  ${RED}✗${RESET}  $1"; }
info() { echo -e "  ${CYAN}→${RESET}  $1"; }

echo ""
echo -e "${BOLD}  🎙  Night Notes — Setup${RESET}"
echo "  ─────────────────────────────"
echo ""

# ── 1. Python ────────────────────────────────────────────────
info "Checking Python 3..."
if command -v python3 &>/dev/null; then
  PY=$(python3 --version 2>&1)
  ok "Found $PY"
else
  err "Python 3 not found. Install from https://python.org and re-run."
  exit 1
fi

# ── 2. pip ───────────────────────────────────────────────────
info "Checking pip..."
if python3 -m pip --version &>/dev/null; then
  ok "pip available"
else
  warn "pip not found — attempting to install via ensurepip..."
  python3 -m ensurepip --upgrade || {
    err "Could not install pip. Install it manually and re-run."
    exit 1
  }
fi

# ── 3. faster-whisper ────────────────────────────────────────
info "Installing faster-whisper..."
if python3 -c "import faster_whisper" &>/dev/null; then
  ok "faster-whisper already installed"
else
  python3 -m pip install faster-whisper --quiet && ok "faster-whisper installed" || {
    err "Install failed. Try manually: pip3 install faster-whisper"
    exit 1
  }
fi

# ── 4. Ollama ────────────────────────────────────────────────
echo ""
info "Checking Ollama..."
if command -v ollama &>/dev/null; then
  ok "Ollama found: $(ollama --version 2>&1 | head -1)"
else
  warn "Ollama not found."
  echo ""
  echo -e "  Install it from ${CYAN}https://ollama.com/download${RESET}"
  echo "  Then run:  ollama pull qwen2.5:14b"
  echo "  Then re-run this script."
  echo ""
  OLLAMA_MISSING=1
fi

if [ -z "$OLLAMA_MISSING" ]; then
  if curl -s http://localhost:11434/api/tags &>/dev/null; then
    ok "Ollama daemon is running"

    MODELS=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}')
    QWEN_MODEL=$(echo "$MODELS" | grep -i "qwen" | head -1)

    if [ -n "$QWEN_MODEL" ]; then
      ok "Found Qwen model: $QWEN_MODEL"
      # Patch server.py with the actual model name
      sed -i.bak "s/^OLLAMA_MODEL = .*/OLLAMA_MODEL = \"$QWEN_MODEL\"/" server.py && rm -f server.py.bak
      ok "Updated server.py → OLLAMA_MODEL = \"$QWEN_MODEL\""
    else
      warn "No Qwen model found. Pulling qwen2.5:14b..."
      ollama pull qwen2.5:14b && ok "qwen2.5:14b ready" || warn "Pull failed — run: ollama pull qwen2.5:14b"
      sed -i.bak "s/^OLLAMA_MODEL = .*/OLLAMA_MODEL = \"qwen2.5:14b\"/" server.py && rm -f server.py.bak
    fi
  else
    warn "Ollama installed but not running."
    echo "  Start it with:  ollama serve"
    echo "  Then re-run this script to auto-detect your model."
  fi
fi

# ── 5. Whisper model pre-download ────────────────────────────
echo ""
info "Pre-downloading Whisper large-v3 model (~1.5 GB)..."
echo "  (Press Ctrl+C to skip — it will download on first transcription instead)"
echo ""
python3 - <<'EOF'
try:
    from faster_whisper import WhisperModel
    print("  Downloading large-v3 — this is a one-time download...")
    WhisperModel("large-v3", device="cpu", compute_type="int8")
    print("  ✓  Whisper large-v3 ready")
except KeyboardInterrupt:
    print("\n  Skipped — will download on first use.")
except Exception as e:
    print(f"  ⚠  Could not pre-download: {e}")
    print("     It will download automatically on first use.")
EOF

# ── Done ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  ✓  Setup complete!${RESET}"
echo ""
echo "  Start the app:"
echo -e "    ${CYAN}python3 server.py${RESET}"
echo ""
echo "  Then open:  http://localhost:8765"
echo ""