#!/usr/bin/env bash
# One-shot installer for the Thunder Compute CLI (tnr) on macOS.
# Uninstalls any stale Homebrew copy and installs the latest from the
# official install script. Idempotent: safe to re-run.

set -euo pipefail

echo "[install_tnr] checking for stale Homebrew install..."
if brew list --formula 2>/dev/null | grep -qx tnr; then
    echo "[install_tnr] removing Homebrew tnr (stuck at 2.0.33)..."
    brew uninstall tnr
fi
if brew tap 2>/dev/null | grep -qix 'thunder-compute/tnr'; then
    echo "[install_tnr] untapping thunder-compute/tnr..."
    brew untap thunder-compute/tnr || true
fi

echo "[install_tnr] installing latest tnr from official script..."
curl -fsSL https://raw.githubusercontent.com/Thunder-Compute/thunder-cli/main/scripts/install.sh | bash

# The installer adds ~/.tnr/bin to PATH in ~/.zshrc but only on next shell.
# Export it for this session so the verify step below works.
export PATH="${HOME}/.tnr/bin:${PATH}"

echo
echo "[install_tnr] verifying..."
if ! command -v tnr >/dev/null 2>&1; then
    echo "tnr not on PATH. Open a new terminal or: source ~/.zshrc" >&2
    exit 1
fi

tnr --version

echo
echo "[install_tnr] done."
echo
echo "Next steps:"
echo "  1. Open a new terminal (or: source ~/.zshrc)"
echo "  2. tnr login --token YOUR_THUNDER_API_TOKEN"
echo "  3. tnr status"
