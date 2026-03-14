#!/bin/bash
# Install COSMIC desktop keyboard shortcuts for rtwhisperctl.
# Idempotent — re-running updates the config.
#
# Shortcuts:
#   Super+Shift+D  → toggle dictation
#   Super+Shift+S  → stop dictation
#   Super+Shift+E  → explain clipboard
#   Super+Shift+R  → reformat clipboard

set -euo pipefail

# Resolve rtwhisperctl to an absolute path that works outside the shell
# (COSMIC shortcuts run with minimal PATH, so relative names won't resolve)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_CTL="$SCRIPT_DIR/.venv/bin/rtwhisperctl"

if [ -x "$VENV_CTL" ]; then
    CTL="$VENV_CTL"
elif command -v rtwhisperctl &>/dev/null; then
    CTL="$(command -v rtwhisperctl)"
else
    echo "Error: rtwhisperctl not found. Run 'uv sync' first."
    exit 1
fi

echo "Using: $CTL"
echo ""

if command -v cosmic-settings &>/dev/null; then
    echo "Add these shortcuts in COSMIC Settings → Keyboard → Custom Shortcuts:"
else
    echo "Add these keyboard shortcuts via Settings → Keyboard → Custom Shortcuts:"
fi
echo ""

echo "┌─────────────────┬──────────────────────────────────────────────────────────────┐"
echo "│ Shortcut        │ Command                                                      │"
echo "├─────────────────┼──────────────────────────────────────────────────────────────┤"
printf "│ Super+Shift+D   │ %-60s │\n" "$CTL toggle --wait"
printf "│ Super+Shift+S   │ %-60s │\n" "$CTL stop --wait"
printf "│ Super+Shift+E   │ %-60s │\n" "$CTL clipboard explain"
printf "│ Super+Shift+R   │ %-60s │\n" "$CTL clipboard reformat"
printf "│ Super+Shift+U   │ %-60s │\n" "$CTL clipboard summarize"
printf "│ Super+Shift+P   │ %-60s │\n" "$CTL clipboard promptify"
printf "│ Super+Shift+I   │ %-60s │\n" "$CTL clipboard implement"
printf "│ Super+Shift+T   │ %-60s │\n" "$CTL clipboard translate"
echo "└─────────────────┴──────────────────────────────────────────────────────────────┘"
echo ""
echo "These use the absolute path so they work from COSMIC shortcuts"
echo "without needing bash -lc or PATH setup."
echo ""
echo "Quick start:"
echo "  1. Open COSMIC Settings → Keyboard → Custom Shortcuts"
echo "  2. Click 'Add Shortcut' for each binding above"
echo "  3. Start the daemon: just start"
echo "  4. Press Super+Shift+D to toggle dictation"
