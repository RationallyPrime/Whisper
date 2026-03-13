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

COSMIC_CUSTOM_DIR="$HOME/.config/cosmic/com.system76.CosmonConfig/v1/custom_shortcuts"

# Resolve rtwhisperctl path — prefer the uv-installed console script
if command -v rtwhisperctl &>/dev/null; then
    CTL="rtwhisperctl"
else
    # Fall back to uv run
    CTL="uv run --project $HOME/Whisper rtwhisperctl"
fi

# COSMIC uses dconf on some builds, custom config on others.
# Try cosmic-settings CLI first, then fall back to manual instructions.
if command -v cosmic-settings &>/dev/null; then
    echo "COSMIC Settings detected."
    echo ""
    echo "Unfortunately, COSMIC custom shortcuts cannot yet be set via CLI."
    echo "Please add these shortcuts manually in COSMIC Settings → Keyboard → Custom Shortcuts:"
    echo ""
else
    echo "COSMIC Settings not found."
    echo "Please add these keyboard shortcuts manually via Settings → Keyboard → Custom Shortcuts:"
    echo ""
fi

echo "┌─────────────────┬────────────────────────────────────────────────────────────┐"
echo "│ Shortcut        │ Command                                                    │"
echo "├─────────────────┼────────────────────────────────────────────────────────────┤"
echo "│ Super+Shift+D   │ bash -lc '$CTL toggle --wait'                              │"
echo "│ Super+Shift+S   │ bash -lc '$CTL stop --wait'                                │"
echo "│ Super+Shift+E   │ bash -lc '$CTL clipboard explain'                          │"
echo "│ Super+Shift+R   │ bash -lc '$CTL clipboard reformat'                         │"
echo "│ Super+Shift+U   │ bash -lc '$CTL clipboard summarize'                        │"
echo "│ Super+Shift+P   │ bash -lc '$CTL clipboard promptify'                        │"
echo "│ Super+Shift+I   │ bash -lc '$CTL clipboard implement'                        │"
echo "│ Super+Shift+T   │ bash -lc '$CTL clipboard translate'                        │"
echo "└─────────────────┴────────────────────────────────────────────────────────────┘"
echo ""
echo "Note: Commands are wrapped in 'bash -lc' to ensure PATH resolution."
echo "      The --wait flag makes the shortcut block until the daemon acknowledges."
echo ""
echo "Quick start:"
echo "  1. Open COSMIC Settings → Keyboard → Custom Shortcuts"
echo "  2. Click 'Add Shortcut' for each binding above"
echo "  3. Start the daemon: rtwhisperctl daemon  (or: just start)"
echo "  4. Press Super+Shift+D to toggle dictation"
