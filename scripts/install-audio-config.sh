#!/bin/bash
# Install WirePlumber device-scoped config for Blue Yeti Nano.
# Idempotent — skips if content already matches.

set -euo pipefail

SRC="$(dirname "$0")/../config/wireplumber/10-yeti-nano.conf"
DEST_DIR="$HOME/.config/wireplumber/wireplumber.conf.d"
DEST="$DEST_DIR/10-yeti-nano.conf"

# Check WirePlumber version (conf file format requires >= 0.5)
if command -v wireplumber &>/dev/null; then
    WP_VERSION=$(wireplumber --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1)
    WP_MAJOR=$(echo "$WP_VERSION" | cut -d. -f1)
    WP_MINOR=$(echo "$WP_VERSION" | cut -d. -f2)
    if [ "${WP_MAJOR:-0}" -eq 0 ] && [ "${WP_MINOR:-0}" -lt 5 ]; then
        echo "Warning: WirePlumber $WP_VERSION detected. Conf file format requires >= 0.5."
        echo "Older versions use Lua config in main.lua.d/ — manual setup required."
        exit 1
    fi
    echo "WirePlumber $WP_VERSION detected (>= 0.5, conf format supported)."
else
    echo "Warning: WirePlumber not found. Installing config anyway for future use."
fi

# Check if already installed and identical
if [ -f "$DEST" ] && diff -q "$SRC" "$DEST" &>/dev/null; then
    echo "Audio config already installed and up to date."
    exit 0
fi

mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST"
echo "Installed Yeti Nano WirePlumber config to $DEST"
echo "Restart WirePlumber to apply: systemctl --user restart wireplumber"
