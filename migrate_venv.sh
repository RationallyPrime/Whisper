#!/bin/bash
# Script to migrate from old virtual environment to new one

set -e  # Exit on error

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"

echo "Migrating RT-Whisper virtual environment..."

# Check if old environments exist
if [ -d "venv" ]; then
    echo "Removing old 'venv' directory..."
    rm -rf venv
fi

# Check if .venv_new exists
if [ -d ".venv_new" ]; then
    # Check if .venv already exists
    if [ -d ".venv" ]; then
        echo "Removing existing '.venv' directory..."
        rm -rf .venv
    fi
    
    echo "Renaming '.venv_new' to '.venv'..."
    mv .venv_new .venv
    echo "Successfully renamed virtual environment."
else
    echo "'.venv_new' not found. No migration needed."
    
    # If .venv doesn't exist, create it
    if [ ! -d ".venv" ]; then
        echo "Creating new virtual environment '.venv' with Python 3.10..."
        # Use python3 to ensure we get system Python 3.10
        uv venv --python /usr/bin/python3 .venv
        
        echo "Installing dependencies..."
        source .venv/bin/activate
        uv pip install -e .
    fi
fi

echo "Setting up permissions..."
chmod +x transcribe.sh
chmod +x kill_transcribe.sh

echo "Migration complete."
echo "You can now use RT-Whisper with './transcribe.sh'" 
