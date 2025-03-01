#!/bin/bash
# Script to set up Claude integration for RT-Whisper

cd "$(dirname "$0")"
source .venv/bin/activate

# Run the setup script using python3 (Python 3.10)
python3 setup_claude.py

echo "Claude setup complete." 
