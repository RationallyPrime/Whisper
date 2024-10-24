#!/usr/bin/env python3
import json
from pathlib import Path
from dotenv import load_dotenv
import os

def setup_config():
    config_dir = Path.home() / ".whisper_config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.json"
    
    print("Setting up Claude and TTS configuration...")
    
    # Try to load API key from .env file
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # If not found in .env, prompt for manual input
    if not api_key:
        print("No ANTHROPIC_API_KEY found in .env file")
        api_key = input("Enter your Anthropic API key: ").strip()
    else:
        print("Found ANTHROPIC_API_KEY in environment variables")
    
    if not api_key:
        print("Error: API key is required")
        return
    
    config = {
        "enable_claude_tts": True,
        "anthropic_api_key": api_key
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    print("Configuration saved! You can now use F10 to process text with Claude and TTS.")
    print("TIP: To change the API key later, either:")
    print("  1. Update your .env file and run this script again")
    print(f"  2. Directly edit {config_path}")

if __name__ == "__main__":
    setup_config()