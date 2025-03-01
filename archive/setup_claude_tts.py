from pathlib import Path
import json
import os
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPTS

def validate_prompts():
    """Validate that all required prompts are present and well-formed"""
    required_prompts = ["promptify", "reformat", "implement", "command"]
    missing_prompts = [p for p in required_prompts if p not in SYSTEM_PROMPTS]
    
    if missing_prompts:
        print(f"Warning: Missing required prompts: {', '.join(missing_prompts)}")
        return False
    return True

def setup_modern_tts():
    """Setup function for modern TTS configuration"""
    config_dir = Path.home() / ".whisper_config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.json"
    
    print("Setting up Claude and modern TTS configuration...")
    
    # Validate system prompts
    if not validate_prompts():
        print("Error: System prompts validation failed")
        return
    
    # Create TTS output directory
    tts_output_dir = Path.home() / ".tts_output"
    tts_output_dir.mkdir(exist_ok=True)
    
    # Try to load API key from .env file
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # If not found in .env, prompt for manual input
    if not api_key:
        print("No ANTHROPIC_API_KEY found in .env file")
        api_key = input("Enter your Anthropic API key: ").strip()
    
    if not api_key:
        print("Error: API key is required")
        return
    
    config = {
        "enable_claude_tts": True,
        "anthropic_api_key": api_key,
        "available_commands": list(SYSTEM_PROMPTS.keys())
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    print("\nConfiguration is ready!")
    print("Available commands:")
    for cmd in SYSTEM_PROMPTS.keys():
        print(f"  - {cmd} this: [content]")
    print("\nYou can now use:")
    print("  - F12 to start recording")
    print("  - F11 to stop recording")
    print("  - F10 to process text with Claude and TTS")
    print("\nTIP: To change the API key later, either:")
    print("  1. Update your .env file and run this script again")
    print(f"  2. Directly edit {config_path}")

if __name__ == "__main__":
    setup_modern_tts()