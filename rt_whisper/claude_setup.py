"""Setup utility for Claude API integration with RT-Whisper."""

from pathlib import Path
import json
import os
from dotenv import load_dotenv
from .prompts import SYSTEM_PROMPTS

def validate_prompts():
    """Validate that all required prompts are present and well-formed."""
    required_prompts = ["promptify", "reformat", "implement", "command", "explain", "translate", "summarize"]
    missing_prompts = [p for p in required_prompts if p not in SYSTEM_PROMPTS]
    
    if missing_prompts:
        print(f"Warning: Missing required prompts: {', '.join(missing_prompts)}")
        return False
    return True

def setup_claude():
    """Setup function for Claude API integration."""
    config_dir = Path.home() / ".whisper_config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.json"
    
    print("Setting up Claude API integration for RT-Whisper...")
    
    # Validate system prompts
    if not validate_prompts():
        print("Error: System prompts validation failed")
        return
    
    # Try to load API key from .env file
    env_path = Path.home() / "Whisper" / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try default locations
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # If not found in .env, prompt for manual input
    if not api_key:
        print("No ANTHROPIC_API_KEY found in .env file")
        api_key = input("Enter your Anthropic API key: ").strip()
    
    if not api_key:
        print("Error: API key is required")
        return
    
    # Check for existing config and update only Claude-related settings
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            config = {}
    else:
        config = {}
    
    # Update config with Claude settings
    config.update({
        "enable_claude": True,
        "anthropic_api_key": api_key,
        "available_commands": list(SYSTEM_PROMPTS.keys())
    })
    
    # Write updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print("\nConfiguration is ready!")
    print("Available commands:")
    for cmd in SYSTEM_PROMPTS.keys():
        print(f"  - {cmd} this: [content]")
    print("\nYou can now use RT-Whisper with Claude integration.")
    print("Standard commands (All require Ctrl+Alt+Shift):")
    print("  - Press and hold F12: Record while held (release to transcribe)")
    print("  - F10: Process clipboard text with Claude (Explain)")
    print("  - F9: Reformat transcribed text")
    print("  - F8: Create LLM prompts")
    print("  - F7: Generate code implementation")
    print("  - F6: Generate terminal commands")
    print("  - F5: Summarize text")
    
    print("\nStream Deck compatible commands (No modifiers needed):")
    print("  - Press and hold Numpad 0: Record while held (release to transcribe)")
    print("  - Numpad 1: Process clipboard with Claude (Explain)")
    print("  - Numpad 2: Reformat transcribed text")
    print("  - Numpad 3: Create LLM prompts")
    print("  - Numpad 4: Generate code implementation")
    print("  - Numpad 5: Generate terminal commands")
    print("  - Numpad 6: Summarize text")
    
    print("\nTIP: To change the API key later, either:")
    print("  1. Update your .env file with ANTHROPIC_API_KEY=your_key")
    print(f"  2. Directly edit {config_path}")

if __name__ == "__main__":
    setup_claude() 
