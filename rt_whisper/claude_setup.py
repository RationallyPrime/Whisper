"""Setup utility for Claude API integration with RT-Whisper."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPTS


def validate_prompts() -> bool:
    """Validate that all required prompts are present and well-formed."""
    required_prompts = [
        "promptify", "reformat", "implement", "command", "explain", "translate", "summarize"
    ]
    missing_prompts = [p for p in required_prompts if p not in SYSTEM_PROMPTS]

    if missing_prompts:
        print(f"Warning: Missing required prompts: {', '.join(missing_prompts)}")
        return False
    return True


def setup_claude() -> None:
    """Setup function for Claude API integration."""
    config_dir = Path.home() / ".whisper_config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.json"

    print("Setting up Claude API integration for RT-Whisper...")

    if not validate_prompts():
        print("Error: System prompts validation failed")
        return

    env_path = Path.home() / "Whisper" / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("No ANTHROPIC_API_KEY found in .env file")
        api_key = input("Enter your Anthropic API key: ").strip()

    if not api_key:
        print("Error: API key is required")
        return

    config: dict[str, object] = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            config = {}

    config.update({
        "enable_claude": True,
        "anthropic_api_key": api_key,
        "available_commands": list(SYSTEM_PROMPTS.keys()),
    })

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\nConfiguration is ready!")
    print("Available commands:")
    for cmd in SYSTEM_PROMPTS:
        print(f"  - {cmd} this: [content]")
    print("\nUse StreamDeck buttons or `just dev` to start RT-Whisper.")
    print("\nTIP: To change the API key later, either:")
    print("  1. Update your .env file with ANTHROPIC_API_KEY=your_key")
    print(f"  2. Directly edit {config_path}")


if __name__ == "__main__":
    setup_claude()
