#!/usr/bin/env -S uv run python
"""Send a command to process clipboard content with Claude in RT-Whisper command mode."""

import json
import time
import sys
import argparse
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
log_dir = Path.home() / ".whisper_logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "streamdeck.log"

# Reset root logger's handlers for Python 3.12+ compatibility
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            log_file,
            maxBytes=1 * 1024 * 1024,  # 1 MB
            backupCount=3,
            encoding="utf-8",
        )
    ],
)

def send_command(command):
    """Send a command to RT-Whisper by writing to the command file."""
    command_file = log_dir / "command.json"
    command_data = {
        "command": command,
        "timestamp": time.time(),
    }
    
    try:
        with open(command_file, "w") as f:
            json.dump(command_data, f)
        logging.info(f"Command sent: {command}")
        print(f"Command sent: {command}")
        return True
    except Exception as e:
        logging.error(f"Error sending command: {e}")
        print(f"Error: {e}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Send commands to Claude via RT-Whisper")
    parser.add_argument("--explain", action="store_true", help="Explain the clipboard content")
    parser.add_argument("--summarize", action="store_true", help="Summarize the clipboard content")
    parser.add_argument("--promptify", action="store_true", help="Convert clipboard content to a prompt")
    parser.add_argument("--reformat", action="store_true", help="Reformat the clipboard content")
    parser.add_argument("--implement", action="store_true", help="Implement code from clipboard content")
    parser.add_argument("--command", action="store_true", help="Generate command line from clipboard content")
    parser.add_argument("--translate", action="store_true", help="Translate clipboard content to English")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Default command
    command = "process_clipboard"
    
    # Check for command-line arguments using argparse
    if args.explain:
        command = "explain_clipboard"
    elif args.summarize:
        command = "summarize_clipboard"
    elif args.promptify:
        command = "promptify_clipboard"
    elif args.reformat:
        command = "reformat_clipboard"
    elif args.implement:
        command = "implement_clipboard"
    elif args.command:
        command = "command_clipboard"
    elif args.translate:
        command = "translate_clipboard"
    # Also support the old positional argument style
    elif len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        action = sys.argv[1].lower()
        if action == "explain":
            command = "explain_clipboard"
        elif action == "summarize":
            command = "summarize_clipboard"
        elif action == "promptify":
            command = "promptify_clipboard"
        elif action == "reformat":
            command = "reformat_clipboard"
        elif action == "implement":
            command = "implement_clipboard"
        elif action == "command":
            command = "command_clipboard"
        elif action == "translate":
            command = "translate_clipboard"
    
    logging.info(f"StreamDeck: Sending {command} command")
    send_command(command)