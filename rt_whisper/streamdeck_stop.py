#!/usr/bin/env python3
"""
StreamDeck button for stopping RT-Whisper recording
"""
from pathlib import Path
import logging
import json

# Configure logging
log_dir = Path.home() / ".whisper_logs"
log_dir.mkdir(exist_ok=True)

# Reset root logger's handlers for Python 3.12+ compatibility
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=str(log_dir / "streamdeck.log"),
    filemode="a",
)

def main():
    """Stop recording with RT-Whisper"""
    # Signal RT-Whisper to stop recording
    signal_path = log_dir / "command.json"
    
    import time
    with open(signal_path, "w") as f:
        json.dump({"command": "stop_recording", "timestamp": time.time()}, f)
    
    logging.info("Stop recording command sent")
    print("Recording stopped. Transcribing...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Error in streamdeck_stop.py")
        print(f"Error: {e}")
