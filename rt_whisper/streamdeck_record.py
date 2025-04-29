#!/usr/bin/env python
"""
StreamDeck button for starting RT-Whisper recording
"""

import os
import sys
import time
from pathlib import Path
import logging
import json

# Configure logging
log_dir = Path.home() / ".whisper_logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=str(log_dir / "streamdeck.log"),
    filemode="a",
)


def find_rt_whisper_process():
    """Check if RT-Whisper is already running"""
    import psutil

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["name"] == "python" and any(
                "rt_whisper" in arg for arg in proc.info["cmdline"] if arg
            ):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def main():
    """Start recording with RT-Whisper"""
    # Path to the Whisper directory
    whisper_dir = Path.home() / "Whisper"

    # Check if RT-Whisper is already running
    pid = find_rt_whisper_process()
    logging.info(f"Checking for existing RT-Whisper process: {pid}")

    # If RT-Whisper is not running, start it first
    if pid is None:
        logging.info("Starting RT-Whisper first...")
        start_script = whisper_dir / "streamdeck_start.sh"
        if not start_script.exists():
            logging.error(f"Start script not found at {start_script}")
            print(f"Error: Start script not found at {start_script}")
            sys.exit(1)

        os.system(f"bash {start_script}")
        logging.info("Waiting for RT-Whisper to initialize...")
        # Give it a second to start up
        time.sleep(2)

    # Signal RT-Whisper to start recording
    signal_path = log_dir / "command.json"
    with open(signal_path, "w") as f:
        json.dump({"command": "start_recording", "timestamp": time.time()}, f)

    logging.info("Start recording command sent")
    print("Recording started...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Error in streamdeck_record.py")
        print(f"Error: {e}")
