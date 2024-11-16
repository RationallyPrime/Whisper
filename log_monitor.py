import time
from pathlib import Path
from typing import Generator, Optional
import logging
from datetime import datetime

def monitor_log_file(filepath: str, interval: float = 0.1) -> Generator[str, None, None]:
    """
    Monitors a log file for changes and yields new lines in real-time.
    
    Args:
        filepath: Path to the log file to monitor
        interval: Time in seconds between file checks
        
    Yields:
        New lines added to the log file as they appear
        
    Raises:
        FileNotFoundError: If the specified log file does not exist
    """
    try:
        with open(filepath, 'r') as file:
            # Seek to end of file
            file.seek(0, 2)
            
            while True:
                current_position = file.tell()
                line = file.readline()
                
                if not line:
                    # No new data, sleep briefly
                    file.seek(current_position)
                    time.sleep(interval)
                else:
                    yield line.rstrip()

    except FileNotFoundError:
        logging.error(f"Log file not found: {filepath}")
        raise

def main():
    """Example usage of the log monitor"""
    log_path = Path.home() / ".whisper_logs" / "whisper.log"
    
    try:
        print(f"\n🔍 Monitoring {log_path}...")
        print("Press Ctrl+C to stop\n")
        
        for new_line in monitor_log_file(str(log_path)):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {new_line}")
            
    except KeyboardInterrupt:
        print("\n✨ Stopping log monitor...")
    except Exception as e:
        logging.error(f"Error monitoring log file: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
