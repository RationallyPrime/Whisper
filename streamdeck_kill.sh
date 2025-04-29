#!/bin/bash
# StreamDeck script to kill RT-Whisper

# Set up log directory path
LOGS_DIR=~/.whisper_logs
PID_FILE="$LOGS_DIR/rt_whisper.pid"

echo "$(date): Attempting to stop RT-Whisper..." >> "$LOGS_DIR/streamdeck.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Found PID file. Stopping RT-Whisper process (PID: $PID)..." >> "$LOGS_DIR/streamdeck.log"
    
    if kill -15 "$PID" 2>/dev/null; then
        echo "RT-Whisper process stopped gracefully." >> "$LOGS_DIR/streamdeck.log"
        echo "RT-Whisper stopped."
        rm "$PID_FILE"
    else
        echo "Process with PID $PID not found or couldn't be stopped gracefully." >> "$LOGS_DIR/streamdeck.log"
        echo "Attempting forced termination..." >> "$LOGS_DIR/streamdeck.log"
        
        if kill -9 "$PID" 2>/dev/null; then
            echo "RT-Whisper process terminated." >> "$LOGS_DIR/streamdeck.log"
            echo "RT-Whisper terminated."
        else
            echo "Could not terminate process. It may have already ended." >> "$LOGS_DIR/streamdeck.log"
            echo "RT-Whisper not running."
        fi
        
        rm -f "$PID_FILE"
    fi
else
    echo "No PID file found at $PID_FILE." >> "$LOGS_DIR/streamdeck.log"
    echo "Looking for any RT-Whisper processes..." >> "$LOGS_DIR/streamdeck.log"
    pkill -f "python.*rt_whisper"
    
    # Check if any processes were found and killed
    if [ $? -eq 0 ]; then
        echo "RT-Whisper processes stopped." >> "$LOGS_DIR/streamdeck.log"
        echo "RT-Whisper stopped."
    else
        echo "No RT-Whisper processes found running." >> "$LOGS_DIR/streamdeck.log"
        echo "RT-Whisper not running."
    fi
fi

# Clean up any command files
rm -f "$LOGS_DIR/command.json" 2>/dev/null
