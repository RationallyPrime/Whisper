#!/bin/bash
# Kill the running RT-Whisper transcriber processes

PID_FILE=~/.whisper_logs/rt_whisper.pid

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Stopping RT-Whisper process (PID: $PID)..."
    
    if kill -15 "$PID" 2>/dev/null; then
        echo "RT-Whisper process stopped."
        rm "$PID_FILE"
    else
        echo "Process with PID $PID not found. It may have already terminated."
        rm "$PID_FILE"
        
        # As a fallback, try to find any RT-Whisper processes
        echo "Looking for any other RT-Whisper processes..."
        pkill -f "python.*rt_whisper"
    fi
else
    echo "No PID file found. Looking for RT-Whisper processes..."
    pkill -f "python.*rt_whisper"
    
    # Check if any processes were found and killed
    if [ $? -eq 0 ]; then
        echo "RT-Whisper processes stopped."
    else
        echo "No RT-Whisper processes found running."
    fi
fi

