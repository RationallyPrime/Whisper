#!/bin/bash
# Launch the RT-Whisper transcriber as a background process

cd "$(dirname "$0")"
source .venv/bin/activate

# Run RT-Whisper in the background using python3 (Python 3.10), redirect output to a log file
nohup python3 -m rt_whisper "$@" > ~/.whisper_logs/rt_whisper_stdout.log 2>&1 &

# Save the process ID for potential future termination
echo $! > ~/.whisper_logs/rt_whisper.pid

echo "RT-Whisper started in background (PID: $!)."
echo "Use kill_transcribe.sh to stop it."
