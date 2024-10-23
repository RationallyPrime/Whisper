#!/bin/bash
# Find the PID of the transcription script and kill it
pid=$(ps aux | grep '[r]t_whisper.py' | awk '{print $2}')

if [ -z "$pid" ]; then
    echo "No transcription process running."
else
    kill $pid
    echo "Transcription process (PID $pid) has been killed."
fi

