#!/bin/bash
nohup /home/rationallyprime/Whisper/.venv/bin/python3.10 /home/rationallyprime/Whisper/rt_whisper.py > transcribe_log.txt 2>&1 &
