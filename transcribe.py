#!/usr/bin/env python3
"""Simple launcher for rt_whisper."""

import sys
from rt_whisper.transcriber import WhisperTranscriber

if __name__ == "__main__":
    print("Starting RT-Whisper Transcriber...")
    try:
        # Use CUDA by default, with large-v3 model and float16 for best performance
        transcriber = WhisperTranscriber(
            model_name="large-v3",
            device="cuda",
            compute_type="float16"
        )
        transcriber.run()
    except KeyboardInterrupt:
        print("\nTranscriber stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 
