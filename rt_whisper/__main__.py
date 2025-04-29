"""Main entry point for RT-Whisper."""

import argparse
import logging
import sounddevice as sd
from pathlib import Path
from .transcriber import WhisperTranscriber


def list_audio_devices():
    """List available audio devices."""
    print("\nAvailable audio devices:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} (Channels: {device['max_input_channels']})")
    print("-" * 50)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Real-time Whisper Transcription")
    
    parser.add_argument(
        "-l", "--list-devices", 
        action="store_true", 
        help="List audio devices and exit"
    )
    
    parser.add_argument(
        "--device", 
        type=int,
        default=None, 
        help="Audio device ID to use for recording"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "distil-large-v3"],
        help="Faster-Whisper model to use"
    )
    
    parser.add_argument(
        "--device-type", 
        type=str, 
        default="cuda", 
        choices=["cuda", "cpu"],
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--compute-type", 
        type=str, 
        default="float16", 
        choices=["float16", "float32", "int8", "int8_float16"],
        help="Compute type for inference"
    )
    
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default=None,
        help="Directory to cache models"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--command-mode",
        action="store_true",
        help="Run in command mode instead of hotkey mode"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Debug logging enabled")
    
    # Parse the cache directory
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    
    # Initialize the transcriber
    transcriber = WhisperTranscriber(
        model_name=args.model,
        device=args.device_type,
        compute_type=args.compute_type,
        cache_dir=cache_dir,
    )
    
    # Configure audio device if specified
    if args.device is not None:
        transcriber.audio_config.device = args.device
    
    # Run the transcriber
    try:
        transcriber.run(command_mode=args.command_mode)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main() 
