"""Main entry point for RT-Whisper — CLI parsing and DI wiring."""

from __future__ import annotations

import argparse
import logging
import warnings
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import sounddevice as sd

from .audio import AudioCapture
from .claude_client import ClaudeClient
from .commands import CommandHandler
from .config import WhisperConfig
from .orchestrator import Orchestrator
from .transcriber import WhisperTranscriber


def setup_logging(log_dir: Path, debug: bool = False) -> None:
    """Configure rotating file logging."""
    log_dir.mkdir(exist_ok=True, parents=True, mode=0o755)
    log_file = log_dir / "rt_whisper.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
        ],
    )
    warnings.filterwarnings("ignore")
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def list_audio_devices() -> None:
    """List available audio devices."""
    print("\nAvailable audio devices:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} (Channels: {device['max_input_channels']})")
    print("-" * 50)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:].
    """
    parser = argparse.ArgumentParser(description="Real-time Whisper Transcription")

    parser.add_argument(
        "-l",
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device ID to use for recording",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Faster-Whisper model to use",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=None,
        choices=["float16", "float32", "int8", "int8_float16"],
        help="Compute type for inference",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache models",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args(argv)


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Map CLI args to dotted config override keys."""
    overrides: dict[str, Any] = {}
    if args.device is not None:
        overrides["audio.device"] = args.device
    if args.model is not None:
        overrides["transcription.model"] = args.model
    if args.device_type is not None:
        overrides["transcription.device_type"] = args.device_type
    if args.compute_type is not None:
        overrides["transcription.compute_type"] = args.compute_type
    if args.cache_dir is not None:
        overrides["transcription.cache_dir"] = args.cache_dir
    return overrides


def main(argv: list[str] | None = None) -> None:
    """Main entry point — DI wiring and run.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:].
    """
    args = parse_args(argv)

    if args.list_devices:
        list_audio_devices()
        return

    # Load config with CLI overrides
    overrides = _build_cli_overrides(args)
    config = WhisperConfig.load(cli_overrides=overrides)

    # Set up logging
    setup_logging(config.daemon.log_dir, debug=args.debug)

    # Ensure directories exist
    config.daemon.tmp_dir.mkdir(exist_ok=True, parents=True)
    config.daemon.config_dir.mkdir(exist_ok=True, parents=True)

    # DI wiring
    audio = AudioCapture(config.audio, config.daemon.tmp_dir)
    transcriber = WhisperTranscriber(config.transcription)

    claude: ClaudeClient | None = None
    if config.claude.enable and config.claude.api_key:
        claude = ClaudeClient(config.claude, audio)

    commands = CommandHandler(
        config.daemon,
        recorder=audio,
        transcriber=transcriber,
        text_processor=claude,
    )
    orchestrator = Orchestrator(config, commands)

    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
