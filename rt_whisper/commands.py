"""Command handler for file-based IPC dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

import pyperclip

if TYPE_CHECKING:
    from pathlib import Path

    from .config import DaemonConfig
    from .protocols import Recorder, TextProcessor, Transcriber

logger = logging.getLogger(__name__)

# Maps command names to the prefix prepended to clipboard text before Claude processing.
# Commands not in this map are handled directly (start/stop recording, process_clipboard).
_COMMAND_PREFIX_MAP: dict[str, str] = {
    "explain_clipboard": "explain this",
    "summarize_clipboard": "summarize this",
    "promptify_clipboard": "promptify this",
    "reformat_clipboard": "reformat this",
    "implement_clipboard": "implement this",
    "command_clipboard": "command line this",
    "translate_clipboard": "translate this into English",
}

_COMMAND_TTL_SECONDS = 5


class CommandHandler:
    """Parse and dispatch commands from command.json.

    Accepts protocol-typed dependencies — no concrete type coupling.
    """

    def __init__(
        self,
        config: DaemonConfig,
        recorder: Recorder,
        transcriber: Transcriber,
        text_processor: TextProcessor | None = None,
    ) -> None:
        self._config = config
        self._recorder = recorder
        self._transcriber = transcriber
        self._text_processor = text_processor
        self._command_file = config.log_dir / "command.json"

    def check_for_commands(self) -> None:
        """Check for a command file and dispatch if present and fresh."""
        if not self._command_file.exists():
            return

        try:
            with open(self._command_file) as f:
                command_data = json.load(f)

            command: str = command_data.get("command", "")
            timestamp: float = command_data.get("timestamp", 0)

            if time.time() - timestamp > _COMMAND_TTL_SECONDS:
                self._command_file.unlink(missing_ok=True)
                return

            logger.info("Processing command: %s", command)
            self._dispatch(command)
            self._command_file.unlink(missing_ok=True)
        except Exception:
            logger.exception("Error processing command")
            self._command_file.unlink(missing_ok=True)

    def _dispatch(self, command: str) -> None:
        """Route a command string to the appropriate handler."""
        if command == "start_recording" and not self._recorder.is_recording:
            self._recorder.start_recording()
            return

        if command == "stop_recording" and self._recorder.is_recording:
            self._on_stop_recording()
            return

        if command == "process_clipboard" and self._text_processor:
            asyncio.run(self._text_processor.process_clipboard())
            return

        if command in _COMMAND_PREFIX_MAP and self._text_processor:
            self._on_prefixed_command(_COMMAND_PREFIX_MAP[command])
            return

    def _on_stop_recording(self) -> None:
        """Stop recording, transcribe, copy to clipboard, beep, cleanup."""
        wav_path: Path | None = None
        try:
            wav_path = self._recorder.stop_recording()
            text = self._transcriber.transcribe_audio(wav_path)

            if text and text != "Transcription failed":
                pyperclip.copy(text)
                logger.info("Transcribed and copied to clipboard: %s", text)
                self._recorder.play_beep(660, 0.1)
            else:
                logger.warning("No text transcribed")
                self._recorder.play_beep(330, 0.2)
        except RuntimeError as e:
            logger.warning("Stop recording: %s", e)
            self._recorder.play_beep(330, 0.2)
        except Exception:
            logger.exception("Processing error during stop_recording")
            self._recorder.play_beep(220, 0.3)
        finally:
            if wav_path is not None and wav_path.exists():
                wav_path.unlink()

    def _on_prefixed_command(self, prefix: str) -> None:
        """Prepend prefix to clipboard text and process with Claude."""
        text = pyperclip.paste()
        if not text:
            return
        pyperclip.copy(f"{prefix}: {text}")
        asyncio.run(self._text_processor.process_clipboard())  # type: ignore[union-attr]
