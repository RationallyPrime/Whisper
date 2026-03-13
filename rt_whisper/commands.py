"""Command handler for file-based IPC dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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


def write_status(
    log_dir: Path,
    *,
    is_recording: bool,
    pid: int,
    started_at: float,
    last_command: str | None = None,
    last_command_time: float | None = None,
) -> None:
    """Atomically write daemon status to status.json."""
    payload = {
        "is_recording": is_recording,
        "pid": pid,
        "started_at": started_at,
        "last_command": last_command,
        "last_command_time": last_command_time,
    }
    status_path = log_dir / "status.json"
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, status_path)


def _write_ack(
    log_dir: Path,
    *,
    command_id: str | None,
    command: str,
    success: bool,
    error: str | None = None,
) -> None:
    """Atomically write command acknowledgement to ack.json."""
    if command_id is None:
        return
    payload = {
        "command_id": command_id,
        "command": command,
        "success": success,
        "error": error,
        "timestamp": time.time(),
    }
    ack_path = log_dir / "ack.json"
    tmp = ack_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, ack_path)


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
        self._started_at = time.time()

    def check_for_commands(self) -> None:
        """Check for a command file and dispatch if present and fresh."""
        if not self._command_file.exists():
            return

        command = ""
        command_id: str | None = None
        try:
            with open(self._command_file) as f:
                command_data = json.load(f)

            command = command_data.get("command", "")
            command_id = command_data.get("command_id")
            timestamp: float = command_data.get("timestamp", 0)

            if time.time() - timestamp > _COMMAND_TTL_SECONDS:
                self._command_file.unlink(missing_ok=True)
                return

            logger.info("Processing command: %s", command)
            self._dispatch(command)
            self._command_file.unlink(missing_ok=True)

            self._update_status(command)
            _write_ack(
                self._config.log_dir,
                command_id=command_id,
                command=command,
                success=True,
            )
        except Exception:
            logger.exception("Error processing command")
            self._command_file.unlink(missing_ok=True)
            _write_ack(
                self._config.log_dir,
                command_id=command_id,
                command=command,
                success=False,
                error="dispatch error",
            )

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

    def _update_status(self, command: str) -> None:
        """Write current daemon status to status.json."""
        write_status(
            self._config.log_dir,
            is_recording=self._recorder.is_recording,
            pid=os.getpid(),
            started_at=self._started_at,
            last_command=command,
            last_command_time=time.time(),
        )
