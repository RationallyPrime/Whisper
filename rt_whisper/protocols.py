"""Protocol definitions for RT-Whisper module boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class Recorder(Protocol):
    """Audio recording interface."""

    def start_recording(self) -> None: ...

    def stop_recording(self) -> Path: ...

    def play_beep(self, frequency: int = 440, duration: float = 0.1) -> None: ...

    @property
    def is_recording(self) -> bool: ...


@runtime_checkable
class Transcriber(Protocol):
    """Speech-to-text interface."""

    def transcribe_audio(self, audio_path: Path) -> str: ...


@runtime_checkable
class TextProcessor(Protocol):
    """Post-transcription text processing (e.g., Claude)."""

    async def process_clipboard(self) -> None: ...
