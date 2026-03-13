"""Shared fixtures and protocol fakes for RT-Whisper tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rt_whisper.config import AudioConfig, ClaudeConfig, DaemonConfig, TranscriptionConfig, WhisperConfig
from rt_whisper.protocols import Recorder, TextProcessor, Transcriber


class FakeRecorder:
    """In-memory Recorder implementing the protocol for testing."""

    def __init__(self) -> None:
        self._is_recording = False
        self.beeps: list[tuple[int, float]] = []
        self.start_count = 0
        self.stop_count = 0
        self._wav_path: Path | None = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start_recording(self) -> None:
        self._is_recording = True
        self.start_count += 1

    def stop_recording(self) -> Path:
        self._is_recording = False
        self.stop_count += 1
        if self._wav_path is None:
            raise RuntimeError("No audio data recorded")
        return self._wav_path

    def play_beep(self, frequency: int = 440, duration: float = 0.1) -> None:
        self.beeps.append((frequency, duration))

    def set_wav_path(self, path: Path) -> None:
        self._wav_path = path


class FakeTranscriber:
    """In-memory Transcriber implementing the protocol for testing."""

    def __init__(self, result: str = "hello world") -> None:
        self.result = result
        self.transcribed_paths: list[Path] = []

    def transcribe_audio(self, audio_path: Path) -> str:
        self.transcribed_paths.append(audio_path)
        return self.result


class FakeTextProcessor:
    """In-memory TextProcessor implementing the protocol for testing."""

    def __init__(self) -> None:
        self.process_count = 0

    async def process_clipboard(self) -> None:
        self.process_count += 1


# Verify fakes satisfy protocols
assert isinstance(FakeRecorder(), Recorder)
assert isinstance(FakeTranscriber(), Transcriber)
assert isinstance(FakeTextProcessor(), TextProcessor)


@pytest.fixture
def fake_recorder() -> FakeRecorder:
    return FakeRecorder()


@pytest.fixture
def fake_transcriber() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def fake_text_processor() -> FakeTextProcessor:
    return FakeTextProcessor()


@pytest.fixture
def tmp_config(tmp_path: Path) -> WhisperConfig:
    """WhisperConfig with all directories pointing to tmp_path."""
    return WhisperConfig(
        audio=AudioConfig(),
        transcription=TranscriptionConfig(device_type="cpu", compute_type="float32"),
        claude=ClaudeConfig(),
        daemon=DaemonConfig(
            log_dir=tmp_path / "logs",
            config_dir=tmp_path / "config",
            tmp_dir=tmp_path / "tmp",
        ),
    )
