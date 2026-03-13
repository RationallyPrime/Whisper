"""Whisper transcription module implementing the Transcriber protocol."""

from __future__ import annotations

import logging
from pathlib import Path

from faster_whisper import WhisperModel

from .config import TranscriptionConfig

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Speech-to-text using Faster-Whisper.

    Implements the Transcriber protocol. Owns the WhisperModel lifecycle.
    Does not know about audio capture, clipboard, or IPC.
    """

    def __init__(self, config: TranscriptionConfig) -> None:
        self._config = config
        config.cache_dir.mkdir(exist_ok=True, parents=True)

        try:
            self.model = WhisperModel(
                config.model,
                device=config.device_type.value,
                compute_type=config.compute_type.value,
                download_root=str(config.cache_dir),
            )
            logger.info(
                "Faster-Whisper model '%s' loaded on %s",
                config.model,
                config.device_type.value,
            )
        except Exception:
            logger.exception("Failed to load Faster-Whisper model")
            raise

    def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe an audio file and return the text.

        Args:
            audio_path: Path to WAV file to transcribe.

        Returns:
            Transcribed text, or "Transcription failed" on error.
        """
        try:
            segments, _info = self.model.transcribe(
                str(audio_path),
                language=self._config.language,
                task="transcribe",
                beam_size=self._config.beam_size,
                vad_filter=self._config.vad_filter,
                vad_parameters=self._config.vad_parameters,
            )
            text = " ".join(segment.text for segment in segments).strip()
            return text
        except Exception:
            logger.exception("Transcription error")
            return "Transcription failed"
