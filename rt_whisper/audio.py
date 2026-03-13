"""Audio capture module implementing the Recorder protocol."""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """Audio recording and playback via sounddevice.

    Implements the Recorder protocol. Owns audio I/O — does not know
    about transcription or models.
    """

    def __init__(self, config: AudioConfig, tmp_dir: Path) -> None:
        self._config = config
        self._tmp_dir = tmp_dir
        self._tmp_dir.mkdir(exist_ok=True, parents=True)

        self._is_recording = False
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._recording_data: list[np.ndarray] = []
        self._recording_thread: threading.Thread | None = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start_recording(self) -> None:
        """Start recording audio in a background thread."""
        if self._is_recording:
            return
        self._is_recording = True
        self._recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self._recording_thread.start()
        logger.info("Recording started")
        self.play_beep(880, 0.05)

    def stop_recording(self) -> Path:
        """Stop recording and return the path to the WAV file.

        Blocks until the recording thread finishes and the audio is written.
        Caller is responsible for cleaning up the returned file.

        Raises:
            RuntimeError: If no audio data was recorded.
        """
        if not self._is_recording:
            raise RuntimeError("Not currently recording")

        self._is_recording = False
        if self._recording_thread is not None:
            self._recording_thread.join()
            self._recording_thread = None

        if not self._recording_data:
            raise RuntimeError("No audio data recorded")

        full_audio = np.concatenate(self._recording_data)
        self._recording_data.clear()

        wav_path = self._tmp_dir / "current_recording.wav"
        sf.write(wav_path, full_audio, self._config.samplerate)
        logger.info("Recording saved to %s", wav_path)
        return wav_path

    def play_beep(self, frequency: int = 440, duration: float = 0.1) -> None:
        """Play a short beep tone at the configured samplerate."""
        try:
            sr = self._config.samplerate
            t = np.linspace(0, duration, int(sr * duration))
            beep = np.sin(2 * np.pi * frequency * t) * 0.3
            sd.play(beep, sr)
            sd.wait()
        except Exception as e:
            logger.warning("Could not play beep: %s", e)

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Sounddevice input stream callback."""
        if status:
            logger.warning("Audio callback status: %s", status)
        self._audio_queue.put(indata.copy())

    def _record_audio(self) -> None:
        """Record audio chunks until ``_is_recording`` is cleared."""
        self._recording_data.clear()
        try:
            with sd.InputStream(
                samplerate=self._config.samplerate,
                channels=self._config.channels,
                dtype=self._config.dtype,
                device=self._config.device,
                blocksize=self._config.blocksize,
                callback=self._audio_callback,
            ):
                while self._is_recording:
                    audio_chunks: list[np.ndarray] = []
                    try:
                        while not self._audio_queue.empty():
                            audio_chunks.append(self._audio_queue.get_nowait())
                    except queue.Empty:
                        pass

                    if audio_chunks:
                        self._recording_data.append(np.concatenate(audio_chunks))
                    time.sleep(0.01)
        except Exception as e:
            logger.error("Recording error: %s", e)
            self._is_recording = False
