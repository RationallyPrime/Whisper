"""Tests for rt_whisper.audio — AudioCapture."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rt_whisper.audio import AudioCapture
from rt_whisper.config import AudioConfig


@pytest.fixture
def audio_capture(tmp_path: Path) -> AudioCapture:
    return AudioCapture(AudioConfig(), tmp_path)


class TestPlayBeep:
    @patch("rt_whisper.audio.sd")
    def test_beep_uses_config_samplerate(self, mock_sd: MagicMock, tmp_path: Path) -> None:
        config = AudioConfig(samplerate=96000)
        capture = AudioCapture(config, tmp_path)
        capture.play_beep(440, 0.1)

        mock_sd.play.assert_called_once()
        args = mock_sd.play.call_args
        assert args[0][1] == 96000  # samplerate argument

    @patch("rt_whisper.audio.sd")
    def test_beep_default_samplerate_48k(self, mock_sd: MagicMock, audio_capture: AudioCapture) -> None:
        audio_capture.play_beep()
        args = mock_sd.play.call_args
        assert args[0][1] == 48000

    @patch("rt_whisper.audio.sd")
    def test_beep_generates_correct_frequency(self, mock_sd: MagicMock, audio_capture: AudioCapture) -> None:
        audio_capture.play_beep(880, 0.1)
        beep_data = mock_sd.play.call_args[0][0]
        assert isinstance(beep_data, np.ndarray)
        assert len(beep_data) == int(48000 * 0.1)


class TestRecordingState:
    def test_not_recording_initially(self, audio_capture: AudioCapture) -> None:
        assert audio_capture.is_recording is False

    def test_stop_recording_when_not_recording_raises(self, audio_capture: AudioCapture) -> None:
        with pytest.raises(RuntimeError, match="Not currently recording"):
            audio_capture.stop_recording()
