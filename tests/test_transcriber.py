"""Tests for rt_whisper.transcriber — WhisperTranscriber."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rt_whisper.config import TranscriptionConfig
from rt_whisper.transcriber import WhisperTranscriber


class TestTranscribeAudio:
    @patch("rt_whisper.transcriber.WhisperModel")
    def test_passes_vad_and_beam_params(self, MockModel: MagicMock, tmp_path: Path) -> None:
        config = TranscriptionConfig(
            model="tiny",
            device_type="cpu",
            compute_type="float32",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            language="en",
            cache_dir=tmp_path / "cache",
        )

        mock_model_instance = MockModel.return_value
        mock_segment = MagicMock()
        mock_segment.text = "hello"
        mock_model_instance.transcribe.return_value = ([mock_segment], MagicMock())

        transcriber = WhisperTranscriber(config)
        wav = tmp_path / "test.wav"
        wav.write_text("fake")

        result = transcriber.transcribe_audio(wav)

        assert result == "hello"
        mock_model_instance.transcribe.assert_called_once_with(
            str(wav),
            language="en",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )

    @patch("rt_whisper.transcriber.WhisperModel")
    def test_transcription_error_returns_failure_string(self, MockModel: MagicMock, tmp_path: Path) -> None:
        config = TranscriptionConfig(
            model="tiny",
            device_type="cpu",
            compute_type="float32",
            cache_dir=tmp_path / "cache",
        )
        mock_model_instance = MockModel.return_value
        mock_model_instance.transcribe.side_effect = RuntimeError("boom")

        transcriber = WhisperTranscriber(config)
        result = transcriber.transcribe_audio(tmp_path / "bad.wav")
        assert result == "Transcription failed"

    @patch("rt_whisper.transcriber.WhisperModel")
    def test_model_init_uses_config(self, MockModel: MagicMock, tmp_path: Path) -> None:
        config = TranscriptionConfig(
            model="base",
            device_type="cpu",
            compute_type="int8",
            cache_dir=tmp_path / "cache",
        )
        WhisperTranscriber(config)
        MockModel.assert_called_once_with(
            "base",
            device="cpu",
            compute_type="int8",
            download_root=str(tmp_path / "cache"),
        )
