"""Tests for rt_whisper.config — Pydantic v2 configuration models."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from rt_whisper.config import (
    AudioConfig,
    ClaudeConfig,
    ComputeType,
    DeviceType,
    TranscriptionConfig,
    WhisperConfig,
)


class TestAudioConfig:
    def test_defaults(self) -> None:
        cfg = AudioConfig()
        assert cfg.samplerate == 48000
        assert cfg.channels == 1
        assert cfg.dtype == "float32"
        assert cfg.device is None
        assert cfg.blocksize == 2048

    def test_invalid_samplerate(self) -> None:
        with pytest.raises(ValidationError):
            AudioConfig(samplerate=0)

    def test_invalid_blocksize(self) -> None:
        with pytest.raises(ValidationError):
            AudioConfig(blocksize=-1)

    @given(st.integers(min_value=1, max_value=384000))
    def test_valid_samplerates(self, sr: int) -> None:
        cfg = AudioConfig(samplerate=sr)
        assert cfg.samplerate == sr


class TestTranscriptionConfig:
    def test_defaults(self) -> None:
        cfg = TranscriptionConfig()
        assert cfg.model == "large-v3"
        assert cfg.device_type == DeviceType.CUDA
        assert cfg.compute_type == ComputeType.FLOAT16
        assert cfg.beam_size == 3
        assert cfg.vad_filter is True
        assert cfg.vad_parameters == {"min_silence_duration_ms": 500}
        assert cfg.language == "en"

    def test_invalid_beam_size(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptionConfig(beam_size=0)

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_validation_fails(self, _mock: object) -> None:
        with pytest.raises(ValidationError, match="CUDA is not available"):
            TranscriptionConfig(device_type="cuda")

    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_validation_passes(self, _mock: object) -> None:
        cfg = TranscriptionConfig(device_type="cuda")
        assert cfg.device_type == DeviceType.CUDA

    def test_cpu_no_cuda_check(self) -> None:
        cfg = TranscriptionConfig(device_type="cpu", compute_type="float32")
        assert cfg.device_type == DeviceType.CPU


class TestClaudeConfig:
    def test_defaults(self) -> None:
        cfg = ClaudeConfig()
        assert cfg.enable is False
        assert cfg.api_key is None

    def test_api_key_alias(self) -> None:
        cfg = ClaudeConfig(anthropic_api_key="sk-test")
        assert cfg.api_key == "sk-test"


class TestWhisperConfig:
    def test_defaults(self) -> None:
        cfg = WhisperConfig()
        assert cfg.audio.samplerate == 48000
        assert cfg.transcription.vad_filter is True
        assert cfg.claude.enable is False

    def test_load_nonexistent_config(self, tmp_path: Path) -> None:
        cfg = WhisperConfig.load(config_path=tmp_path / "nope.json")
        assert cfg.audio.samplerate == 48000

    def test_load_partial_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"audio": {"samplerate": 44100}}))
        cfg = WhisperConfig.load(config_path=config_file)
        assert cfg.audio.samplerate == 44100
        assert cfg.audio.blocksize == 2048  # default preserved

    def test_load_full_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "audio": {"samplerate": 96000, "channels": 2},
                    "transcription": {
                        "model": "tiny",
                        "device_type": "cpu",
                        "compute_type": "float32",
                    },
                    "claude": {"enable": False},
                }
            )
        )
        cfg = WhisperConfig.load(config_path=config_file)
        assert cfg.audio.samplerate == 96000
        assert cfg.audio.channels == 2
        assert cfg.transcription.model == "tiny"

    def test_load_invalid_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text("not json")
        # Should fall back to defaults, not crash
        cfg = WhisperConfig.load(config_path=config_file)
        assert cfg.audio.samplerate == 48000

    def test_cli_overrides(self, tmp_path: Path) -> None:
        cfg = WhisperConfig.load(
            config_path=tmp_path / "nope.json",
            cli_overrides={"audio.device": 4, "transcription.model": "tiny"},
        )
        assert cfg.audio.device == 4
        assert cfg.transcription.model == "tiny"

    def test_backward_compat_flat_keys(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "enable_claude": True,
                    "anthropic_api_key": "sk-legacy",
                }
            )
        )
        cfg = WhisperConfig.load(config_path=config_file)
        assert cfg.claude.enable is True
        assert cfg.claude.api_key == "sk-legacy"

    @given(st.integers(min_value=1, max_value=384000))
    def test_hypothesis_samplerate_roundtrip(self, sr: int) -> None:
        cfg = WhisperConfig(audio=AudioConfig(samplerate=sr))
        assert cfg.audio.samplerate == sr
