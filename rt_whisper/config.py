"""Pydantic v2 configuration models for RT-Whisper."""

from __future__ import annotations

import json
import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator


class DeviceType(StrEnum):
    """Inference device type."""

    CUDA = "cuda"
    CPU = "cpu"


class ComputeType(StrEnum):
    """Inference compute type."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"


class AudioConfig(BaseModel):
    """Audio capture configuration."""

    samplerate: int = 48000
    channels: int = 1
    dtype: str = "float32"
    device: int | None = None
    blocksize: int = 2048

    @field_validator("samplerate")
    @classmethod
    def samplerate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("samplerate must be positive")
        return v

    @field_validator("blocksize")
    @classmethod
    def blocksize_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("blocksize must be positive")
        return v


class TranscriptionConfig(BaseModel):
    """Whisper model and transcription configuration."""

    model: str = "large-v3"
    device_type: DeviceType = DeviceType.CUDA
    compute_type: ComputeType = ComputeType.FLOAT16
    beam_size: int = 3
    vad_filter: bool = True
    vad_parameters: dict[str, Any] = Field(default_factory=lambda: {"min_silence_duration_ms": 500})
    language: str = "en"
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".whisper_cache")

    @field_validator("beam_size")
    @classmethod
    def beam_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("beam_size must be >= 1")
        return v

    @model_validator(mode="after")
    def validate_cuda(self) -> TranscriptionConfig:
        if self.device_type == DeviceType.CUDA:
            try:
                import torch

                if not torch.cuda.is_available():
                    raise ValueError(
                        "device_type is 'cuda' but CUDA is not available. "
                        "Set device_type to 'cpu' or install CUDA."
                    )
            except ImportError:
                raise ValueError("device_type is 'cuda' but torch is not installed.") from None
        return self


class ClaudeConfig(BaseModel):
    """Claude API integration configuration."""

    enable: bool = False
    api_key: str | None = Field(default=None, alias="anthropic_api_key")
    model: str = "claude-sonnet-4-5-20250929"

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def load_api_key_from_env(self) -> ClaudeConfig:
        if self.enable and not self.api_key:
            env_path = Path.home() / "Whisper" / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            else:
                load_dotenv()
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        return self


class DaemonConfig(BaseModel):
    """Daemon runtime configuration."""

    log_dir: Path = Field(default_factory=lambda: Path.home() / ".whisper_logs")
    config_dir: Path = Field(default_factory=lambda: Path.home() / ".whisper_config")
    tmp_dir: Path = Field(default_factory=lambda: Path.home() / ".whisper_tmp")
    command_poll_interval: float = 0.1


# Legacy flat-key mapping: old flat config key -> nested path
_LEGACY_KEY_MAP: dict[str, tuple[str, str]] = {
    "enable_claude": ("claude", "enable"),
    "anthropic_api_key": ("claude", "api_key"),
}


class WhisperConfig(BaseModel):
    """Root configuration composing all sub-configs."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> WhisperConfig:
        """Load config from JSON file with legacy flat-key mapping and CLI overrides.

        Precedence: model defaults < config file < CLI overrides.

        Args:
            config_path: Path to config JSON file. Defaults to ~/.whisper_config/config.json.
            cli_overrides: Dict of dotted-path overrides, e.g. {"audio.device": 4}.
        """
        if config_path is None:
            config_path = Path.home() / ".whisper_config" / "config.json"

        raw: dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    raw = json.load(f)
            except Exception as e:
                logging.warning("Failed to load config from %s: %s", config_path, e)

        # Map legacy flat keys to nested structure
        nested = _flat_to_nested(raw)

        # Apply CLI overrides
        if cli_overrides:
            for dotted_key, value in cli_overrides.items():
                if value is None:
                    continue
                parts = dotted_key.split(".")
                target = nested
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value

        return cls.model_validate(nested)


def _flat_to_nested(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat config dict (possibly with legacy keys) to nested structure.

    Handles both already-nested dicts and flat legacy keys.
    """
    nested: dict[str, Any] = {}

    for key, value in raw.items():
        if key in _LEGACY_KEY_MAP:
            section, field = _LEGACY_KEY_MAP[key]
            nested.setdefault(section, {})[field] = value
        elif isinstance(value, dict):
            # Already nested
            nested.setdefault(key, {}).update(value)
        # Skip unknown flat keys (e.g. "available_commands")

    return nested
