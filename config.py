from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
@dataclass
class AudioConfig:
    """Audio configuration settings"""
    samplerate: int = 16000
    channels: int = 1
    dtype: np.dtype = np.float32
    device: Optional[int] = None
    blocksize: int = 1024 * 4

@dataclass
class TTSConfig:
    """Modern TTS configuration settings"""
    model_dir: Path = Path.home() / ".tts_models"
    output_dir: Path = Path.home() / ".tts_output"
    model_name: str = "tts_models/en/vctk/vits"
    sample_rate: int = 22050
    speaker_id: Optional[str] = None
    language: str = "en"