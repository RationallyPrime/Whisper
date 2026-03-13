"""RT-Whisper: Realtime speech-to-text with Faster-Whisper."""

from ._version import __version__
from .audio import AudioCapture
from .config import WhisperConfig
from .transcriber import WhisperTranscriber

__all__ = ["__version__", "WhisperConfig", "AudioCapture", "WhisperTranscriber"]
