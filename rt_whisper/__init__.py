"""RT-Whisper: Realtime speech-to-text with Faster-Whisper."""

from ._version import __version__
from .transcriber import WhisperTranscriber
from . import prompts
from . import claude_client
from . import claude_setup

__all__ = ["WhisperTranscriber", "prompts", "claude_client", "claude_setup"] 
