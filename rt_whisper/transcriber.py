"""Whisper Transcriber using Faster-Whisper for real-time transcription."""

from pathlib import Path
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import queue
import warnings
import soundfile as sf
import pyperclip
# pynput imported lazily only in legacy hotkey mode
import threading
import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
import os
import json
import asyncio
import time
from dotenv import load_dotenv

# Optional import for Claude integration
try:
    from .claude_client import ClaudeClient
except ImportError:
    ClaudeClient = None

# Silence all warnings and unnecessary logs
warnings.filterwarnings("ignore")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# Configure logging
log_dir = Path.home() / ".whisper_logs"
log_file = log_dir / "rt_whisper.log"
log_dir.mkdir(exist_ok=True, mode=0o755)

# Configure logging with RotatingFileHandler
# First reset the root logger to handle Python 3.12 deprecation of force=True
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
    ]
)

# Directory setup
TEMP_DIR = Path.home() / ".whisper_tmp"
TEMP_DIR.mkdir(exist_ok=True)

MODEL_CACHE_DIR = Path.home() / ".whisper_cache"
MODEL_CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class AudioConfig:
    """Audio configuration settings for recording."""

    samplerate: int = 44100
    channels: int = 1
    dtype: np.dtype = np.float32
    device: Optional[int] = None
    blocksize: int = 2048 * 8


class WhisperTranscriber:
    """Whisper transcriber using Faster-Whisper for real-time transcription."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda", 
        compute_type: str = "float16",
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the transcriber.
        
        Args:
            model_name: The name of the Whisper model to use
            device: The device to use for inference ('cuda' or 'cpu')
            compute_type: The compute type to use ('float16', 'int8', 'int8_float16')
            cache_dir: The directory to cache the model in
        """
        # Initialize audio configuration
        self.audio_config = AudioConfig()
        self.temp_file = TEMP_DIR / "current_recording.wav"
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_data: List[np.ndarray] = []
        self.recording_thread: Optional[threading.Thread] = None
        
        # Set up the cache directory
        download_root = str(MODEL_CACHE_DIR)
        if cache_dir:
            download_root = str(cache_dir)
        
        # Load model
        try:
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
            )
            logging.info(f"Faster-Whisper model '{model_name}' loaded on {device}")
        except Exception as e:
            logging.error(f"Failed to load Faster-Whisper model: {e}")
            raise
            
        # Load configuration for Claude integration
        self.config = self._load_config()
        
        # Initialize Claude client if enabled
        self.claude_client = None
        if self.config.get("enable_claude"):
            self._initialize_claude()
            
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        config_dir = Path.home() / ".whisper_config"
        config_path = config_dir / "config.json"
        
        if not config_path.exists():
            return {"enable_claude": False}
            
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Config load error: {e}")
            return {"enable_claude": False}
            
    def _initialize_claude(self):
        """Initialize Claude client."""
        if ClaudeClient is None:
            logging.error("Claude client module not available")
            return
            
        try:
            api_key = self.config.get("anthropic_api_key")
            if not api_key:
                # Try loading from .env file
                env_path = Path.home() / "Whisper" / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
                else:
                    load_dotenv()
                api_key = os.getenv("ANTHROPIC_API_KEY")
                
            if api_key:
                self.claude_client = ClaudeClient(api_key)
                logging.info("Claude client initialized successfully")
            else:
                logging.error("No Anthropic API key found")
        except Exception as e:
            logging.error(f"Failed to initialize Claude: {e}")

    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: sd.CallbackFlags,
        status: sd.CallbackFlags,
    ) -> None:
        """Process audio input in real-time."""
        if status:
            logging.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())

    def record_audio(self) -> None:
        """Record audio in a separate thread."""
        self.recording_data.clear()

        try:
            with sd.InputStream(
                samplerate=self.audio_config.samplerate,
                channels=self.audio_config.channels,
                dtype=self.audio_config.dtype,
                device=self.audio_config.device,
                blocksize=self.audio_config.blocksize,
                callback=self.audio_callback,
            ):
                while self.is_recording:
                    audio_chunks = []
                    try:
                        while not self.audio_queue.empty():
                            audio_chunks.append(self.audio_queue.get_nowait())
                    except queue.Empty:
                        pass

                    if audio_chunks:
                        self.recording_data.append(np.concatenate(audio_chunks))
                    time.sleep(0.01)  # Reduce CPU churn
        except Exception as e:
            logging.error(f"Recording error: {e}")
            self.is_recording = False

    def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe audio file using Faster-Whisper.
        
        Args:
            audio_path: Path to the audio file to transcribe
            
        Returns:
            The transcribed text
        """
        try:
            # Transcribe with Faster-Whisper
            segments, info = self.model.transcribe(
                str(audio_path),
                language="en",
                task="transcribe",
            )
            
            # Collect all segment texts
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text + " "
            
            return transcribed_text.strip()
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return "Transcription failed"

    def start_recording(self) -> None:
        """Start recording in a new thread."""
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
            logging.info("Recording started")
            print("\a")  # System beep

    def stop_recording(self) -> None:
        """Stop recording and transcribe."""
        if self.is_recording:
            self.is_recording = False
            if self.recording_thread:
                self.recording_thread.join()

            if not self.recording_data:
                logging.warning("No audio data recorded")
                return

            try:
                full_audio = np.concatenate(self.recording_data)
                sf.write(self.temp_file, full_audio, self.audio_config.samplerate)

                transcribed_text = self.transcribe_audio(self.temp_file)

                if transcribed_text:
                    pyperclip.copy(transcribed_text)
                    logging.info(f"Transcribed and copied to clipboard: {transcribed_text}")
                    self._play_beep(660, 0.1)  # Success beep
                else:
                    logging.warning("No text transcribed")
                    self._play_beep(330, 0.2)  # Lower pitch for no text

            except Exception as e:
                logging.error(f"Processing error: {e}")
                self._play_beep(220, 0.3)  # Error beep
            finally:
                # Cleanup temp file
                if self.temp_file.exists():
                    self.temp_file.unlink()
            
    def _play_beep(self, frequency: int = 440, duration: float = 0.1):
        """Play a beep sound."""
        try:
            t = np.linspace(0, duration, int(44100 * duration))
            beep = np.sin(2 * np.pi * frequency * t) * 0.3
            sd.play(beep, 44100)
            sd.wait()
        except Exception as e:
            logging.warning(f"Could not play beep: {e}")

    def _run_async_safely(self, coro):
        """Run an async coroutine safely, compatible with Python 3.12+.
        
        Args:
            coro: The coroutine to run
        """
        try:
            # In Python 3.12+, asyncio.run() creates a new event loop when needed
            # and handles the closure of the loop more reliably
            import sys
            if sys.version_info >= (3, 12):
                # Python 3.12+ approach - simpler and more reliable
                return asyncio.run(coro)
            else:
                # Backwards compatibility for older Python versions
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_closed():
                        raise RuntimeError("Event loop is closed")
                    # We're in an existing loop context, need custom handling
                    future = asyncio.ensure_future(coro, loop=loop)
                    return loop.run_until_complete(future)
                except RuntimeError:
                    # No running event loop, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(coro)
                    finally:
                        try:
                            loop.close()
                        except Exception as e:
                            logging.warning(f"Error closing event loop: {e}")
        except Exception as e:
            logging.error(f"Error running async operation: {e}")
            return None
            
    async def process_with_claude(self, prefix: Optional[str] = None):
        """Process clipboard content with Claude.
        
        Args:
            prefix: Optional prefix to add to the clipboard content
        """
        if not self.claude_client:
            logging.warning("Claude is not enabled or properly configured")
            return
            
        try:
            # If prefix is provided, modify the clipboard content
            if prefix:
                text = pyperclip.paste()
                if text:
                    pyperclip.copy(f"{prefix}: {text}")
                    
            # Process the clipboard content
            await self.claude_client.process_clipboard()
        except Exception as e:
            logging.error(f"Claude processing error: {e}")

    def check_for_commands(self):
        """Check for command files and execute them."""
        command_file = Path.home() / ".whisper_logs" / "command.json"
        if not command_file.exists():
            return
        
        try:
            with open(command_file, "r") as f:
                command_data = json.load(f)
            
            command = command_data.get("command")
            timestamp = command_data.get("timestamp", 0)
            
            if time.time() - timestamp > 5:
                command_file.unlink()
                return
            
            logging.info(f"Processing command: {command}")
            
            if command == "start_recording" and not self.is_recording:
                self.start_recording()
            elif command == "stop_recording" and self.is_recording:
                self.stop_recording()
            elif command == "process_clipboard" and self.claude_client:
                self._run_async_safely(self.process_with_claude())
            elif command == "explain_clipboard" and self.claude_client:
                text = pyperclip.paste()
                if text:
                    pyperclip.copy(f"explain this: {text}")
                    self._run_async_safely(self.process_with_claude())
            elif command == "summarize_clipboard" and self.claude_client:
                text = pyperclip.paste()
                if text:
                    pyperclip.copy(f"summarize this: {text}")
                    self._run_async_safely(self.process_with_claude())
            elif command == "promptify_clipboard" and self.claude_client:
                text = pyperclip.paste()
                if text:
                    pyperclip.copy(f"promptify this: {text}")
                    self._run_async_safely(self.process_with_claude())
            elif command == "reformat_clipboard" and self.claude_client:
                text = pyperclip.paste()
                if text:
                    pyperclip.copy(f"reformat this: {text}")
                    self._run_async_safely(self.process_with_claude())
            elif command == "implement_clipboard" and self.claude_client:
                text = pyperclip.paste()
                if text:
                    pyperclip.copy(f"implement this: {text}")
                    self._run_async_safely(self.process_with_claude())
            elif command == "command_clipboard" and self.claude_client:
                text = pyperclip.paste()
                if text:
                    pyperclip.copy(f"command line this: {text}")
                    self._run_async_safely(self.process_with_claude())
            elif command == "translate_clipboard" and self.claude_client:
                text = pyperclip.paste()
                if text:
                    pyperclip.copy(f"translate this into English: {text}")
                    self._run_async_safely(self.process_with_claude())
            
            command_file.unlink()
        except Exception as e:
            logging.error(f"Error processing command: {e}")
            command_file.unlink()

    def run_command_mode(self):
        """Run in command mode, waiting for commands."""
        logging.info("RT-Whisper started in command mode.")
        
        try:
            while True:
                self.check_for_commands()
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            if self.is_recording:
                self.stop_recording()

    def run(self, command_mode: bool = True) -> None:
        """Main execution loop. Runs in command mode (file-based IPC)."""
        self.run_command_mode() 
