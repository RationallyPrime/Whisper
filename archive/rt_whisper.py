from pathlib import Path
import whisper
import sounddevice as sd
import numpy as np
import queue
import warnings
import soundfile as sf
import pyperclip
from pynput import keyboard
import threading
import torch
from typing import Optional
from dataclasses import dataclass
import logging
import os
from anthropic import AsyncAnthropic
import asyncio
import json
import transformers
from prompts import SYSTEM_PROMPTS
import re
from logging.handlers import RotatingFileHandler
# Silence all warnings and unnecessary logs
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress audioread warning
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress torch.load warning
warnings.filterwarnings("ignore")  # Suppress all warnings
transformers.logging.set_verbosity_error()  # Suppress transformer warnings
logging.getLogger("transformers.generation.utils").setLevel(
    logging.ERROR
)  # Suppress attention mask warning
logging.getLogger("torch.distributed.distributed_c10d").setLevel(
    logging.ERROR
)  # Suppress torch distributed warnings
logging.getLogger("numba").setLevel(logging.ERROR)  # Suppress numba warnings
logging.getLogger("matplotlib").setLevel(logging.ERROR)  # Suppress matplotlib warnings

# Configure logging for our app
log_dir = Path.home() / ".whisper_logs"
log_file = log_dir / "whisper.log"

# Ensure directory exists with proper permissions
log_dir.mkdir(exist_ok=True, mode=0o755)

# Configure logging to write only to a file with RotatingFileHandler


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
    ],
    force=True,  # Ensure our configuration takes precedence
)

# Add a test log message at startup
logging.info("RT-Whisper service started")

# Directory setup
output_dir = Path.home() / ".whisper_tmp"
output_dir.mkdir(exist_ok=True)

model_cache_dir = Path.home() / ".whisper_cache"
os.environ["WHISPER_CACHE_DIR"] = str(model_cache_dir)


@dataclass
class AudioConfig:
    """Audio configuration settings for recording"""

    samplerate: int = 44100  # Increased from 16000
    channels: int = 1
    dtype: np.dtype = np.float32
    device: Optional[int] = None
    blocksize: int = 2048 * 4  # Increased buffer size


class ClaudeClient:
    """Client for interacting with Anthropic's Claude API"""

    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.system_prompts = (
            SYSTEM_PROMPTS  # Ensure SYSTEM_PROMPTS includes "translate" and "summarize"
        )

    async def get_response(self, text: str) -> str:
        """Get response from Claude for the given text"""
        try:
            # Enhanced logging
            logging.info(f"Clipboard input: {text}")

            # Determine which system prompt to use but keep the prefix in content
            system_prompt = None
            content = text  # Keep the full text including prefix

            if text.lower().startswith("promptify this"):
                system_prompt = self.system_prompts["promptify"]
            elif text.lower().startswith("reformat this"):
                system_prompt = self.system_prompts["reformat"]
            elif text.lower().startswith("implement this"):
                system_prompt = self.system_prompts["implement"]
            elif text.lower().startswith("command line this"):
                system_prompt = self.system_prompts["command"]
            elif text.lower().startswith("explain this"):
                system_prompt = self.system_prompts["explain"]
            elif text.lower().startswith("translate this into"):
                system_prompt = self.system_prompts["translate"]
            elif text.lower().startswith("summarize this"):
                system_prompt = self.system_prompts.get(
                    "summarize", "Please provide a summary of the following text."
                )

            # Add more detailed logging
            logging.debug(
                f"System prompt selected: {system_prompt[:100] if system_prompt else 'None'}"
            )
            logging.debug(f"Content being sent: {content[:100]}")

            # Log the outgoing request details
            logging.info("Sending request to Claude API:")
            logging.info("- Model: claude-3-5-sonnet-20241022")
            logging.info(
                f"- System prompt: {system_prompt[:100] if system_prompt else 'None'}..."
            )
            logging.info(f"- Content: {content[:100]}...")

            # Create message with the correct format
            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=system_prompt if system_prompt else "",
                messages=[
                    {
                        "role": "user",
                        "content": content,  # Send the full text including prefix
                    }
                ],
            )

            response_text = response.content[0].text
            logging.info(
                f"Received response from Claude API ({len(response_text)} chars)"
            )
            logging.info(f"Claude response: {response_text}")

            if text.lower().startswith("reformat this:"):
                # Log any [Note: ...] content
                notes = re.findall(r"\[Note:.*?\]", response_text)
                for note in notes:
                    logging.info(f"Transcription note: {note}")

                # Remove [Note: ...] from the response
                response_text = re.sub(r"\[Note:.*?\]\s*", "", response_text)

                # Add signature
                response_text = (
                    response_text + "\n\nMessage dictated, not typed.\n— Hákon Freyr"
                )

            pyperclip.copy(response_text)
            logging.info(f"Claude response: {response_text}")

            # Play a quiet beep
            sd.default.samplerate = 44100
            duration = 0.1  # seconds
            frequency = 440  # Hz (A4 note)
            t = np.linspace(0, duration, int(44100 * duration))
            beep = np.sin(2 * np.pi * frequency * t) * 0.3
            sd.play(beep, 44100)
            sd.wait()

            # Force flush the logs
            for handler in logging.getLogger().handlers:
                handler.flush()

            return response_text
        except Exception as e:
            error_msg = f"Claude API error: {e}"
            logging.error(error_msg)
            return f"Sorry, I encountered an error: {str(e)}"


class WhisperTranscriber:
    def __init__(self, model_name: str = "turbo", device: str = "cuda"):
        # Initialize audio and Whisper model
        self.audio_config = AudioConfig()
        self.temp_file = Path.home() / ".whisper_tmp" / "current_recording.wav"
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_data = []
        self.recording_thread: Optional[threading.Thread] = None

        # Load Whisper model
        try:
            self.model = whisper.load_model(model_name, device=device)
            logging.info(f"Whisper model loaded on device: {self.model.device}")
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            raise

        # Load configuration
        self.config_path = Path.home() / ".whisper_config" / "config.json"
        self.config_dir = self.config_path.parent
        self.config_dir.mkdir(exist_ok=True)
        self.claude_config = self._load_config()

        # Initialize double-press detection for F10
        self.last_f10_press = 0
        self.F10_DOUBLE_PRESS_THRESHOLD = 0.5  # seconds

        # Initialize Claude if enabled
        if self.claude_config.get("enable_claude_tts"):
            self._initialize_claude()

    def _load_config(self) -> dict:
        """Load configuration from file"""
        if not self.config_path.exists():
            return {"enable_claude_tts": False}

        try:
            with open(self.config_path) as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Config load error: {e}")
            return {"enable_claude_tts": False}

    def _initialize_claude(self):
        """Initialize Claude client"""
        try:
            self.claude_client = ClaudeClient(self.claude_config["anthropic_api_key"])
            logging.info("Claude client initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Claude: {e}")

    def audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: sd.CallbackFlags,
        status: sd.CallbackFlags,
    ) -> None:
        """Process audio input in real-time"""
        if status:
            logging.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())

    def record_audio(self) -> None:
        """Record audio in a separate thread"""
        self.recording_data.clear()
        logging.info("Recording started...")

        try:
            with sd.InputStream(
                samplerate=self.audio_config.samplerate,
                channels=self.audio_config.channels,
                dtype=self.audio_config.dtype,
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
        except Exception as e:
            logging.error(f"Recording error: {e}")
            self.is_recording = False

    def transcribe_audio(self, audio_path: Path) -> dict:
        """Transcribe audio file using Whisper"""
        try:
            with torch.inference_mode():
                result = self.model.transcribe(
                    str(audio_path),
                    fp16=torch.cuda.is_available(),
                    language="en",
                    task="transcribe",
                )
            return result
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return {"text": "Transcription failed"}

    def start_recording(self) -> None:
        """Start recording in a new thread"""
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
            logging.info("Recording started")
            print("\a")  # System beep

    def stop_recording(self) -> None:
        """Stop recording and transcribe"""
        if self.is_recording:
            self.is_recording = False
            self.recording_thread.join()

            if not self.recording_data:
                logging.warning("No audio data recorded")
                return

            try:
                full_audio = np.concatenate(self.recording_data)
                sf.write(self.temp_file, full_audio, self.audio_config.samplerate)

                result = self.transcribe_audio(self.temp_file)
                transcribed_text = result.get("text", "").strip()

                if transcribed_text:
                    pyperclip.copy(transcribed_text)
                    logging.info(
                        f"Transcribed and copied to clipboard: {transcribed_text}"
                    )
                else:
                    logging.warning("No text transcribed")

            except Exception as e:
                logging.error(f"Processing error: {e}")
            finally:
                # Cleanup using temp_file from tts_config
                if self.temp_file.exists():
                    self.temp_file.unlink()
            print("\a")  # System beep

    async def process_with_claude(self):
        if not self.claude_config.get("enable_claude_tts"):
            logging.warning("Claude is not enabled.")
            return

        try:
            text = pyperclip.paste()
            if not text:
                logging.warning("No text in clipboard")
                return

            # Get Claude's response
            response = await self.claude_client.get_response(text)
            if not response:
                return

            # Copy response to clipboard
            pyperclip.copy(response)
            logging.info("Response copied to clipboard")

        except Exception as e:
            logging.error(f"Claude processing error: {e}")

    def on_press(self, key: keyboard.Key) -> None:
        """Handle keyboard events for both Whisper and Claude"""
        try:
            # Get the current state of modifier keys
            ctrl_pressed = keyboard.Controller().pressed(keyboard.Key.ctrl)
            alt_pressed = keyboard.Controller().pressed(keyboard.Key.alt)
            shift_pressed = keyboard.Controller().pressed(keyboard.Key.shift)

            if (
                ctrl_pressed and alt_pressed and shift_pressed
            ):  # Only process if all modifiers are held
                if key == keyboard.Key.f12 and not self.is_recording:
                    self.start_recording()
                elif key == keyboard.Key.f11 and self.is_recording:
                    self.stop_recording()
                elif key == keyboard.Key.f10:
                    logging.info(
                        "F10 pressed - Processing with Claude for general explanation"
                    )
                    text = pyperclip.paste()
                    if text:
                        pyperclip.copy(f"explain this: {text}")
                    asyncio.run(self.process_with_claude())
                elif key == keyboard.Key.f9:
                    logging.info("F9 pressed - Processing with Claude for reformatting")
                    text = pyperclip.paste()
                    if text:
                        pyperclip.copy(f"reformat this: {text}")
                    asyncio.run(self.process_with_claude())
                elif key == keyboard.Key.f8:
                    logging.info(
                        "F8 pressed - Processing with Claude for prompt creation"
                    )
                    text = pyperclip.paste()
                    if text:
                        pyperclip.copy(f"promptify this: {text}")
                    asyncio.run(self.process_with_claude())
                elif key == keyboard.Key.f7:
                    logging.info(
                        "F7 pressed - Processing with Claude for code implementation"
                    )
                    text = pyperclip.paste()
                    if text:
                        pyperclip.copy(f"implement this: {text}")
                    asyncio.run(self.process_with_claude())
                elif key == keyboard.Key.f6:
                    logging.info(
                        "F6 pressed - Processing with Claude for command line generation"
                    )
                    text = pyperclip.paste()
                    if text:
                        pyperclip.copy(f"command line this: {text}")
                    asyncio.run(self.process_with_claude())
                elif key == keyboard.Key.f5:
                    logging.info(
                        "F5 pressed - Processing with Claude for summarization"
                    )
                    text = pyperclip.paste()
                    if text:
                        pyperclip.copy(f"summarize this: {text}")
                    asyncio.run(self.process_with_claude())
        except Exception as e:
            logging.error(f"Keyboard handling error: {e}")

    def run(self) -> None:
        """Main execution loop"""
        if self.claude_config.get("enable_claude_tts"):
            logging.info("""
Commands (All require Ctrl+Alt+Shift):
- F12: Start recording
- F11: Stop recording
- F10: Process clipboard with Claude (Explain)
- F9: Reformat transcribed text
- F8: Create LLM prompts
- F7: Generate code implementation
- F6: Generate terminal commands
- F5: Summarize text
- Press 'Ctrl+C' to exit
""")
        else:
            logging.info("""
Commands (All require Ctrl+Alt+Shift):
- F12: Start recording
- F11: Stop recording
- Press 'Ctrl+C' to exit
""")

        with keyboard.Listener(on_press=self.on_press) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                logging.info("Shutting down...")
                if self.is_recording:
                    self.stop_recording()


if __name__ == "__main__":
    transcriber = WhisperTranscriber(model_name="turbo", device="cuda")
    transcriber.run()
