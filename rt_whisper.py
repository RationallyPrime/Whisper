import sys
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
from TTS.api import TTS
import transformers
from prompts import SYSTEM_PROMPTS
import re

# Silence all warnings and unnecessary logs
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress audioread warning
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress torch.load warning
warnings.filterwarnings("ignore")  # Suppress all warnings
transformers.logging.set_verbosity_error()  # Suppress transformer warnings
logging.getLogger("TTS.utils.synthesizer").setLevel(logging.ERROR)  # Suppress TTS info messages
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)  # Suppress attention mask warning
logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.ERROR)  # Suppress torch distributed warnings
logging.getLogger("numba").setLevel(logging.ERROR)  # Suppress numba warnings
logging.getLogger("matplotlib").setLevel(logging.ERROR)  # Suppress matplotlib warnings

# Configure logging for our app
log_dir = Path.home() / ".whisper_logs"
log_file = log_dir / "whisper.log"

# Debug print absolute paths
print(f"Log directory: {log_dir.absolute()}")
print(f"Log file: {log_file.absolute()}")

# Ensure directory exists with proper permissions
log_dir.mkdir(exist_ok=True, mode=0o755)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ],
    force=True  # Ensure our configuration takes precedence
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

@dataclass
class TTSConfig:
    """Modern TTS configuration settings"""
    model_dir: Path = Path.home() / ".tts_models"
    output_dir: Path = Path.home() / ".tts_output"
    temp_dir: Path = Path.home() / ".whisper_tmp"
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    sample_rate: int = 44100  # Increased from 22050
    reference_audio: Path = Path.home() / ".voice_references" / "Rupert_Degas_20_15.wav"
    language: str = "en"
    audio_quality: str = "high"  # Can be 'low', 'medium', 'high'

    def __post_init__(self):
        """Ensure all directories exist"""
        for dir_path in [self.model_dir, self.output_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)

class TTSEngine:
    """Modern Text-to-Speech engine using Coqui TTS"""
    def __init__(self, config: TTSConfig):
        self.config = config
        self.config.model_dir.mkdir(exist_ok=True)
        self.config.output_dir.mkdir(exist_ok=True)
        
        # Initialize TTS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tts = TTS(model_name=config.model_name, progress_bar=False)
            if torch.cuda.is_available():
                self.tts.to(self.device)
            logging.info(f"TTS model loaded on {self.device}")
        except Exception as e:
            logging.error(f"Failed to initialize TTS: {e}")
            raise

    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech from text and return path to audio file"""
        if output_path is None:
            output_path = self.config.output_dir / f"speech_{hash(text)}.wav"
        
        try:
            # Generate speech using XTTS with optimized settings
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=str(self.config.reference_audio),
                language=self.config.language,
                split_sentences=True,     # Better phrasing
                temperature=0.7,          # Controls variability (0.5-0.8 is good)
                length_penalty=1.0,       # Helps with pacing
                repetition_penalty=2.0,   # Reduces repetitive patterns
                top_k=50,                # More natural prosody
                enable_text_splitting=True  # Better handling of long texts
            )
            return output_path
        except Exception as e:
            logging.error(f"Speech synthesis failed: {e}")
            raise


class ClaudeClient:
    """Client for interacting with Anthropic's Claude API"""
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.system_prompts = SYSTEM_PROMPTS
    
    async def get_response(self, text: str) -> str:
        """Get response from Claude for the given text"""
        try:
            # Enhanced logging
            logging.info(f"Clipboard input: {text}")
            
            # Setup logging
            log_file = Path.home() / ".whisper_logs" / "whisper.log"
            logging.info(f"Request: {text[:200]}...")  # Log first 200 chars of request
            
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
            
            # Add more detailed logging
            logging.debug(f"System prompt selected: {system_prompt[:100] if system_prompt else 'None'}")
            logging.debug(f"Content being sent: {content[:100]}")
            
            # Create message with the correct format
            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=system_prompt if system_prompt else "",
                messages=[{
                    "role": "user",
                    "content": content  # Send the full text including prefix
                }]
            )
            
            response_text = response.content[0].text
            
            if text.lower().startswith("reformat this"):
                # Log any [Note: ...] content
                notes = re.findall(r'\[Note:.*?\]', response_text)
                for note in notes:
                    logging.info(f"Transcription note: {note}")
                
                # Remove [Note: ...] from the response
                response_text = re.sub(r'\[Note:.*?\]\s*', '', response_text)
                
                # Add signature
                response_text = response_text + "\n\nMessage dictated, not typed.\n— Hákon Freyr"
            
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
        # Initialize both configs
        self.audio_config = AudioConfig()
        self.tts_config = TTSConfig()
        
        # Use temp_file from tts_config for recordings
        self.temp_file = self.tts_config.temp_dir / "current_recording.wav"
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_data = []
        self.recording_thread: Optional[threading.Thread] = None
        
        # Ensure CUDA is available if requested
        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        
        # Load Whisper model
        try:
            self.model = whisper.load_model(model_name, device=device)
            logging.info(f"Whisper model loaded on device: {self.model.device}")
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            raise

        # Claude and TTS initialization
        self.config_path = Path.home() / ".whisper_config" / "config.json"
        self.config_dir = self.config_path.parent
        self.config_dir.mkdir(exist_ok=True)
        self.claude_config = self._load_config()
        
        # Initialize Claude and TTS if configured
        self.claude_client = None
        self.tts_engine = None
        if self.claude_config.get("enable_claude_tts", False):
            self._initialize_claude_tts()

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
    
    def _initialize_claude_tts(self):
        """Initialize Claude client and TTS engine"""
        try:
            self.claude_client = ClaudeClient(self.claude_config["anthropic_api_key"])
            self.tts_engine = TTSEngine(TTSConfig())
            logging.info("Claude and TTS components initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Claude/TTS: {e}")

    def audio_callback(self, indata: np.ndarray, frames: int, time: sd.CallbackFlags, status: sd.CallbackFlags) -> None:
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
                callback=self.audio_callback
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
                    language='en',
                    task='transcribe'
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
                # Fix: Use audio_config instead of config
                sf.write(self.temp_file, full_audio, self.audio_config.samplerate)
                
                result = self.transcribe_audio(self.temp_file)
                transcribed_text = result.get('text', '').strip()
                
                if transcribed_text:
                    pyperclip.copy(transcribed_text)
                    logging.info(f"Transcribed and copied to clipboard: {transcribed_text}")
                else:
                    logging.warning("No text transcribed")
                    
            except Exception as e:
                logging.error(f"Processing error: {e}")
            finally:
                # Cleanup using temp_file from tts_config
                if self.temp_file.exists():
                    self.temp_file.unlink()
            print("\a")  # System beep

    async def process_with_claude_tts(self):
        try:
            text = pyperclip.paste()
            if not text:
                logging.warning("No text in clipboard")
                return
            
            text_lower = text.lower().strip()
            
            # Check if it's an explain command
            is_explain_command = text_lower.startswith(("explain this:", "explain this.", "explain this", "explain this colon"))
            
            # Check for other keyword commands
            is_other_keyword_command = (
                text_lower.startswith(("promptify this:", "promptify this.", "promptify this"))
                or text_lower.startswith(("reformat this:", "reformat this.", "reformat this"))
                or text_lower.startswith(("implement this:", "implement this.", "implement this"))
                or text_lower.startswith(("command line this:", "command line this.", "command line this"))
                or text_lower.startswith("promptify this colon")
                or text_lower.startswith("reformat this colon")
                or text_lower.startswith("implement this colon")
                or text_lower.startswith("command line this colon")
            )
            
            # Get Claude's response
            response = await self.claude_client.get_response(text)
            if not response:
                return
            
            # Copy response to clipboard
            pyperclip.copy(response)
            logging.info("Response copied to clipboard")
            
            # Only synthesize speech for explain commands
            if is_explain_command:
                logging.info("Explain command detected - synthesizing speech...")
                # Synthesize speech
                audio_path = self.tts_engine.synthesize(response)
                
                # Play audio with enhanced quality
                logging.info("Playing audio response...")
                data, samplerate = sf.read(str(audio_path))
                
                if np.abs(data).max() > 0:
                    data = data / np.abs(data).max() * 0.9
                
                sd.default.samplerate = samplerate
                sd.default.channels = 1
                sd.default.dtype = np.float32
                sd.default.latency = 'low'
                sd.default.blocksize = 2048
                
                sd.play(data, samplerate)
                sd.wait()
            elif is_other_keyword_command:
                logging.info("Other keyword command detected - skipping speech synthesis")
                
        except Exception as e:
            logging.error(f"Claude/TTS processing error: {e}")

    def on_press(self, key: keyboard.Key) -> None:
        """Handle keyboard events for both Whisper and Claude+TTS"""
        try:
            if key == keyboard.Key.f12 and not self.is_recording:
                self.start_recording()
            elif key == keyboard.Key.f11 and self.is_recording:
                self.stop_recording()
            elif key == keyboard.Key.f10 and self.claude_config.get("enable_claude_tts"):
                logging.info("Processing with Claude and TTS...")
                asyncio.run(self.process_with_claude_tts())
        except Exception as e:
            logging.error(f"Keyboard handling error: {e}")

    def run(self) -> None:
        """Main execution loop"""
        if self.claude_config.get("enable_claude_tts"):
            logging.info("""
        Commands:
        - Press 'F12' to start recording
        - Press 'F11' to stop recording
        - Press 'F10' for Claude+TTS
        - Start speech with 'promptify this:' to create LLM prompts
        - Start speech with 'reformat this:' to clean up transcribed text
        - Press 'Ctrl+C' to exit
        """)
        else:
            logging.info("Press 'F12' to start recording, 'F11' to stop recording, 'Ctrl+C' to exit")
        
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
