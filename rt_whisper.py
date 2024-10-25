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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Directory setup
output_dir = Path.home() / ".whisper_tmp"
output_dir.mkdir(exist_ok=True)

model_cache_dir = Path.home() / ".whisper_cache"
os.environ["WHISPER_CACHE_DIR"] = str(model_cache_dir)

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
    speaker_id: str = "p273"  # Added default speaker ID
    #language: str = "en"



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
            # Generate speech
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker=self.config.speaker_id,  # Make sure this line is present
                #language=self.config.language
            )
            return output_path
        except Exception as e:
            logging.error(f"Speech synthesis failed: {e}")
            raise


class ClaudeClient:
    """Client for interacting with Anthropic's Claude API"""
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)  # Changed to AsyncAnthropic
    
    async def get_response(self, text: str) -> str:
        """Get response from Claude for the given text"""
        try:
            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": text
                }]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Claude API error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

class WhisperTranscriber:
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        # Audio configuration
        self.config = AudioConfig()
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
                samplerate=self.config.samplerate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.blocksize,
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

            output_path = Path("output.wav")
            try:
                full_audio = np.concatenate(self.recording_data)
                sf.write(output_path, full_audio, self.config.samplerate)
                
                result = self.transcribe_audio(output_path)
                transcribed_text = result.get('text', '').strip()
                
                if transcribed_text:
                    pyperclip.copy(transcribed_text)
                    logging.info(f"Transcribed: {transcribed_text}")
                    logging.info("Text copied to clipboard")
                else:
                    logging.warning("No text transcribed")
                    
            except Exception as e:
                logging.error(f"Processing error: {e}")
            finally:
                # Cleanup
                if output_path.exists():
                    output_path.unlink()
            print("\a")  # System beep

    async def process_with_claude_tts(self):
        """Process clipboard content with Claude and TTS"""
        if not self.claude_client or not self.tts_engine:
            logging.error("Claude/TTS not initialized")
            return
        
        try:
            # Get clipboard content
            text = pyperclip.paste()
            if not text:
                logging.warning("No text in clipboard")
                return
            
            logging.info("Sending to Claude: " + text[:100] + "...")
            
            # Get Claude's response
            response = await self.claude_client.get_response(text)
            if not response:
                return
            
            logging.info("Received from Claude: " + response[:100] + "...")
            
            # Synthesize speech
            logging.info("Synthesizing speech...")
            audio_path = self.tts_engine.synthesize(response)
            
            # Play audio
            logging.info("Playing audio response...")
            data, samplerate = sf.read(str(audio_path))
            sd.play(data, samplerate)
            sd.wait()
            
            # Copy Claude's response to clipboard
            pyperclip.copy(response)
            logging.info("Claude's response copied to clipboard")
            
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
            logging.info("Press 'F12' to start recording, 'F11' to stop recording, 'F10' for Claude+TTS, 'Ctrl+C' to exit")
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