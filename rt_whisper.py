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
from pathlib import Path
import logging
import os
import anthropic
import asyncio
from tacotron2.model import Tacotron2
from tacotron2.text import text_to_sequence
from waveglow.denoiser import Denoiser
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Original directory setup
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
    """TTS configuration settings"""
    model_dir: Path = Path.home() / ".tts_models"
    tacotron_path: Path = model_dir / "tacotron2_statedict.pt"
    waveglow_path: Path = model_dir / "waveglow_256channels_universal_v5.pt"
    output_dir: Path = Path.home() / ".tts_output"
    sample_rate: int = 22050

class TTSEngine:
    """Text-to-Speech engine using Tacotron2 and WaveGlow"""
    def __init__(self, config: TTSConfig):
        self.config = config
        self.config.model_dir.mkdir(exist_ok=True)
        self.config.output_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tacotron2 = self._load_tacotron()
        self.waveglow = self._load_waveglow()
        self.denoiser = Denoiser(self.waveglow) if self.device.type == 'cuda' else None
        
    def _load_tacotron(self):
        model = Tacotron2().to(self.device)
        checkpoint = torch.load(self.config.tacotron_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model
    
    def _load_waveglow(self):
        model = torch.load(self.config.waveglow_path, map_location=self.device)['model']
        model.eval()
        return model
    
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech from text and return path to audio file"""
        if output_path is None:
            output_path = self.config.output_dir / f"speech_{hash(text)}.wav"
        
        sequence = torch.tensor(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = sequence.to(self.device)
        
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron2.inference(sequence)
            audio = self.waveglow.infer(mel_outputs_postnet)
            
            if self.denoiser is not None:
                audio = self.denoiser(audio, strength=0.01)
        
        audio_np = audio[0].cpu().numpy()
        sf.write(str(output_path), audio_np, self.config.sample_rate)
        return output_path

class ClaudeClient:
    """Client for interacting with Anthropic's Claude API"""
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    async def get_response(self, text: str) -> str:
        """Get response from Claude for the given text"""
        try:
            message = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": text
                }]
            )
            return message.content[0].text
        except Exception as e:
            logging.error(f"Claude API error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

class WhisperTranscriber:
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        # Original Whisper initialization
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
            logging.info(f"Model loaded on device: {self.model.device}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

        # New Claude and TTS initialization
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

    # Original audio callback method
    def audio_callback(self, indata: np.ndarray, frames: int, time: sd.CallbackFlags, status: sd.CallbackFlags) -> None:
        """Process audio input in real-time"""
        if status:
            logging.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())

    # Original recording method
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

    # Original transcription method
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

    # Original recording start method
    def start_recording(self) -> None:
        """Start recording in a new thread"""
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()
            logging.info("Recording started")
            print("\a")  # System beep

    # Original recording stop method
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