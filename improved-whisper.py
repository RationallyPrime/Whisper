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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add this to your configuration
output_dir = Path.home() / ".whisper_tmp"  # Hidden directory for temp files
output_dir.mkdir(exist_ok=True)

# Add this to prevent reloading the model each time
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

class WhisperTranscriber:
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        self.config = AudioConfig()
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_data = []
        self.recording_thread: Optional[threading.Thread] = None
        
        # Ensure CUDA is available if requested
        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        
        try:
            self.model = whisper.load_model(model_name, device=device)
            logging.info(f"Model loaded on device: {self.model.device}")
        except Exception as e:
            print(f"Failed to load model: {e}")  # Simple print is fine for personal use
            raise

    def audio_callback(self, indata: np.ndarray, frames: int, time: sd.CallbackTime, status: sd.CallbackFlags) -> None:
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
                    # Process all available chunks
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

    def on_press(self, key: keyboard.Key) -> None:
        """Handle keyboard events"""
        try:
            if key == keyboard.Key.f12 and not self.is_recording:
                self.start_recording()
            elif key == keyboard.Key.f11 and self.is_recording:
                self.stop_recording()
        except Exception as e:
            logging.error(f"Keyboard handling error: {e}")

    def run(self) -> None:
        """Main execution loop"""
        logging.info("Press 'F12' to start recording, 'F11' to stop recording, 'Ctrl+C' to exit")
        
        with keyboard.Listener(on_press=self.on_press) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                logging.info("Shutting down...")
                if self.is_recording:
                    self.stop_recording()

if __name__ == "__main__":
    transcriber = WhisperTranscriber(model_name="base", device="cuda")
    transcriber.run()
