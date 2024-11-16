from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
from pytube import YouTube
import librosa
import warnings
import subprocess
import tempfile
import time
warnings.filterwarnings('ignore')

class AudioPrep:
    def __init__(self):
        self.output_dir = Path.home() / ".voice_references"
        self.output_dir.mkdir(exist_ok=True)
        self.sample_rate = 22050  # XTTS expects this sample rate
        
    def record_voice(self, duration=5):
        """Record voice from microphone"""
        print(f"\nRecording for {duration} seconds...")
        print("Speak after the beep...")
        print("\a")  # System beep
        
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()
        print("Recording finished!")
        
        output_path = self.output_dir / f"recorded_reference_{int(time.time())}.wav"
        sf.write(output_path, recording, self.sample_rate)
        return output_path
    
    def from_youtube(self, url, start_time=0, duration=5):
        """Extract audio from YouTube video"""
        print("\nDownloading YouTube video...")
        try:
            # Download audio only
            yt = YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            
            # Download to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            audio_stream.download(filename=temp_file.name)
            
            # Convert to WAV and trim
            output_path = self.output_dir / f"youtube_reference_{int(time.time())}.wav"
            
            # Use ffmpeg to convert and trim
            subprocess.run([
                'ffmpeg', '-y',
                '-i', temp_file.name,
                '-ss', str(start_time),
                '-t', str(duration),
                '-ar', str(self.sample_rate),
                '-ac', '1',
                str(output_path)
            ], capture_output=True)
            
            Path(temp_file.name).unlink()  # Clean up temp file
            return output_path
            
        except Exception as e:
            print(f"Error processing YouTube video: {e}")
            return None
    
    def from_file(self, input_path, start_time=0, duration=5):
        """Process existing audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(input_path, sr=self.sample_rate, offset=start_time, duration=duration)
            
            # Save processed audio
            output_path = self.output_dir / f"processed_reference_{int(time.time())}.wav"
            sf.write(output_path, audio, self.sample_rate)
            return output_path
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return None
    
    def preview_audio(self, audio_path):
        """Preview the processed audio"""
        try:
            data, sr = sf.read(audio_path)
            print("\nPlaying preview...")
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            print(f"Error playing audio: {e}")

def main():
    prep = AudioPrep()
    
    while True:
        print("\nReference Audio Preparation Tool")
        print("1. Record your voice")
        print("2. Extract from YouTube")
        print("3. Process existing audio file")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            duration = float(input("Enter recording duration in seconds (default 5): ") or 5)
            output_path = prep.record_voice(duration)
            
        elif choice == '2':
            url = input("Enter YouTube URL: ")
            start_time = float(input("Enter start time in seconds (default 0): ") or 0)
            duration = float(input("Enter duration in seconds (default 5): ") or 5)
            output_path = prep.from_youtube(url, start_time, duration)
            
        elif choice == '3':
            input_path = input("Enter path to audio file: ")
            start_time = float(input("Enter start time in seconds (default 0): ") or 0)
            duration = float(input("Enter duration in seconds (default 5): ") or 5)
            output_path = prep.from_file(input_path, start_time, duration)
            
        elif choice == '4':
            break
            
        else:
            print("Invalid choice!")
            continue
        
        if output_path and output_path.exists():
            print(f"\nAudio saved to: {output_path}")
            preview = input("Preview the audio? (y/n): ")
            if preview.lower() == 'y':
                prep.preview_audio(output_path)

if __name__ == "__main__":
    main()

