from pathlib import Path
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS
import librosa
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress audioread warning

def process_reference_audio(input_file, start_time=0, duration=10):  # Changed default to 10 seconds
    """Process the reference audio file to get a clean segment"""
    try:
        # Load audio with librosa (resamples automatically)
        audio, sr = librosa.load(
            input_file, 
            sr=22050,  # XTTS expected sample rate
            offset=start_time,
            duration=duration
        )
        
        # Save processed segment
        output_dir = Path.home() / ".voice_references"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"processed_reference_{int(time.time())}.wav"
        sf.write(output_path, audio, sr)
        return output_path
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def test_xtts_voice():
    """Test XTTS v2 with our reference audio"""
    
    # Initialize XTTS v2
    print("Initializing XTTS v2 model...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    
    # Input audio file
    input_file = "./Meeting Abenthy ｜ The Name of the Wind ｜ The Kingkiller Chronicle.webm"
    
    # Create output directory
    output_dir = Path.home() / ".voice_tests"
    output_dir.mkdir(exist_ok=True)
    
    while True:
        try:
            # Get timestamp and duration from user
            time_input = input("\nEnter start time in seconds (or 'q' to quit): ")
            if time_input.lower() == 'q':
                break
                
            start_time = float(time_input)
            duration = float(input("Enter duration in seconds (default 10): ") or 10)
                
            # Process reference audio
            ref_path = process_reference_audio(input_file, start_time, duration)
            if not ref_path:
                continue
                
            # Preview reference segment
            print("\nPlaying reference segment...")
            data, sr = sf.read(ref_path)
            sd.play(data, sr)
            sd.wait()
            
            proceed = input("\nUse this segment? (y/n): ")
            if proceed.lower() != 'y':
                continue
            
            # Test text - using varied content
            test_text = "Hi there! I'm testing my voice with different types of sentences. How do I sound? This is a question. And this is an excited statement!"
            
            output_path = output_dir / "xtts_test.wav"
            
            # Generate speech
            print("\nGenerating speech...")
            tts.tts_to_file(
                text=test_text,
                file_path=str(output_path),
                speaker_wav=str(ref_path),
                language="en"
            )
            
            # Play the generated audio
            print("Playing generated audio...")
            data, samplerate = sf.read(str(output_path))
            sd.play(data, samplerate)
            sd.wait()
            
            # Get feedback
            rating = input("\nRate this voice (1-5) or 'q' to quit: ")
            if rating.lower() == 'q':
                break
            
            if rating.isdigit():
                print(f"Rated voice: {rating}/5")
            
        except KeyboardInterrupt:
            print("\nStopping playback...")
            sd.stop()
        except ValueError as e:
            if "quit" in str(e).lower():
                break
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\nTest files saved in {output_dir}")

if __name__ == "__main__":
    test_xtts_voice()
