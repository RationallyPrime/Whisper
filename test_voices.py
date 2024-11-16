from pathlib import Path
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS
import time

def test_tts_voices():
    """Test different TTS voices and play them immediately"""
    
    # Initialize TTS
    print("Initializing TTS model...")
    tts = TTS("tts_models/en/vctk/vits")
    
    # Test text - using a more varied test to better evaluate the voice
    test_text = "Hi there! I'm testing my voice with different types of sentences. How do I sound? This is a question. And this is an excited statement!"
    
    # Create output directory
    output_dir = Path.home() / ".voice_tests"
    output_dir.mkdir(exist_ok=True)
    
    # New selection of speakers to try
    speakers_to_try = [
        'p262',  # Female, clear British accent
        'p333',  # Male, crisp pronunciation
        'p299',  # Female, natural intonation
        'p286',  # Male, good pacing
        'p276',  # Female, professional tone
        'p317',  # Male, engaging voice
        'p294',  # Female, good clarity
        'p347',  # Male, natural rhythm
        'p261',  # Female, good expression
        'p308',  # Male, clear diction
    ]
    
    print("\nAvailable speakers:", tts.speakers)
    print("\nTesting voices. Press Ctrl+C to skip to next voice.")
    
    for speaker in speakers_to_try:
        try:
            print(f"\nTesting speaker {speaker}")
            output_path = output_dir / f"test_{speaker}.wav"
            
            # Generate speech
            tts.tts_to_file(
                text=test_text,
                file_path=str(output_path),
                speaker=speaker
            )
            
            # Play the generated audio
            print(f"Playing {speaker}...")
            data, samplerate = sf.read(str(output_path))
            sd.play(data, samplerate)
            sd.wait()
            
            # Ask if user wants to continue
            response = input("\nRate this voice (1-5) or 'q' to quit: ")
            if response.lower() == 'q':
                break
            
            # Optional: Save rating
            if response.isdigit():
                print(f"Rated {speaker}: {response}/5")
                
        except KeyboardInterrupt:
            print("\nSkipping to next voice...")
            sd.stop()
            continue
        except Exception as e:
            print(f"Error testing speaker {speaker}: {e}")
    
    print(f"\nVoice test files saved in {output_dir}")

if __name__ == "__main__":
    test_tts_voices()
