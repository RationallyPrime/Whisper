import whisper
import sounddevice as sd
import numpy as np
import queue
import warnings
import soundfile as sf
import pyperclip  # For copying transcription result to clipboard
from pynput import keyboard  # For listening to keyboard events
import threading  # For running the recording in a separate thread

warnings.filterwarnings("ignore", category=FutureWarning)

# Load the Whisper model (using 'base', 'small', 'medium', or 'large')
model = whisper.load_model("turbo", device="cuda")  # For GPU optimization with float16 precision
print(model.device)

# Global variables
is_recording = False
audio_queue = queue.Queue()
samplerate = 16000
recording_data = []
recording_thread = None  # We'll store the recording thread here

# Callback function for sounddevice input stream
def audio_callback(indata, frames, time, status):
    """Callback function to store audio data in real-time."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def record_audio():
    """Record audio in real-time in a separate thread."""
    global recording_data
    recording_data = []  # Reset the recorded data
    print("Recording started...")

    # Start the input stream and store audio data
    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
        while is_recording:  # Keep capturing audio until stopped
            audio_data = []
            while not audio_queue.empty():
                audio_data.append(audio_queue.get())
            if audio_data:
                recording_data.append(np.concatenate(audio_data, axis=0))

    print("Recording thread finished.")

def start_recording():
    """Start the recording process in a new thread."""
    global is_recording, recording_thread
    is_recording = True
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()

def stop_recording():
    """Stop recording and save the audio file."""
    global is_recording, recording_data
    is_recording = False  # Signal the recording thread to stop
    recording_thread.join()  # Wait for the recording thread to finish
    print("Recording stopped.")

    # Combine recorded audio data
    if recording_data:
        full_audio = np.concatenate(recording_data)
        sf.write('output.wav', full_audio, samplerate)  # Save to file

        # Transcribe the recorded audio
        print("Transcribing...")
        result = model.transcribe('output.wav', fp16=False, verbose=True, task='transcribe')
        print(f"Transcribed Text: {result['text']}")
        print(f"Full Transcription Result: {result}")

        # Copy the transcribed text to the clipboard
        pyperclip.copy(result['text'])
        print("Transcribed text copied to clipboard!")

def on_press(key):
    """Callback for key press event."""
    try:
        if key == keyboard.Key.f12:  # Use F12 to start recording
            if not is_recording:
                start_recording()
        elif key == keyboard.Key.f11:  # Use F11 to stop recording
            if is_recording:
                stop_recording()
    except AttributeError:
        pass

def transcribe_real_time():
    """Waits for key press to start/stop recording and transcribes in real-time."""
    print("Press 'F12' to start recording, 'F11' to stop recording, 'Ctrl+C' to exit.")
    
    # Listen for key presses
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    transcribe_real_time()
