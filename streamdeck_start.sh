#!/bin/bash
# Launch the RT-Whisper transcriber as a background process

# Navigate to script directory
cd "$(dirname "$0")"

# Set up log directory if it doesn't exist
LOGS_DIR=~/.whisper_logs
mkdir -p "$LOGS_DIR"

# Activate the virtual environment
source .venv/bin/activate

# Set up CUDA environment variables
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Add Python-specific CUDA libraries to LD_LIBRARY_PATH
PYTHON_CUDA_PATHS=$(python -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))' 2>/dev/null)
if [ $? -eq 0 ]; then
    export LD_LIBRARY_PATH="$PYTHON_CUDA_PATHS:$LD_LIBRARY_PATH"
    echo "CUDA libraries configured successfully."
else
    echo "Warning: CUDA Python libraries not found. You may need to install them in this virtual environment."
    echo "Run: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*"
fi

# Run RT-Whisper in command mode in the background with specific settings for StreamDeck
echo "Starting RT-Whisper in command mode with optimal settings..."
nohup python -m rt_whisper --device-type cuda --compute-type float16 --command-mode --device $(python -c "import sounddevice as sd; devices = sd.query_devices(); print([i for i, d in enumerate(devices) if 'Poly VSurround 80' in d.get('name', '') and d.get('max_input_channels', 0) > 0][0] if [i for i, d in enumerate(devices) if 'Poly VSurround 80' in d.get('name', '') and d.get('max_input_channels', 0) > 0] else 0)") > "$LOGS_DIR/rt_whisper_stdout.log" 2>&1 &

# Save the process ID for future termination
PID=$!
echo $PID > "$LOGS_DIR/rt_whisper.pid"

echo "RT-Whisper started in background (PID: $PID)."
echo "Output logs: $LOGS_DIR/rt_whisper_stdout.log"
echo "Use kill_transcribe.sh to stop it."
