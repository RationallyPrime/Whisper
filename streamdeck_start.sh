#!/bin/bash
# Launch the RT-Whisper transcriber as a background process

# Navigate to script directory
cd "$(dirname "$0")"

# Set up log directory if it doesn't exist
LOGS_DIR=~/.whisper_logs
mkdir -p "$LOGS_DIR"

# Set up CUDA environment variables
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Add Python-specific CUDA libraries to LD_LIBRARY_PATH
PYTHON_CUDA_PATHS=$(uv run python -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))' 2>/dev/null)
if [ $? -eq 0 ]; then
    export LD_LIBRARY_PATH="$PYTHON_CUDA_PATHS:$LD_LIBRARY_PATH"
    echo "CUDA libraries configured successfully."
else
    echo "Warning: CUDA Python libraries not found. You may need to install them."
    echo "Run: uv add nvidia-cublas-cu12 nvidia-cudnn-cu12"
fi

# Detect Yeti Nano device ID dynamically
YETI_DEVICE=$(uv run python -c 'import sounddevice as sd; devices = sd.query_devices(); yeti = [i for i, d in enumerate(devices) if "Yeti" in d["name"] and d["max_input_channels"] > 0]; print(yeti[0] if yeti else 5)' 2>/dev/null)
echo "Detected Yeti Nano at device ID: $YETI_DEVICE"

# Run RT-Whisper in the background via rtwhisperctl
echo "Starting RT-Whisper with optimal settings..."
nohup uv run rtwhisperctl daemon --device "$YETI_DEVICE" --device-type cuda --compute-type float16 > "$LOGS_DIR/rt_whisper_stdout.log" 2>&1 &

# Save the process ID for future termination
PID=$!
echo $PID > "$LOGS_DIR/rt_whisper.pid"

echo "RT-Whisper started in background (PID: $PID)."
echo "Output logs: $LOGS_DIR/rt_whisper_stdout.log"
echo "Use streamdeck_kill.sh or 'just kill' to stop it."
