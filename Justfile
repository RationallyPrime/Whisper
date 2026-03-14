default:
    @just --list

# Start the RT-Whisper daemon in the background
start *ARGS:
    #!/usr/bin/env bash
    mkdir -p ~/.whisper_logs
    nohup uv run rtwhisperctl daemon {{ARGS}} > ~/.whisper_logs/rt_whisper_stdout.log 2>&1 &
    echo "Daemon started (PID: $!). Logs: ~/.whisper_logs/"

# Stop recording (if active)
stop:
    uv run rtwhisperctl stop

# Toggle dictation on/off
toggle:
    uv run rtwhisperctl toggle

# Show daemon status
status:
    uv run rtwhisperctl status

# Start daemon in foreground (for development/debugging)
dev *ARGS:
    uv run rtwhisperctl daemon {{ARGS}}

# Kill the daemon process
kill:
    bash streamdeck_kill.sh

lint:
    uv run ruff check rt_whisper/

fmt:
    uv run ruff format rt_whisper/ tests/

check:
    uv run ty check rt_whisper/

test:
    uv run pytest tests/ -v

install-audio-config:
    @./scripts/install-audio-config.sh

# Show COSMIC keyboard shortcut bindings
shortcuts:
    @./scripts/install-cosmic-shortcuts.sh

setup:
    uv sync
    just install-audio-config
    @echo ""
    @echo "Run 'just shortcuts' to see keyboard shortcut bindings."
