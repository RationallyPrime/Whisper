default:
    @just --list

dev *ARGS:
    uv run python -m rt_whisper {{ARGS}}

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

setup:
    uv sync
    just install-audio-config
