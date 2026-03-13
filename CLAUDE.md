# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Setup: `uv sync && just install-audio-config`
- Start daemon: `just start` (background) or `just dev` (foreground)
- Stop daemon: `just kill`
- Toggle dictation: `just toggle`
- Check status: `just status`
- Lint: `just lint`
- Format: `just fmt`
- Type check: `just check`
- Test: `just test`
- List devices: `just dev --list-devices`
- Show keyboard shortcuts: `just shortcuts`

## Architecture
- `rtwhisperctl` is the primary CLI entry point (console_script)
- Daemon polls `~/.whisper_logs/command.json` for commands
- Status IPC via `~/.whisper_logs/status.json` (atomic writes)
- Ack IPC via `~/.whisper_logs/ack.json` (for --wait flag)
- StreamDeck scripts still work but are no longer the primary interface

## Code Style
- Line length: 100 characters
- Formatting/linting: ruff (no black, no pylint, no flake8)
- Type checking: ty (no pyright, no mypy)
- Package management: uv with pyproject.toml (no pip, no poetry)
- Python version target: 3.12+
- Import order: stdlib, third-party, local (enforced by ruff `I` rule)
- Use Pydantic v2 models for configuration
- Use Protocol classes for module boundaries
- Handle exceptions with try/except and logging
- Use type hints for all function parameters and returns
- Document functions with docstrings (Google style)
- Use Path from pathlib for file operations
- Log errors and important events via logging module
