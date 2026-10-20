# RT-Whisper

Desktop dictation using local Faster-Whisper speech recognition, with a background daemon, command-line recording controls, and optional Claude clipboard processing.

This README covers the `whisper-dictation` default branch, packaged as `rt-whisper` 0.4.0. The owner reports extensive testing and successful use of the Whisper implementation.

## Requirements

- Python 3.12 or later and [uv](https://docs.astral.sh/uv/).
- A microphone and working PortAudio support for Python sounddevice.
- For default GPU inference: a CUDA-capable NVIDIA GPU and compatible PyTorch/CTranslate2 runtime libraries.
- Desktop clipboard support for clipboard features.

Linux audio and COSMIC keyboard shortcut helpers are included; these integrations are platform-specific.

## Install

```sh
git clone --branch whisper-dictation https://github.com/RationallyPrime/Whisper.git
cd Whisper
uv sync --extra dev
```

## Run

List input devices:

```sh
uv run python -m rt_whisper --list-devices
```

Start the daemon in a terminal:

```sh
uv run rtwhisperctl daemon
```

Defaults are the `large-v3` model, CUDA, float16 and English. CPU inference can be selected explicitly:

```sh
uv run rtwhisperctl daemon --device-type cpu --compute-type int8
```

Use `--device <index>` to select a microphone or `--model <name>` to change the Faster-Whisper model. The default model cache is `~/.whisper_cache`; weights may need downloading on first use.

Control recording from another terminal:

```sh
uv run rtwhisperctl start
uv run rtwhisperctl stop
uv run rtwhisperctl toggle
uv run rtwhisperctl status
uv run rtwhisperctl --json status
```

`start` begins recording in an already-running daemon. `stop` ends recording and triggers transcription. Put `--wait` before a command to wait for acknowledgement.

With [just](https://github.com/casey/just) installed, `just start` launches the daemon in the background, `just dev` runs it in the foreground, and `just status` checks its state. `just stop` stops recording; `just kill` invokes the daemon shutdown helper.

## Configuration

The daemon reads `~/.whisper_config/config.json`. Precedence is defaults, then JSON configuration, then CLI overrides. Example:

```json
{
  "transcription": {
    "model": "large-v3",
    "device_type": "cpu",
    "compute_type": "int8",
    "language": "en"
  },
  "claude": {
    "enable": false
  }
}
```

Sections are `audio`, `transcription`, `claude` and `daemon`. See [rt_whisper/config.py](rt_whisper/config.py) for all fields and defaults. The control CLI uses command, status and acknowledgement files in `~/.whisper_logs`; daemon logs also go there by default.

## Optional Claude integration

Enable `claude.enable` and provide `ANTHROPIC_API_KEY` in the environment or a dotenv file loaded by the configuration. This is separate from local Whisper transcription. Clipboard processing sends text to the configured Claude service.

```sh
uv run rtwhisperctl clipboard explain
uv run rtwhisperctl clipboard summarize
uv run rtwhisperctl clipboard translate
```

Use `uv run rtwhisperctl --help` for the full control interface.

## Development

```sh
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check rt_whisper/
uv run ruff format --check rt_whisper/ tests/
```

`just setup` also installs host audio configuration. Inspect [Justfile](Justfile) and its scripts before using that task on a new machine.

## Layout

| Path | Purpose |
| --- | --- |
| `rt_whisper/` | Capture, transcription, orchestration, configuration and CLI. |
| `config/` | Repository configuration assets. |
| `scripts/` | Audio and desktop integration helpers. |
| `tests/` | Automated tests. |
| `docs/superpowers/specs/` | Design specifications. |
| `archive/` | Earlier implementations. |
