# Whisper Tier A: Audio Quality + Config-Driven Refactor

## Context

RationallyPrime/Whisper is a local speech-to-text tool using Faster-Whisper, with StreamDeck integration and Claude API processing. A deep research report identified audio capture misalignment (44.1kHz defaults vs 48kHz Yeti Nano), missing VAD filtering, and scattered configuration as the highest-impact improvements for daily dictation quality.

This spec covers Tier A — the audio and transcription quality layer plus project tooling modernization. Tiers B (control plane/UX) and C (model flexibility) follow in later specs.

## Goals

1. Align audio capture to Blue Yeti Nano's native 48kHz/24-bit spec
2. Add VAD filtering to reduce silence and end-of-utterance hallucinations
3. Replace scattered config with validated Pydantic v2 models
4. Split monolithic transcriber into focused modules
5. Modernize project tooling (Python 3.12, uv, ruff, ty, Justfile)
6. Auto-install device-scoped WirePlumber config for the Yeti Nano

## Non-Goals

- `rtwhisperctl` atomic command writer (Tier B)
- Status/ack IPC mechanism (Tier B)
- COSMIC keyboard shortcuts or tray icon (Tier B)
- NeMo Canary/Parakeet model backends (Tier C)
- Hotword/wake word detection
- Local web UI
- systemd user service

## Design

### 1. Config Model Architecture

A Pydantic v2 model hierarchy in `rt_whisper/config.py` serves as the single source of truth for all settings.

```
WhisperConfig (root)
├── AudioConfig          # samplerate, channels, dtype, device, blocksize
├── TranscriptionConfig  # model, device_type, compute_type, beam_size,
│                        # vad_filter, vad_parameters, language, cache_dir
├── ClaudeConfig         # enable, api_key, model
└── DaemonConfig         # log_dir, config_dir, tmp_dir, command_poll_interval
```

**Defaults (key changes from current):**

| Field | Old Default | New Default | Rationale |
|---|---|---|---|
| `AudioConfig.samplerate` | 44100 | 48000 | Yeti Nano native spec |
| `AudioConfig.blocksize` | 16384 (2048*8) | 2048 | Snappier push-to-talk feel |
| `TranscriptionConfig.vad_filter` | (not set) | True | Reduce silence/hallucinations |
| `TranscriptionConfig.vad_parameters` | (not set) | `{"min_silence_duration_ms": 500}` | Safe default per Faster-Whisper docs |
| `TranscriptionConfig.beam_size` | (not set) | 3 | Balanced speed/quality |

**Precedence:** model defaults < config file (`~/.whisper_config/config.json`) < CLI args.

**Loading:** `WhisperConfig.load()` class method reads JSON config if it exists, merges with defaults, validates. CLI arg overrides applied after load.

**Validation:** Pydantic v2 validators enforce constraints (e.g., samplerate > 0, beam_size >= 1, valid compute_type enum). `TranscriptionConfig.device_type = "cuda"` validates `torch.cuda.is_available()` at load time — fails fast with a clear message rather than crashing deep inside Faster-Whisper.

### 2. Protocol Definitions

Protocols decouple module boundaries and enable testing with fakes.

```python
from typing import Protocol
from pathlib import Path

class Recorder(Protocol):
    """Audio recording interface."""
    def start_recording(self) -> None: ...
    def stop_recording(self) -> Path: ...
    def play_beep(self, frequency: int = 440, duration: float = 0.1) -> None: ...
    @property
    def is_recording(self) -> bool: ...

class Transcriber(Protocol):
    """Speech-to-text interface."""
    def transcribe_audio(self, audio_path: Path) -> str: ...

class TextProcessor(Protocol):
    """Post-transcription text processing (e.g., Claude)."""
    async def process_clipboard(self) -> None: ...
```

These live in `rt_whisper/protocols.py`. Concrete classes implement them without explicit inheritance (structural subtyping).

### 3. Module Split

Current `transcriber.py` (~428 lines) handles config, audio, transcription, IPC, and beeps. Split into focused modules:

```
rt_whisper/
├── config.py              # WhisperConfig + sub-models, load/save/merge
├── protocols.py           # Recorder, Transcriber, TextProcessor protocols
├── audio.py               # AudioCapture(Recorder): record, stop, beep
├── transcriber.py         # WhisperTranscriber(Transcriber): model init, transcribe
├── orchestrator.py        # Orchestrator: main loop, wires command dispatch
├── commands.py            # CommandHandler: parse command.json, dispatch via callbacks
├── claude_client.py       # ClaudeClient(TextProcessor): Claude API (beeps extracted to audio.py)
├── claude_setup.py        # Setup wizard for Claude config (updated to use WhisperConfig)
├── prompts.py             # System prompts (unchanged)
├── streamdeck_claude.py   # StreamDeck Claude command sender (shebang updated for uv)
├── streamdeck_record.py   # StreamDeck record button (shebang updated for uv)
├── streamdeck_stop.py     # StreamDeck stop button (shebang updated for uv)
├── _version.py            # Version (unchanged)
├── __main__.py            # CLI entry, arg parsing, DI wiring
└── __init__.py            # Package exports (updated for new modules)
```

**Dependency direction (one-way, no cycles):**

```
__main__.py
  ├── config.py
  ├── audio.py         → protocols.py, config.py
  ├── transcriber.py   → protocols.py, config.py
  ├── claude_client.py → protocols.py, audio.py (for beeps)
  ├── commands.py      → protocols.py (accepts Recorder, Transcriber, TextProcessor)
  └── orchestrator.py  → commands.py, config.py
```

`CommandHandler` does NOT hold references to concrete types. It receives protocol-typed callables/objects. The orchestrator owns the main loop and passes protocol references down.

**Boundaries:**

- `config.py` — owns all settings. Other modules receive config via constructor injection.
- `protocols.py` — protocol definitions only. No implementation.
- `audio.py` — `AudioCapture` class implementing `Recorder`. Knows `sounddevice`, numpy, WAV writing. Does not know about transcription or models. Receives `AudioConfig`.
- `transcriber.py` — `WhisperTranscriber` implementing `Transcriber`. Owns `WhisperModel` lifecycle. Receives `TranscriptionConfig`.
- `commands.py` — `CommandHandler`. Parses `command.json`, dispatches via protocol references. Receives `DaemonConfig` + protocol-typed dependencies. No circular references.
- `orchestrator.py` — `Orchestrator`. Owns the main event loop (`run()`). Wires `CommandHandler` polling with `AudioCapture` and `WhisperTranscriber`. The old `run()` / `run_command_mode()` logic moves here.
- `claude_client.py` — `ClaudeClient` implementing `TextProcessor`. Beep methods (`_play_success_beep`, `_play_error_beep`) removed; replaced with calls to a `Recorder` instance passed via DI for audio feedback.
- `streamdeck_*.py` — shebangs updated from `.venv/bin/python3` to `#!/usr/bin/env -S uv run python`. Logic unchanged.
- `claude_setup.py` — updated to write config compatible with `WhisperConfig` schema.

**AudioCapture threading model:**

- `start_recording()` spawns a `threading.Thread` internally (same as current). Sets `self._is_recording = True`. The sounddevice callback pushes chunks to an internal `queue.Queue`.
- `stop_recording() -> Path` blocks: sets `_is_recording = False`, joins the thread, writes the accumulated audio to a temp WAV file, returns the path. Caller is responsible for cleanup.
- `play_beep()` is synchronous (calls `sd.play()` + `sd.wait()`).

**DI wiring in `__main__.py`:**

```python
config = WhisperConfig.load(config_path, cli_overrides)
audio = AudioCapture(config.audio)
transcriber = WhisperTranscriber(config.transcription)
claude = ClaudeClient(config.claude, audio) if config.claude.enable else None
commands = CommandHandler(config.daemon, recorder=audio, transcriber=transcriber, text_processor=claude)
orchestrator = Orchestrator(config, commands)
orchestrator.run()
```

### 4. Audio Capture Changes

**Samplerate:** Default 48000. Capture at device-native rate. Faster-Whisper handles the 48k->16k conversion internally via ffmpeg when given a file path — no resampling in our code. Note: this relies on file-path-based transcription. If we ever switch to passing raw numpy arrays, we'd need to resample explicitly.

**Blocksize:** Default 2048 (down from 16384). Reduces perceived latency for start/stop responsiveness from ~371ms (16384/44100) to ~43ms (2048/48000).

**Beep generator:** All beep generation consolidated in `AudioCapture.play_beep()`. Replaces:
- `transcriber.py` `_play_beep()` (hardcoded 44100)
- `transcriber.py` `start_recording()` `print("\a")` (terminal bell — inconsistent)
- `claude_client.py` `_play_success_beep()` and `_play_error_beep()` (hardcoded 44100)

All now use `config.audio.samplerate`:

```python
def play_beep(self, frequency: int = 440, duration: float = 0.1) -> None:
    sr = self.config.samplerate
    t = np.linspace(0, duration, int(sr * duration))
    beep = np.sin(2 * np.pi * frequency * t) * 0.3
    sd.play(beep, sr)
    sd.wait()
```

### 5. Transcription Changes

Pass config-driven parameters to `model.transcribe()`:

```python
segments, info = self.model.transcribe(
    str(audio_path),
    language=self.config.language,
    task="transcribe",
    beam_size=self.config.beam_size,
    vad_filter=self.config.vad_filter,
    vad_parameters=self.config.vad_parameters,
)
```

VAD with `min_silence_duration_ms=500` removes silence gaps > 500ms. This reduces false tails and "end-of-utterance hallucinations" without clipping leading/trailing phonemes aggressively.

### 6. WirePlumber Device-Scoped Config

Auto-install to `~/.config/wireplumber/wireplumber.conf.d/10-yeti-nano.conf`:

```ini
monitor.alsa.rules = [
  {
    matches = [
      { node.name = "~alsa_input.*Yeti.*" }
    ]
    actions = {
      update-props = {
        audio.rate = 48000
        audio.format = "S24LE"
        session.suspend-timeout-seconds = 0
        api.alsa.period-size = 256
        api.alsa.headroom = 1024
      }
    }
  }
]
```

**What each setting does:**

- `audio.rate = 48000` — pins Yeti capture to native rate, avoids PipeWire rate-switching
- `audio.format = "S24LE"` — matches Yeti's 24-bit depth
- `session.suspend-timeout-seconds = 0` — disables suspend (USB mics behave badly on resume)
- `api.alsa.period-size = 256` — reduces buffering delay for USB batch devices
- `api.alsa.headroom = 1024` — stability margin (increase to 2048 if crackling)

**Scope:** Only fires for ALSA input nodes matching `*Yeti*`. No effect on other devices.

**Install mechanism:** `just install-audio-config` copies the file from `config/wireplumber/10-yeti-nano.conf` in the repo to `~/.config/wireplumber/wireplumber.conf.d/`. Idempotent — skips if content matches. Validates WirePlumber >= 0.5 (conf file format; older versions use Lua in `main.lua.d/`) and warns if incompatible.

### 7. Project Tooling Migration

**Python:** `requires-python = ">=3.12"` in `pyproject.toml` (drop upper bound).

**Build backend:** hatchling (unchanged — compatible with uv).

**Package management:** uv. Setup becomes `uv sync`. Lock file committed.

**Dependencies updated in `pyproject.toml`:**
- Add: `pydantic>=2.0`
- Remove: `ruff` from runtime deps (dev tool, invoked via `uv run ruff`)
- Remove: `basedpyright` dev dependency
- Keep: `openai` and `tiktoken` — used by Claude client for token counting
- StreamDeck scripts: update shebangs from `.venv/bin/python3` to `#!/usr/bin/env -S uv run python`

**Linting + formatting:** ruff only. Drop black entirely (no `[tool.black]` config exists, but CLAUDE.md reference removed). Config:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "TCH"]

[tool.ruff.format]
quote-style = "double"
```

**Type checking:** ty. Remove `[tool.pyright]` section and `basedpyright` dependency.

**Task runner:** Justfile:

```just
default:
    @just --list

dev *ARGS:
    uv run python -m rt_whisper {{ARGS}}

kill:
    bash streamdeck_kill.sh

lint:
    uv run ruff check rt_whisper/

fmt:
    uv run ruff format rt_whisper/

check:
    uv run ty check rt_whisper/

test:
    uv run pytest tests/ -v

install-audio-config:
    @./scripts/install-audio-config.sh

setup:
    uv sync
    just install-audio-config
```

**CLAUDE.md:** Update to reflect new tooling (uv, ruff, ty, just, Python 3.12).

## Migration Notes

- Existing `~/.whisper_config/config.json` files are forward-compatible — Pydantic fills missing fields with new defaults
- StreamDeck shell scripts continue to work (they invoke `python -m rt_whisper` which still exists)
- `streamdeck_start.sh` needs minor update: `uv run` instead of direct python, drop venv activation
- StreamDeck Python scripts: shebangs updated for uv
- No data migration needed — log dirs, cache dirs, tmp dirs unchanged

## Testing Strategy

**Test layout:** `tests/` directory at repo root.

**Framework:** pytest (required by standards). hypothesis for property-based testing where it adds value (e.g., config validation edge cases).

**Automated tests (run via `just test`):**

- `tests/test_config.py` — config loading: default config, partial config file, full config file, invalid config (should fail fast), CLI override merging, CUDA validation (mocked torch)
- `tests/test_audio.py` — beep samplerate correctness (mock sounddevice), recording state machine (start -> stop -> path returned)
- `tests/test_commands.py` — command parsing, expiry logic, dispatch to protocol fakes
- `tests/test_transcriber.py` — transcribe call passes correct VAD/beam params (mock WhisperModel)

**Manual tests (documented in README, not automated):**

- Record a few sentences with Yeti Nano, confirm 48kHz capture
- Transcribe and verify quality/latency feel
- WirePlumber config install on live system

**CI note:** Audio device tests use mocked sounddevice — no hardware required. GPU tests mock `torch.cuda.is_available()`.
