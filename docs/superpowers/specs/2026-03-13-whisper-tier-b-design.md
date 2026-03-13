# Whisper Tier B: Control Plane & Hands-Free UX

## Context

Tier A (complete) decomposed the monolithic transcriber into focused modules with config-driven architecture, 48kHz audio alignment, and VAD filtering. The user's core frustration: **"having to run bash commands kills the entire point of using whisper for dictation over typing."** StreamDeck buttons and terminal commands are unacceptable for a dictation tool — the whole point is to avoid touching a keyboard/terminal.

This spec covers Tier B — the control plane that enables true hands-free dictation via keyboard shortcuts, replacing the StreamDeck/bash dependency with a proper CLI tool and IPC mechanism.

## Goals

1. Ship `rtwhisperctl` as an installable console script — single binary for all control operations
2. Add status/ack IPC so control UX is deterministic (not fire-and-forget)
3. Add `toggle` command — one shortcut to start/stop dictation
4. Provide COSMIC desktop keyboard shortcut installation
5. Add `just start` / `just stop` convenience commands
6. Keep StreamDeck scripts working (backward compat) but remove them as the primary interface

## Non-Goals

- System tray icon (future — requires GTK/Qt dependency)
- Hotword/wake word detection (separate feature)
- Local web UI
- systemd user service (nice-to-have, not blocking)
- NeMo Canary/Parakeet model backends (Tier C)

## Design

### 1. `rtwhisperctl` CLI (`rt_whisper/ctl.py`)

A lightweight CLI tool registered as a console script. It writes commands atomically and optionally waits for acknowledgement.

**Entry point:** `pyproject.toml` → `[project.scripts]` → `rtwhisperctl = "rt_whisper.ctl:main"`

**Commands:**

```
rtwhisperctl start              # Send start_recording
rtwhisperctl stop               # Send stop_recording
rtwhisperctl toggle             # Toggle based on status.json
rtwhisperctl status             # Print daemon status (JSON)
rtwhisperctl clipboard          # Send process_clipboard
rtwhisperctl clipboard explain  # Send explain_clipboard
rtwhisperctl clipboard summarize
rtwhisperctl clipboard promptify
rtwhisperctl clipboard reformat
rtwhisperctl clipboard implement
rtwhisperctl clipboard command
rtwhisperctl clipboard translate
rtwhisperctl daemon             # Start the daemon (replaces `python -m rt_whisper`)
```

**Atomic write protocol:**

```python
payload = {
    "command": command_name,
    "command_id": str(uuid4()),
    "timestamp": time.time(),
    "args": {},
}
tmp = CMD_PATH.with_suffix(".tmp")
tmp.write_text(json.dumps(payload))
os.replace(tmp, CMD_PATH)  # Atomic on POSIX
```

**`--wait` flag:** Poll for `ack.json` with matching `command_id` (default timeout: 10s). Exit 0 on success, exit 1 on error/timeout. This makes keyboard shortcut feedback deterministic.

**`--json` flag:** Machine-readable output for `status` command.

### 2. Status IPC (`status.json`)

The daemon writes `~/.whisper_logs/status.json` on every state change. This is the source of truth for `toggle`.

```json
{
    "is_recording": false,
    "pid": 12345,
    "started_at": 1710345600.0,
    "last_command": "stop_recording",
    "last_command_time": 1710345650.0
}
```

**Writers:**
- `Orchestrator.run()` — writes initial status on startup (is_recording=false, pid=os.getpid())
- `CommandHandler._dispatch()` — updates after each command

**Readers:**
- `rtwhisperctl toggle` — reads `is_recording` to decide start vs stop
- `rtwhisperctl status` — prints the file contents

### 3. Ack IPC (`ack.json`)

The daemon writes `~/.whisper_logs/ack.json` after processing each command. This closes the feedback loop for `--wait`.

```json
{
    "command_id": "uuid-here",
    "command": "start_recording",
    "success": true,
    "error": null,
    "timestamp": 1710345660.0
}
```

**Writer:** `CommandHandler.check_for_commands()` — writes ack after `_dispatch()` completes (or on error).

**Reader:** `rtwhisperctl --wait` — polls until `ack.json` appears with matching `command_id`.

### 4. Toggle Logic

```python
def toggle():
    status = read_status()
    if status and status.get("is_recording"):
        write_command("stop_recording")
    else:
        write_command("start_recording")
```

Handles edge cases:
- No `status.json` (daemon not running or first boot): sends `start_recording`
- Stale `status.json` (PID not running): warns and sends `start_recording`

### 5. COSMIC Keyboard Shortcuts

**Script:** `scripts/install-cosmic-shortcuts.sh`

Uses `cosmic-settings` or manual `~/.config/cosmic/` config to register:

| Shortcut | Command |
|---|---|
| `Super+Shift+D` | `rtwhisperctl toggle --wait` |
| `Super+Shift+S` | `rtwhisperctl stop --wait` |
| `Super+Shift+E` | `rtwhisperctl clipboard explain` |
| `Super+Shift+R` | `rtwhisperctl clipboard reformat` |

The script wraps commands with `bash -lc` to ensure PATH resolution when COSMIC runs shortcuts with a limited environment.

**Fallback:** If `cosmic-settings` CLI isn't available, print manual instructions for the COSMIC Settings GUI.

### 6. Updated `check_for_commands()` Flow

```
command.json arrives (atomic write)
  ↓
CommandHandler reads + validates TTL
  ↓
_dispatch(command)
  ↓
_write_status(is_recording=..., last_command=...)
  ↓
_write_ack(command_id, success=True/False, error=...)
  ↓
Delete command.json
```

### 7. Daemon Subcommand

`rtwhisperctl daemon` replaces `python -m rt_whisper`. This means `streamdeck_start.sh` and `just dev` can use `rtwhisperctl daemon` instead. The `__main__.py` entry point still works for backward compat.

### 8. Updated Justfile

```just
start *ARGS:          # rtwhisperctl daemon (background)
stop:                 # rtwhisperctl stop
toggle:               # rtwhisperctl toggle
status:               # rtwhisperctl status
shortcuts:            # scripts/install-cosmic-shortcuts.sh
```

## File Changes

| File | Action |
|---|---|
| `rt_whisper/ctl.py` | **CREATE** — rtwhisperctl CLI |
| `rt_whisper/commands.py` | **MODIFY** — add status/ack writes, parse command_id |
| `rt_whisper/orchestrator.py` | **MODIFY** — write initial status.json |
| `rt_whisper/__main__.py` | **MODIFY** — import ctl.main for daemon subcommand |
| `pyproject.toml` | **MODIFY** — add [project.scripts] entry |
| `Justfile` | **MODIFY** — add start/stop/toggle/status/shortcuts |
| `scripts/install-cosmic-shortcuts.sh` | **CREATE** — COSMIC shortcut installer |
| `streamdeck_start.sh` | **MODIFY** — use rtwhisperctl daemon |
| `tests/test_ctl.py` | **CREATE** — ctl tests |
| `tests/test_commands.py` | **MODIFY** — test status/ack writes |
| `CLAUDE.md` | **MODIFY** — update commands |

## Key Risks

1. **COSMIC shortcut API** — may vary between Pop!_OS versions. Script should detect and fall back to manual instructions.
2. **Race condition on status.json** — `toggle` reads status while daemon writes it. Mitigate with atomic writes (`os.replace`) for status too.
3. **PATH resolution in shortcuts** — COSMIC may run commands with minimal PATH. Wrap with `bash -lc`.
4. **Stale status after daemon crash** — `toggle` should check PID liveness via `os.kill(pid, 0)`.
