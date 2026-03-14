"""rtwhisperctl — CLI control plane for the RT-Whisper daemon.

Writes commands atomically via os.replace() and optionally waits
for acknowledgement from the daemon.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

LOG_DIR = Path.home() / ".whisper_logs"
CMD_PATH = LOG_DIR / "command.json"
STATUS_PATH = LOG_DIR / "status.json"
ACK_PATH = LOG_DIR / "ack.json"

_ACK_POLL_INTERVAL = 0.05  # 50ms
_ACK_DEFAULT_TIMEOUT = 10.0

# Maps clipboard subcommand names to the command string sent to the daemon.
_CLIPBOARD_COMMAND_MAP: dict[str, str] = {
    "explain": "explain_clipboard",
    "summarize": "summarize_clipboard",
    "promptify": "promptify_clipboard",
    "reformat": "reformat_clipboard",
    "implement": "implement_clipboard",
    "command": "command_clipboard",
    "translate": "translate_clipboard",
}


def write_command(command: str, args: dict[str, str] | None = None) -> str:
    """Atomically write a command file. Returns the command_id."""
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    command_id = str(uuid4())
    payload = {
        "command": command,
        "command_id": command_id,
        "timestamp": time.time(),
        "args": args or {},
    }
    tmp = CMD_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, CMD_PATH)
    return command_id


def read_status() -> dict[str, object] | None:
    """Read the daemon status file. Returns None if missing or unreadable."""
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def wait_for_ack(
    command_id: str,
    timeout: float = _ACK_DEFAULT_TIMEOUT,
) -> dict[str, object] | None:
    """Poll for ack.json with matching command_id. Returns ack data or None on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            data = json.loads(ACK_PATH.read_text(encoding="utf-8"))
            if data.get("command_id") == command_id:
                return data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        time.sleep(_ACK_POLL_INTERVAL)
    return None


# ── Command handlers ───────────────────────────────────────────────


def _do_start(wait: bool) -> int:
    cmd_id = write_command("start_recording")
    print("Sent: start_recording")
    return _handle_wait(cmd_id, wait)


def _do_stop(wait: bool) -> int:
    cmd_id = write_command("stop_recording")
    print("Sent: stop_recording")
    return _handle_wait(cmd_id, wait)


def _do_toggle(wait: bool) -> int:
    status = read_status()

    if status:
        pid = status.get("pid")
        if pid and not _pid_alive(int(pid)):
            print("Warning: daemon PID not running (stale status). Sending start.")
            cmd_id = write_command("start_recording")
            print("Sent: start_recording")
            return _handle_wait(cmd_id, wait)

        if status.get("is_recording"):
            return _do_stop(wait)

    return _do_start(wait)


def _do_status(as_json: bool) -> int:
    status = read_status()
    if status is None:
        print("No status — daemon may not be running.")
        return 1

    pid = status.get("pid")
    alive = _pid_alive(int(pid)) if pid else False

    if as_json:
        status["daemon_alive"] = alive
        print(json.dumps(status, indent=2))
    else:
        state = "RECORDING" if status.get("is_recording") else "IDLE"
        daemon_state = "running" if alive else "NOT running"
        print(f"State: {state}")
        print(f"Daemon: {daemon_state} (PID {pid})")
        last_cmd = status.get("last_command", "—")
        print(f"Last command: {last_cmd}")

    return 0


def _do_clipboard(subcommand: str | None, wait: bool) -> int:
    if subcommand is None:
        cmd_id = write_command("process_clipboard")
        print("Sent: process_clipboard")
    elif subcommand in _CLIPBOARD_COMMAND_MAP:
        cmd_id = write_command(_CLIPBOARD_COMMAND_MAP[subcommand])
        print(f"Sent: {_CLIPBOARD_COMMAND_MAP[subcommand]}")
    else:
        print(f"Unknown clipboard subcommand: {subcommand}")
        print(f"Available: {', '.join(_CLIPBOARD_COMMAND_MAP)}")
        return 1
    return _handle_wait(cmd_id, wait)


def _do_daemon(daemon_argv: list[str]) -> int:
    """Start the RT-Whisper daemon (delegates to __main__.main).

    Args:
        daemon_argv: Arguments to forward to the daemon (e.g. --device, --model).
    """
    from .__main__ import main as daemon_main

    daemon_main(daemon_argv)
    return 0


def _handle_wait(command_id: str, wait: bool) -> int:
    """If --wait, poll for ack and report result."""
    if not wait:
        return 0

    ack = wait_for_ack(command_id)
    if ack is None:
        print("Timeout waiting for daemon acknowledgement.")
        return 1

    if ack.get("success"):
        print(f"Acknowledged: {ack.get('command')}")
        return 0

    error = ack.get("error", "unknown error")
    print(f"Command failed: {error}")
    return 1


# ── CLI ────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for rtwhisperctl."""
    parser = argparse.ArgumentParser(
        prog="rtwhisperctl",
        description="Control the RT-Whisper dictation daemon",
    )
    parser.add_argument(
        "--wait",
        "-w",
        action="store_true",
        help="Wait for daemon acknowledgement before exiting",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Machine-readable JSON output (for status command)",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start", help="Start recording")
    sub.add_parser("stop", help="Stop recording and transcribe")
    sub.add_parser("toggle", help="Toggle recording on/off")
    sub.add_parser("status", help="Show daemon status")
    sub.add_parser("daemon", help="Start the RT-Whisper daemon")

    clip = sub.add_parser("clipboard", help="Process clipboard with Claude")
    clip.add_argument(
        "subcommand",
        nargs="?",
        default=None,
        choices=list(_CLIPBOARD_COMMAND_MAP),
        help="Clipboard processing mode (omit for generic process_clipboard)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for rtwhisperctl."""
    parser = build_parser()
    # parse_known_args so daemon can forward extra flags (--device, --model, etc.)
    args, remaining = parser.parse_known_args(argv)

    if args.command == "start":
        code = _do_start(args.wait)
    elif args.command == "stop":
        code = _do_stop(args.wait)
    elif args.command == "toggle":
        code = _do_toggle(args.wait)
    elif args.command == "status":
        code = _do_status(args.json_output)
    elif args.command == "clipboard":
        code = _do_clipboard(args.subcommand, args.wait)
    elif args.command == "daemon":
        code = _do_daemon(remaining)
    else:
        parser.print_help()
        code = 1

    sys.exit(code)
