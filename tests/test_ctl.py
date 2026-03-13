"""Tests for rt_whisper.ctl — rtwhisperctl CLI and IPC."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from rt_whisper.ctl import (
    _CLIPBOARD_COMMAND_MAP,
    _pid_alive,
    build_parser,
    read_status,
    wait_for_ack,
    write_command,
)


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _override_paths(log_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point all IPC paths to the temp directory."""
    monkeypatch.setattr("rt_whisper.ctl.LOG_DIR", log_dir)
    monkeypatch.setattr("rt_whisper.ctl.CMD_PATH", log_dir / "command.json")
    monkeypatch.setattr("rt_whisper.ctl.STATUS_PATH", log_dir / "status.json")
    monkeypatch.setattr("rt_whisper.ctl.ACK_PATH", log_dir / "ack.json")


class TestWriteCommand:
    def test_atomic_write_creates_command_file(self, log_dir: Path) -> None:
        cmd_id = write_command("start_recording")
        cmd_path = log_dir / "command.json"
        assert cmd_path.exists()
        data = json.loads(cmd_path.read_text())
        assert data["command"] == "start_recording"
        assert data["command_id"] == cmd_id
        assert data["timestamp"] <= time.time()
        assert data["args"] == {}

    def test_no_temp_file_left_behind(self, log_dir: Path) -> None:
        write_command("stop_recording")
        tmp = log_dir / "command.tmp"
        assert not tmp.exists()

    def test_command_id_is_unique(self, log_dir: Path) -> None:
        id1 = write_command("start_recording")
        id2 = write_command("stop_recording")
        assert id1 != id2

    def test_args_passed_through(self, log_dir: Path) -> None:
        write_command("start_recording", args={"key": "value"})
        data = json.loads((log_dir / "command.json").read_text())
        assert data["args"] == {"key": "value"}


class TestReadStatus:
    def test_returns_none_when_missing(self) -> None:
        assert read_status() is None

    def test_reads_valid_status(self, log_dir: Path) -> None:
        status = {"is_recording": True, "pid": 1234, "started_at": 100.0}
        (log_dir / "status.json").write_text(json.dumps(status))
        result = read_status()
        assert result is not None
        assert result["is_recording"] is True
        assert result["pid"] == 1234

    def test_returns_none_on_invalid_json(self, log_dir: Path) -> None:
        (log_dir / "status.json").write_text("not json")
        assert read_status() is None


class TestPidAlive:
    def test_current_process_is_alive(self) -> None:
        assert _pid_alive(os.getpid()) is True

    def test_nonexistent_pid(self) -> None:
        assert _pid_alive(99999999) is False


class TestWaitForAck:
    def test_returns_ack_on_match(self, log_dir: Path) -> None:
        cmd_id = "test-uuid-123"
        ack_data = {"command_id": cmd_id, "command": "start_recording", "success": True}
        (log_dir / "ack.json").write_text(json.dumps(ack_data))
        result = wait_for_ack(cmd_id, timeout=1.0)
        assert result is not None
        assert result["command_id"] == cmd_id

    def test_returns_none_on_timeout(self) -> None:
        result = wait_for_ack("nonexistent-id", timeout=0.1)
        assert result is None

    def test_ignores_mismatched_command_id(self, log_dir: Path) -> None:
        ack_data = {"command_id": "other-id", "command": "start_recording", "success": True}
        (log_dir / "ack.json").write_text(json.dumps(ack_data))
        result = wait_for_ack("my-id", timeout=0.2)
        assert result is None


class TestToggle:
    def test_toggle_when_recording_sends_stop(
        self,
        log_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        status = {"is_recording": True, "pid": os.getpid(), "started_at": 100.0}
        (log_dir / "status.json").write_text(json.dumps(status))
        with patch("rt_whisper.ctl.sys.exit"):
            from rt_whisper.ctl import _do_toggle

            _do_toggle(wait=False)
        data = json.loads((log_dir / "command.json").read_text())
        assert data["command"] == "stop_recording"

    def test_toggle_when_idle_sends_start(self, log_dir: Path) -> None:
        status = {"is_recording": False, "pid": os.getpid(), "started_at": 100.0}
        (log_dir / "status.json").write_text(json.dumps(status))
        from rt_whisper.ctl import _do_toggle

        _do_toggle(wait=False)
        data = json.loads((log_dir / "command.json").read_text())
        assert data["command"] == "start_recording"

    def test_toggle_no_status_sends_start(self, log_dir: Path) -> None:
        from rt_whisper.ctl import _do_toggle

        _do_toggle(wait=False)
        data = json.loads((log_dir / "command.json").read_text())
        assert data["command"] == "start_recording"

    def test_toggle_stale_pid_sends_start(
        self,
        log_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        status = {"is_recording": True, "pid": 99999999, "started_at": 100.0}
        (log_dir / "status.json").write_text(json.dumps(status))
        from rt_whisper.ctl import _do_toggle

        _do_toggle(wait=False)
        output = capsys.readouterr().out
        assert "stale" in output.lower()
        data = json.loads((log_dir / "command.json").read_text())
        assert data["command"] == "start_recording"


class TestClipboardSubcommands:
    @pytest.mark.parametrize("subcommand,expected", list(_CLIPBOARD_COMMAND_MAP.items()))
    def test_clipboard_subcommand_maps_correctly(
        self, log_dir: Path, subcommand: str, expected: str
    ) -> None:
        from rt_whisper.ctl import _do_clipboard

        _do_clipboard(subcommand, wait=False)
        data = json.loads((log_dir / "command.json").read_text())
        assert data["command"] == expected

    def test_clipboard_no_subcommand_sends_process(self, log_dir: Path) -> None:
        from rt_whisper.ctl import _do_clipboard

        _do_clipboard(None, wait=False)
        data = json.loads((log_dir / "command.json").read_text())
        assert data["command"] == "process_clipboard"


class TestParser:
    def test_start_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["start"])
        assert args.command == "start"

    def test_toggle_with_wait(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--wait", "toggle"])
        assert args.command == "toggle"
        assert args.wait is True

    def test_clipboard_explain(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["clipboard", "explain"])
        assert args.command == "clipboard"
        assert args.subcommand == "explain"

    def test_status_with_json(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--json", "status"])
        assert args.command == "status"
        assert args.json_output is True

    def test_daemon_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["daemon"])
        assert args.command == "daemon"
