"""Tests for rt_whisper.commands — CommandHandler dispatch and stop flow."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from rt_whisper.commands import CommandHandler
from rt_whisper.config import DaemonConfig
from tests.conftest import FakeRecorder, FakeTextProcessor, FakeTranscriber


@pytest.fixture
def daemon_config(tmp_path: Path) -> DaemonConfig:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return DaemonConfig(
        log_dir=log_dir,
        config_dir=tmp_path / "config",
        tmp_dir=tmp_path / "tmp",
    )


@pytest.fixture
def handler(
    daemon_config: DaemonConfig,
    fake_recorder: FakeRecorder,
    fake_transcriber: FakeTranscriber,
    fake_text_processor: FakeTextProcessor,
) -> CommandHandler:
    return CommandHandler(
        daemon_config,
        recorder=fake_recorder,
        transcriber=fake_transcriber,
        text_processor=fake_text_processor,
    )


def _write_command(log_dir: Path, command: str, age: float = 0) -> None:
    """Helper to write a command.json file."""
    command_file = log_dir / "command.json"
    command_file.write_text(json.dumps({"command": command, "timestamp": time.time() - age}))


class TestCommandParsing:
    def test_start_recording(
        self, handler: CommandHandler, daemon_config: DaemonConfig, fake_recorder: FakeRecorder
    ) -> None:
        _write_command(daemon_config.log_dir, "start_recording")
        handler.check_for_commands()
        assert fake_recorder.start_count == 1
        assert fake_recorder.is_recording is True

    def test_no_command_file(self, handler: CommandHandler) -> None:
        # Should not raise
        handler.check_for_commands()

    def test_expired_command_ignored(
        self, handler: CommandHandler, daemon_config: DaemonConfig, fake_recorder: FakeRecorder
    ) -> None:
        _write_command(daemon_config.log_dir, "start_recording", age=10)
        handler.check_for_commands()
        assert fake_recorder.start_count == 0

    def test_command_file_removed_after_processing(
        self, handler: CommandHandler, daemon_config: DaemonConfig
    ) -> None:
        _write_command(daemon_config.log_dir, "start_recording")
        handler.check_for_commands()
        assert not (daemon_config.log_dir / "command.json").exists()

    def test_unknown_command_does_nothing(
        self, handler: CommandHandler, daemon_config: DaemonConfig, fake_recorder: FakeRecorder
    ) -> None:
        _write_command(daemon_config.log_dir, "unknown_command")
        handler.check_for_commands()
        assert fake_recorder.start_count == 0


class TestStopRecording:
    def test_stop_recording_flow(
        self,
        handler: CommandHandler,
        daemon_config: DaemonConfig,
        fake_recorder: FakeRecorder,
        fake_transcriber: FakeTranscriber,
        tmp_path: Path,
    ) -> None:
        # Set up a fake WAV file
        wav = tmp_path / "test.wav"
        wav.write_text("fake audio")
        fake_recorder.set_wav_path(wav)

        # Start then stop recording
        fake_recorder.start_recording()
        _write_command(daemon_config.log_dir, "stop_recording")

        with patch("rt_whisper.commands.pyperclip") as mock_clip:
            handler.check_for_commands()

        assert fake_recorder.stop_count == 1
        assert len(fake_transcriber.transcribed_paths) == 1
        mock_clip.copy.assert_called_once_with("hello world")
        # Success beep (660 Hz)
        assert (660, 0.1) in fake_recorder.beeps

    def test_stop_recording_no_audio(
        self,
        handler: CommandHandler,
        daemon_config: DaemonConfig,
        fake_recorder: FakeRecorder,
    ) -> None:
        fake_recorder.start_recording()
        _write_command(daemon_config.log_dir, "stop_recording")
        handler.check_for_commands()
        # Warning beep (330 Hz) for no audio
        assert (330, 0.2) in fake_recorder.beeps


class TestClaudeCommands:
    def test_process_clipboard(
        self,
        handler: CommandHandler,
        daemon_config: DaemonConfig,
        fake_text_processor: FakeTextProcessor,
    ) -> None:
        _write_command(daemon_config.log_dir, "process_clipboard")
        handler.check_for_commands()
        assert fake_text_processor.process_count == 1

    @pytest.mark.parametrize(
        "command,expected_prefix",
        [
            ("explain_clipboard", "explain this"),
            ("summarize_clipboard", "summarize this"),
            ("promptify_clipboard", "promptify this"),
            ("reformat_clipboard", "reformat this"),
            ("implement_clipboard", "implement this"),
            ("command_clipboard", "command line this"),
            ("translate_clipboard", "translate this into English"),
        ],
    )
    def test_prefixed_commands(
        self,
        handler: CommandHandler,
        daemon_config: DaemonConfig,
        fake_text_processor: FakeTextProcessor,
        command: str,
        expected_prefix: str,
    ) -> None:
        _write_command(daemon_config.log_dir, command)
        with patch("rt_whisper.commands.pyperclip") as mock_clip:
            mock_clip.paste.return_value = "some text"
            handler.check_for_commands()
        mock_clip.copy.assert_called_once_with(f"{expected_prefix}: some text")
        assert fake_text_processor.process_count == 1
