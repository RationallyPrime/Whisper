"""Main event loop orchestrator for RT-Whisper."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .commands import CommandHandler
    from .config import WhisperConfig

logger = logging.getLogger(__name__)


class Orchestrator:
    """Owns the main polling loop. Delegates command dispatch to CommandHandler."""

    def __init__(self, config: WhisperConfig, command_handler: CommandHandler) -> None:
        self._config = config
        self._command_handler = command_handler

    def run(self) -> None:
        """Poll for commands until interrupted."""
        from .commands import write_status

        logger.info("RT-Whisper started in command mode.")

        write_status(
            self._config.daemon.log_dir,
            is_recording=False,
            pid=os.getpid(),
            started_at=time.time(),
        )

        try:
            while True:
                self._command_handler.check_for_commands()
                time.sleep(self._config.daemon.command_poll_interval)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
