"""Claude API client for RT-Whisper implementing the TextProcessor protocol."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import pyperclip
from anthropic import AsyncAnthropic

from .prompts import SYSTEM_PROMPTS

if TYPE_CHECKING:
    from .config import ClaudeConfig
    from .protocols import Recorder

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Client for interacting with Anthropic's Claude API.

    Implements the TextProcessor protocol. Uses a Recorder for audio feedback.
    """

    def __init__(self, config: ClaudeConfig, recorder: Recorder) -> None:
        self.client = AsyncAnthropic(api_key=config.api_key)
        self._config = config
        self._recorder = recorder
        self.system_prompts = SYSTEM_PROMPTS
        logger.info("Claude client initialized")

    async def get_response(self, text: str) -> str:
        """Get response from Claude for the given text.

        Args:
            text: The input text to process.

        Returns:
            The processed response from Claude.
        """
        try:
            logger.info("Processing text with Claude: %s...", text[:100])

            system_prompt = self._select_system_prompt(text)

            response = await self.client.messages.create(
                model=self._config.model,
                max_tokens=4096,
                system=system_prompt if system_prompt else "",
                messages=[{"role": "user", "content": text}],
            )

            response_text = response.content[0].text
            logger.info("Received response from Claude API (%d chars)", len(response_text))

            if text.lower().startswith("reformat this:"):
                notes = re.findall(r"\[Note:.*?\]", response_text)
                for note in notes:
                    logger.info("Transcription note: %s", note)
                response_text = re.sub(r"\[Note:.*?\]\s*", "", response_text)

            pyperclip.copy(response_text)
            self._recorder.play_beep(440, 0.1)

            for handler in logging.getLogger().handlers:
                handler.flush()

            return response_text
        except Exception:
            logger.exception("Claude API error")
            self._recorder.play_beep(220, 0.2)
            return "Error: Claude API call failed"

    async def process_clipboard(self) -> None:
        """Process the current clipboard content with Claude."""
        try:
            text = pyperclip.paste()
            if not text:
                logger.warning("No text in clipboard")
                self._recorder.play_beep(220, 0.2)
                return

            await self.get_response(text)
            logger.info("Response copied to clipboard")
        except Exception:
            logger.exception("Claude processing error")
            self._recorder.play_beep(220, 0.2)

    @staticmethod
    def _select_system_prompt(text: str) -> str | None:
        """Select the appropriate system prompt based on text prefix."""
        lower = text.lower()
        prefix_map = {
            "promptify this": "promptify",
            "reformat this": "reformat",
            "implement this": "implement",
            "command line this": "command",
            "explain this": "explain",
            "translate this into": "translate",
            "summarize this": "summarize",
        }
        for prefix, key in prefix_map.items():
            if lower.startswith(prefix):
                return SYSTEM_PROMPTS.get(key)
        return None
