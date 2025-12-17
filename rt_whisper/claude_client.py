"""Claude API client for RT-Whisper."""

import logging
import re
import pyperclip
import asyncio
import sounddevice as sd
import numpy as np
from anthropic import AsyncAnthropic
from typing import Optional, Dict, Any
from .prompts import SYSTEM_PROMPTS

class ClaudeClient:
    """Client for interacting with Anthropic's Claude API."""

    def __init__(self, api_key: str):
        """Initialize the Claude client with API key.
        
        Args:
            api_key: Anthropic API key
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.system_prompts = SYSTEM_PROMPTS
        logging.info("Claude client initialized")

    async def get_response(self, text: str) -> str:
        """Get response from Claude for the given text.
        
        Args:
            text: The input text to process
            
        Returns:
            The processed response from Claude
        """
        try:
            # Log the input
            logging.info(f"Processing text with Claude: {text[:100]}...")

            # Determine which system prompt to use but keep the prefix in content
            system_prompt = None
            content = text  # Keep the full text including prefix

            if text.lower().startswith("promptify this"):
                system_prompt = self.system_prompts["promptify"]
            elif text.lower().startswith("reformat this"):
                system_prompt = self.system_prompts["reformat"]
            elif text.lower().startswith("implement this"):
                system_prompt = self.system_prompts["implement"]
            elif text.lower().startswith("command line this"):
                system_prompt = self.system_prompts["command"]
            elif text.lower().startswith("explain this"):
                system_prompt = self.system_prompts["explain"]
            elif text.lower().startswith("translate this into"):
                system_prompt = self.system_prompts["translate"]
            elif text.lower().startswith("summarize this"):
                system_prompt = self.system_prompts["summarize"]

            # Add more detailed logging
            logging.debug(
                f"System prompt selected: {system_prompt[:100] if system_prompt else 'None'}"
            )
            logging.debug(f"Content being sent: {content[:100]}")

            # Create message with the correct format
            response = await self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                system=system_prompt if system_prompt else "",
                messages=[
                    {
                        "role": "user",
                        "content": content,  # Send the full text including prefix
                    }
                ],
            )

            response_text = response.content[0].text
            logging.info(
                f"Received response from Claude API ({len(response_text)} chars)"
            )

            if text.lower().startswith("reformat this:"):
                # Log any [Note: ...] content
                notes = re.findall(r"\[Note:.*?\]", response_text)
                for note in notes:
                    logging.info(f"Transcription note: {note}")

                # Remove [Note: ...] from the response
                response_text = re.sub(r"\[Note:.*?\]\s*", "", response_text)

            # Copy to clipboard and play a quiet beep to indicate completion
            pyperclip.copy(response_text)
            self._play_success_beep()

            # Force flush the logs
            for handler in logging.getLogger().handlers:
                handler.flush()

            return response_text
        except Exception as e:
            error_msg = f"Claude API error: {e}"
            logging.error(error_msg)
            self._play_error_beep()
            return f"Error: {str(e)}"
    
    def _play_success_beep(self):
        """Play a quiet beep to indicate successful processing."""
        try:
            sd.default.samplerate = 44100
            duration = 0.1  # seconds
            frequency = 440  # Hz (A4 note)
            t = np.linspace(0, duration, int(44100 * duration))
            beep = np.sin(2 * np.pi * frequency * t) * 0.3
            sd.play(beep, 44100)
            sd.wait()
        except Exception as e:
            logging.warning(f"Could not play success beep: {e}")
    
    def _play_error_beep(self):
        """Play an error beep to indicate an error occurred."""
        try:
            sd.default.samplerate = 44100
            duration = 0.2  # seconds
            frequency = 220  # Hz (lower pitch for error)
            t = np.linspace(0, duration, int(44100 * duration))
            beep = np.sin(2 * np.pi * frequency * t) * 0.3
            sd.play(beep, 44100)
            sd.wait()
        except Exception as e:
            logging.warning(f"Could not play error beep: {e}")
    
    async def process_clipboard(self, prefix: Optional[str] = None):
        """Process the current clipboard content with Claude.
        
        Args:
            prefix: Optional prefix to add to the clipboard content
        """
        try:
            text = pyperclip.paste()
            if not text:
                logging.warning("No text in clipboard")
                self._play_error_beep()
                return
            
            if prefix:
                text = f"{prefix}: {text}"
            
            # Get Claude's response
            response = await self.get_response(text)
            logging.info("Response copied to clipboard")
            
        except Exception as e:
            logging.error(f"Claude processing error: {e}")
            self._play_error_beep() 
