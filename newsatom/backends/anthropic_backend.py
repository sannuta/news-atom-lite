"""
Anthropic backend.
Requires: pip install anthropic
Env var: ANTHROPIC_API_KEY
"""

import os
from .base import ModelBackend

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a precise structured data extractor for journalism. "
    "Follow the extraction instructions exactly. "
    "Output only valid JSON objects as instructed. "
    "No preamble, no explanation, no markdown formatting."
)


class AnthropicBackend(ModelBackend):

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic backend requires: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1 if "claude-3" not in self.model else None,
        )

        return message.content[0].text
