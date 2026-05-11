"""
OpenAI backend.
Requires: pip install openai
Env var: OPENAI_API_KEY
"""

import os
from .base import ModelBackend

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a precise structured data extractor for journalism. "
    "Follow the extraction instructions exactly. "
    "Output only valid JSON objects as instructed. "
    "No preamble, no explanation, no markdown formatting."
)


class OpenAIBackend(ModelBackend):

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI backend requires: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )

        return response.choices[0].message.content
