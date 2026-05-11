"""
HuggingFace Inference API backend.
Requires: pip install huggingface-hub
Env var: HF_TOKEN

This is also the backend to use for the fine-tuned newsatom model
once it is available on Hugging Face:
  python extract.py --backend huggingface --model sannuta/newsatom-gemma-2b
"""

import os
from .base import ModelBackend

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


class HuggingFaceBackend(ModelBackend):

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "HuggingFace backend requires: pip install huggingface-hub"
            )

        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN environment variable."
            )

        client = InferenceClient(model=self.model, token=token)

        response = client.text_generation(
            prompt,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=True,
            return_full_text=False,
        )

        return response
