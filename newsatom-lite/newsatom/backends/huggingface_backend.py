"""
HuggingFace Inference API backend.
Requires: pip install huggingface-hub. Env var: HF_TOKEN

Use for the fine-tuned newsatom model once available:
  python extract.py --backend huggingface --model sannuta/newsatom-gemma-2b --gemma-format
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
            raise ImportError("HuggingFace backend requires: pip install huggingface-hub")

        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError("Set HF_TOKEN environment variable.")

        client = InferenceClient(model=self.model, token=token)
        return client.text_generation(
            prompt,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=False,
            return_full_text=False,
        )
