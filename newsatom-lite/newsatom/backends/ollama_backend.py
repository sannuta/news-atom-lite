"""Ollama backend. Requires Ollama running locally (https://ollama.com). No API key needed."""
from .base import ModelBackend

DEFAULT_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"


class OllamaBackend(ModelBackend):

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            import requests
        except ImportError:
            raise ImportError("Ollama backend requires: pip install requests")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 4096},
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Could not connect to Ollama. Run: ollama serve")

        return response.json().get("response", "")
