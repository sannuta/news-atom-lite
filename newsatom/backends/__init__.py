"""
Backend registry — maps backend names to classes.
Add new backends here.
"""

from .openai_backend import OpenAIBackend
from .anthropic_backend import AnthropicBackend
from .huggingface_backend import HuggingFaceBackend
from .ollama_backend import OllamaBackend
from .local_backend import LocalBackend


BACKENDS = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "huggingface": HuggingFaceBackend,
    "ollama": OllamaBackend,
    "local": LocalBackend,
}


def get_backend(name: str, model: str = None, model_path: str = None):
    """
    Instantiate and return a backend by name.

    Args:
        name:       Backend key e.g. "openai", "anthropic", "ollama"
        model:      Model name or ID (optional, uses backend default if not set)
        model_path: Path to local model directory (local backend only)
    """
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. "
            f"Available: {', '.join(BACKENDS.keys())}"
        )

    cls = BACKENDS[name]

    if name == "local":
        return cls(model_path=model_path)
    elif model:
        return cls(model=model)
    else:
        return cls()
