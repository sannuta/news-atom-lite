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
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {', '.join(BACKENDS.keys())}")
    cls = BACKENDS[name]
    if name == "local":
        return cls(model_path=model_path)
    elif model:
        return cls(model=model)
    else:
        return cls()
