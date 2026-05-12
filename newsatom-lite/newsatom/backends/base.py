from abc import ABC, abstractmethod


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Run inference and return the model's response (generated text only, not the prompt)."""
        pass
