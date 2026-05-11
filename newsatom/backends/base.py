"""
Abstract base class for all model backends.

To add a new backend:
1. Create a new file e.g. newsatom/backends/mymodel_backend.py
2. Subclass ModelBackend and implement generate()
3. Register it in newsatom/backends/__init__.py
"""

from abc import ABC, abstractmethod


class ModelBackend(ABC):
    """Base class for model backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Run inference and return the model's response.

        Args:
            prompt: The full formatted prompt string.

        Returns:
            The model's generated text (the extraction output only,
            not the prompt itself).
        """
        pass
