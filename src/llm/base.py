"""Abstract LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text given a prompt."""
        ...

    @property
    def model_id(self) -> str:
        """Return the backend's model identifier, or ``"unknown"``.

        Default impl reads ``self._model_id`` if present so existing
        subclasses work without modification.
        """
        return getattr(self, "_model_id", "unknown")
