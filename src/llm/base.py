"""Abstract LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResult:
    """Return type for :meth:`LLMBackend.generate` / :meth:`generate_with_usage`.

    ``input_tokens`` / ``output_tokens`` / ``total_tokens`` are ``0`` when the
    backend can't report usage (e.g. mocks in unit tests). ``total_tokens``
    should reflect the provider total when available; otherwise callers may use
    ``input_tokens + output_tokens``.
    """
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class LLMBackend(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Generate text given a prompt; includes usage when the backend provides it."""
        ...

    def generate_with_usage(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Generate text and report token usage.

        Default impl delegates to :meth:`generate` (same fields as usage-aware
        backends).
        """
        return self.generate(prompt, temperature, max_tokens)

    @property
    def model_id(self) -> str:
        """Return the backend's model identifier, or ``"unknown"``.

        Default impl reads ``self._model_id`` if present so existing
        subclasses work without modification.
        """
        return getattr(self, "_model_id", "unknown")
