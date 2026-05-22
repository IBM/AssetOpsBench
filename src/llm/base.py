"""Abstract LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generation import GenerationParams


@dataclass(frozen=True)
class LLMResult:
    """Return type for :meth:`LLMBackend.generate_with_usage`.

    ``input_tokens`` / ``output_tokens`` are ``0`` when the backend can't
    report usage (e.g. mocks in unit tests).
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMBackend(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        *,
        params: GenerationParams | None = None,
    ) -> str:
        """Generate text given a prompt.

        Args:
            prompt: The prompt to send to the model.
            temperature: Sampling temperature — **always** overrides any value
                stored in *params* when provided (even ``0.0``).
            params: Optional :class:`~llm.GenerationParams` override applied on
                top of the backend's stored defaults.  ``None`` means use
                backend defaults unchanged.
        """
        ...

    def generate_with_usage(
        self,
        prompt: str,
        temperature: float = 0.0,
        *,
        params: GenerationParams | None = None,
    ) -> LLMResult:
        """Generate text and report token usage.

        Default impl delegates to :meth:`generate` and reports zero usage —
        backends that can surface counts (e.g. LiteLLM) should override.
        """
        return LLMResult(text=self.generate(prompt, temperature, params=params))

    @property
    def model_id(self) -> str:
        """Return the backend's model identifier, or ``"unknown"``.

        Default impl reads ``self._model_id`` if present so existing
        subclasses work without modification.
        """
        return getattr(self, "_model_id", "unknown")
