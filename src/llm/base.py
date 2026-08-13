"""Abstract LLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class EmptyCompletionError(RuntimeError):
    """The backend returned a completion with no visible content.

    Raised where it happens rather than letting ``None`` travel. A reasoning
    model can spend its whole token budget on ``reasoning_content`` and return
    ``content=None`` with ``finish_reason='length'``; passed on, that surfaces
    several frames later as ``TypeError: expected string or bytes-like object,
    got 'NoneType'`` inside a regex, which blames the parser for the backend's
    result.
    """

    def __init__(
        self,
        model: str,
        finish_reason: str | None = None,
        completion_tokens: int = 0,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.finish_reason = finish_reason
        self.completion_tokens = completion_tokens
        self.max_tokens = max_tokens

        detail = f"{model} returned no content (finish_reason={finish_reason!r})"
        if finish_reason == "length":
            detail += (
                f"; the completion hit the token cap"
                f"{f' of {max_tokens}' if max_tokens else ''}"
                f" after {completion_tokens} tokens, which reasoning models can"
                f" consume entirely on reasoning_content. Raise the cap with"
                f" AOB_LLM_MAX_TOKENS."
            )
        super().__init__(detail)


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
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text given a prompt."""
        ...

    def generate_with_usage(self, prompt: str, temperature: float = 0.0) -> LLMResult:
        """Generate text and report token usage.

        Default impl delegates to :meth:`generate` and reports zero usage —
        backends that can surface counts (e.g. LiteLLM) should override.
        """
        return LLMResult(text=self.generate(prompt, temperature))

    @property
    def model_id(self) -> str:
        """Return the backend's model identifier, or ``"unknown"``.

        Default impl reads ``self._model_id`` if present so existing
        subclasses work without modification.
        """
        return getattr(self, "_model_id", "unknown")
