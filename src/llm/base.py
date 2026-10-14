"""Abstract LLM backend interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_TOKENS = 2048
MAX_TOKENS_ENV = "AOB_LLM_MAX_TOKENS"

# OpenAI reports truncation as "length"; Anthropic-family providers reached
# through a proxy sometimes leak their native "max_tokens".
_TRUNCATED = frozenset({"length", "max_tokens"})


@dataclass(frozen=True)
class LLMResult:
    """Return type for :meth:`LLMBackend.generate_with_usage`.

    ``input_tokens`` / ``output_tokens`` are ``0`` when the backend can't
    report usage (e.g. mocks in unit tests).
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class EmptyCompletionError(RuntimeError):
    """Raised when a completion carries no visible content.

    Reasoning models spend the completion budget on ``reasoning_content`` as
    well as ``content``, so a demanding question can exhaust the cap and come
    back with ``content: null`` and ``finish_reason: "length"`` regardless of
    prompt size.
    """

    def __init__(
        self,
        *,
        model: str,
        finish_reason: str | None,
        completion_tokens: int,
        max_tokens: int,
    ) -> None:
        self.model = model
        self.finish_reason = finish_reason
        self.completion_tokens = completion_tokens
        self.max_tokens = max_tokens
        self.truncated = finish_reason in _TRUNCATED
        remedy = (
            "The budget was spent before any content was emitted — reasoning "
            f"models charge hidden reasoning against it. Raise {MAX_TOKENS_ENV}."
            if self.truncated
            else "finish_reason does not indicate truncation, so raising "
            f"{MAX_TOKENS_ENV} is unlikely to help."
        )
        super().__init__(
            f"{model} returned an empty completion "
            f"(finish_reason={finish_reason!r}, "
            f"completion_tokens={completion_tokens}, max_tokens={max_tokens}). "
            + remedy
        )


def resolve_max_tokens() -> int:
    """Completion-token cap, overridable via ``AOB_LLM_MAX_TOKENS``."""
    raw = os.environ.get(MAX_TOKENS_ENV)
    if not raw:
        return DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{MAX_TOKENS_ENV} must be a positive integer, got {raw!r}"
        ) from None
    if value <= 0:
        raise ValueError(f"{MAX_TOKENS_ENV} must be a positive integer, got {raw!r}")
    return value


def result_from_response(response: Any, *, model: str, max_tokens: int) -> LLMResult:
    """Build an :class:`LLMResult` from an OpenAI-shaped chat completion.

    Raises:
        EmptyCompletionError: if the completion carries no visible content, so
            a ``None`` never propagates into downstream parsing.
    """
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    content = getattr(choice.message, "content", None)

    if content is None or not content.strip():
        raise EmptyCompletionError(
            model=model,
            finish_reason=getattr(choice, "finish_reason", None),
            completion_tokens=output_tokens,
            max_tokens=max_tokens,
        )

    return LLMResult(
        text=content,
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=output_tokens,
    )


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
