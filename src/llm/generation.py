"""Generation parameters shared across all LLM backends and agent runners.

Single source of truth for configurable generation knobs.  Every backend and
runner reads from :class:`GenerationParams`, which is populated from env vars
via :func:`from_env` and merged with per-constructor / per-call overrides via
:func:`resolve_params`.

Environment variables (all optional):

    LLM_MAX_TOKENS            int   (default 4096)
    LLM_TEMPERATURE           float (default 0.0)
    LLM_TOP_P                 float (default: omit)
    LLM_REASONING_EFFORT      none|low|medium|high|max  (default: none)
    LLM_THINKING_BUDGET_TOKENS int  (default: omit)
    LLM_STOP                  comma-separated stop sequences (default: omit)
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, replace
from typing import Literal

_log = logging.getLogger(__name__)

ReasoningEffort = Literal["none", "low", "medium", "high", "max"]

# Maps our canonical effort labels to the OpenAI Agents SDK's Reasoning.effort vocab.
EFFORT_TO_OPENAI: dict[str, str] = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "xhigh",
}

# Model-id substrings that indicate a reasoning-capable model (checked on lowercased id).
_REASONING_SUBSTRINGS = (
    "claude",
    "anthropic",
    "o1",
    "o3",
    "gpt-5",
    "gpt-o",
)

# Prefixes whose models are never reasoning-capable regardless of the rest of the id.
_NO_REASONING_PREFIXES = ("watsonx/",)


def reasoning_supported(model_id: str) -> bool:
    """Return ``True`` when *model_id* is known to support reasoning / thinking.

    Heuristic: checks the full model string (including ``litellm_proxy/...``
    tails) against known reasoning-capable families.  WatsonX is always False.
    Unknown models return False (safe default — unknown effort will be stripped
    with a warning at the mapper level).
    """
    lower = model_id.lower()
    for prefix in _NO_REASONING_PREFIXES:
        if lower.startswith(prefix):
            return False
    for substr in _REASONING_SUBSTRINGS:
        if substr in lower:
            return True
    return False


@dataclass(frozen=True)
class GenerationParams:
    """Immutable generation configuration shared across all backends/runners.

    Optional fields (``top_p``, ``stop``, ``thinking_budget_tokens``) default
    to ``None``, which means **omit from API calls** — they are not sent to the
    provider unless explicitly set.

    Merge instances with :func:`resolve_params`; the explicit ``temperature``
    argument on :meth:`~llm.LLMBackend.generate` **always** overrides this.
    """

    max_tokens: int = 4096
    temperature: float = 0.0
    reasoning_effort: ReasoningEffort = "none"
    top_p: float | None = None
    thinking_budget_tokens: int | None = None
    stop: tuple[str, ...] | None = None


def from_env() -> GenerationParams:
    """Build a :class:`GenerationParams` from environment variables.

    All variables are optional; missing or empty ones fall back to the
    :class:`GenerationParams` dataclass defaults.
    """
    kwargs: dict = {}

    if raw := os.environ.get("LLM_MAX_TOKENS"):
        kwargs["max_tokens"] = int(raw)

    if raw := os.environ.get("LLM_TEMPERATURE"):
        kwargs["temperature"] = float(raw)

    if raw := os.environ.get("LLM_TOP_P"):
        kwargs["top_p"] = float(raw)

    if raw := os.environ.get("LLM_REASONING_EFFORT"):
        effort = raw.strip().lower()
        if effort not in ("none", "low", "medium", "high", "max"):
            _log.warning(
                "LLM_REASONING_EFFORT=%r is not a valid value "
                "(none|low|medium|high|max); using 'none'.",
                raw,
            )
            effort = "none"
        kwargs["reasoning_effort"] = effort

    if raw := os.environ.get("LLM_THINKING_BUDGET_TOKENS"):
        kwargs["thinking_budget_tokens"] = int(raw)

    if raw := os.environ.get("LLM_STOP"):
        parts = tuple(s.strip() for s in raw.split(",") if s.strip())
        if parts:
            kwargs["stop"] = parts

    return GenerationParams(**kwargs)


def resolve_params(
    base: GenerationParams,
    *,
    override: GenerationParams | None = None,
    temperature: float | None = None,
) -> GenerationParams:
    """Merge *override* onto *base*, then optionally pin *temperature*.

    Merge semantics:
    - ``None`` fields in *override* **do not** replace values in *base*
      (None means "not set / omit").
    - Non-``None`` override fields always win over *base*.
    - *temperature*, when provided as a keyword argument, **always** wins
      over any merged value — matches the ``generate(prompt, temperature=…)``
      contract.

    Returns a new frozen :class:`GenerationParams`.
    """
    merged = base

    if override is not None:
        changes = {
            f.name: getattr(override, f.name)
            for f in dataclasses.fields(override)
            if getattr(override, f.name) is not None
        }
        if changes:
            merged = replace(merged, **changes)

    if temperature is not None:
        merged = replace(merged, temperature=temperature)

    return merged
