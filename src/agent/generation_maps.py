"""Maps :class:`~llm.GenerationParams` to each agent SDK's native config types.

Import chain is intentionally one-way:
    llm.generation  ← (no agent deps)
    agent.generation_maps  → imports from SDK packages only when called

Three public helpers, one per SDK:

    to_claude_agent_options(options, params, model_id)
        Applies reasoning / thinking + strips-and-warns other params.

    to_model_settings(params, model_id) -> ModelSettings
        Returns an openai-agents ModelSettings.

    to_chat_openai_kwargs(params, model_id) -> dict
        Returns init / bind kwargs for langchain_openai.ChatOpenAI.
"""

from __future__ import annotations

import logging

from llm.generation import (
    GenerationParams,
    EFFORT_TO_OPENAI,
    reasoning_supported,
)

_log = logging.getLogger(__name__)


# ── Claude Agent SDK ──────────────────────────────────────────────────────────


def to_claude_agent_options(
    options,  # claude_agent_sdk.ClaudeAgentOptions (avoid hard import at module level)
    params: GenerationParams,
    model_id: str,
) -> None:
    """Mutate *options* in-place with generation params.

    Claude Agent SDK fields handled natively:
        - ``effort``   → ``options.effort``
        - ``thinking`` → ``options.thinking``

    All other params (max_tokens, temperature, top_p, stop) are forwarded via
    ``options.extra_args`` if non-default, with a warning that support depends
    on the underlying CLI version.
    """
    effort = params.reasoning_effort

    if reasoning_supported(model_id):
        if effort == "none":
            from claude_agent_sdk.types import ThinkingConfigDisabled

            options.thinking = ThinkingConfigDisabled(type="disabled")
            options.effort = None
        elif params.thinking_budget_tokens is not None:
            from claude_agent_sdk.types import ThinkingConfigEnabled

            options.thinking = ThinkingConfigEnabled(
                type="enabled",
                budget_tokens=params.thinking_budget_tokens,
            )
            options.effort = None
        else:
            from claude_agent_sdk.types import ThinkingConfigAdaptive

            options.thinking = ThinkingConfigAdaptive(type="adaptive")
            options.effort = effort  # type: ignore[assignment]
    elif effort != "none":
        _log.warning(
            "reasoning_effort=%r requested but model %r does not support "
            "reasoning on claude-agent — stripping thinking kwargs.",
            effort,
            model_id,
        )

    extra: dict[str, str | None] = dict(options.extra_args or {})

    # max_tokens via extra_args (CLI flag name; strip+warn if unsupported at runtime)
    if params.max_tokens != GenerationParams.max_tokens:
        _log.warning(
            "claude-agent: max_tokens=%d forwarded via extra_args; "
            "support depends on the installed Claude Code CLI version.",
            params.max_tokens,
        )
        extra["max-tokens"] = str(params.max_tokens)

    if params.temperature != GenerationParams.temperature:
        _log.warning(
            "claude-agent: temperature=%.3g — ClaudeAgentOptions has no "
            "native temperature field; stripping.",
            params.temperature,
        )

    if params.top_p is not None:
        _log.warning(
            "claude-agent: top_p=%.3g — ClaudeAgentOptions has no native "
            "top_p field; stripping.",
            params.top_p,
        )

    if params.stop:
        _log.warning(
            "claude-agent: stop sequences — ClaudeAgentOptions has no native "
            "stop field; stripping.",
        )

    options.extra_args = extra


# ── OpenAI Agents SDK ─────────────────────────────────────────────────────────


def to_model_settings(params: GenerationParams, model_id: str):
    """Return an ``agents.ModelSettings`` populated from *params*.

    ``reasoning_effort`` is mapped to ``ModelSettings.reasoning`` with the
    OpenAI-compatible vocab (``max`` → ``xhigh``).  Unsupported models get the
    reasoning field stripped with a warning.
    """
    from agents import ModelSettings
    from openai.types.shared import Reasoning

    kwargs: dict = {
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
    }

    if params.top_p is not None:
        kwargs["top_p"] = params.top_p

    if params.stop:
        kwargs["extra_args"] = {"stop": list(params.stop)}

    effort = params.reasoning_effort
    if effort != "none":
        if reasoning_supported(model_id):
            openai_effort = EFFORT_TO_OPENAI[effort]
            kwargs["reasoning"] = Reasoning(effort=openai_effort)  # type: ignore[arg-type]
        else:
            _log.warning(
                "reasoning_effort=%r requested but model %r does not support "
                "reasoning on openai-agent — stripping.",
                effort,
                model_id,
            )

    return ModelSettings(**kwargs)


# ── LangChain ChatOpenAI (deep-agent) ─────────────────────────────────────────


def to_chat_openai_kwargs(params: GenerationParams, model_id: str) -> dict:
    """Return init / ``.bind()`` kwargs for ``langchain_openai.ChatOpenAI``.

    Passes generation params through ``model_kwargs`` so the LiteLLM proxy
    (which presents an OpenAI-compatible interface) forwards them correctly.
    """
    kwargs: dict = {
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
    }

    if params.top_p is not None:
        kwargs["top_p"] = params.top_p

    if params.stop:
        kwargs["stop"] = list(params.stop)

    effort = params.reasoning_effort
    if effort != "none":
        if reasoning_supported(model_id):
            kwargs["reasoning_effort"] = EFFORT_TO_OPENAI[effort]
        else:
            _log.warning(
                "reasoning_effort=%r requested but model %r does not support "
                "reasoning on deep-agent — stripping.",
                effort,
                model_id,
            )

    return kwargs
