"""Unified LLM backend via the litellm library.

Supports any model string that litellm recognizes.  The provider is encoded
in the model-string prefix — no separate platform flag is needed:

    watsonx/meta-llama/llama-3-3-70b-instruct   → IBM WatsonX
    litellm_proxy/GCP/claude-4-sonnet            → LiteLLM proxy

Credentials are resolved from environment variables based on the prefix:

    watsonx/*  :  WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL (optional)
    otherwise  :  LITELLM_API_KEY, LITELLM_BASE_URL
"""

from __future__ import annotations

import logging
import os

from .base import LLMBackend, LLMResult
from .generation import (
    GenerationParams,
    from_env,
    reasoning_supported,
    resolve_params,
)

_log = logging.getLogger(__name__)
_WATSONX_PREFIX = "watsonx/"


def to_litellm_kwargs(model_id: str, params: GenerationParams) -> dict:
    """Build the extra kwargs dict to pass to ``litellm.completion``.

    Maps :class:`GenerationParams` → litellm-shaped parameters, gating
    reasoning/thinking fields by :func:`~.generation.reasoning_supported`.
    Strips and warns when reasoning is requested but the model doesn't support it.
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
            kwargs["reasoning_effort"] = effort
        else:
            _log.warning(
                "reasoning_effort=%r requested but model %r does not support "
                "reasoning — stripping thinking kwargs.",
                effort,
                model_id,
            )
    elif reasoning_supported(model_id):
        # Explicitly disable thinking on supported models so behaviour is
        # deterministic rather than provider-default.
        kwargs["thinking"] = {"type": "disabled"}

    if params.thinking_budget_tokens is not None:
        if reasoning_supported(model_id) and effort != "none":
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": params.thinking_budget_tokens,
            }
        elif params.thinking_budget_tokens is not None:
            _log.warning(
                "thinking_budget_tokens set but reasoning_effort=%r or model "
                "%r does not support thinking — ignoring budget.",
                effort,
                model_id,
            )

    return kwargs


class LiteLLMBackend(LLMBackend):
    """LLM backend using the litellm library.

    Args:
        model_id: litellm model string with provider prefix, e.g.:
                  ``"watsonx/meta-llama/llama-3-3-70b-instruct"``
                  ``"litellm_proxy/GCP/claude-4-sonnet"``
        params: Generation parameters.  Defaults to :func:`~.generation.from_env`
                when not provided.
    """

    def __init__(
        self,
        model_id: str,
        params: GenerationParams | None = None,
    ) -> None:
        self._model_id = model_id
        self._params: GenerationParams = params if params is not None else from_env()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        *,
        params: GenerationParams | None = None,
    ) -> str:
        return self.generate_with_usage(prompt, temperature, params=params).text

    def generate_with_usage(
        self,
        prompt: str,
        temperature: float = 0.0,
        *,
        params: GenerationParams | None = None,
    ) -> LLMResult:
        import litellm

        effective = resolve_params(
            self._params, override=params, temperature=temperature
        )
        extra = to_litellm_kwargs(self._model_id, effective)

        kwargs: dict = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            **extra,
        }

        if self._model_id.startswith(_WATSONX_PREFIX):
            kwargs["api_key"] = os.environ["WATSONX_APIKEY"]
            kwargs["project_id"] = os.environ["WATSONX_PROJECT_ID"]
            if url := os.environ.get("WATSONX_URL"):
                kwargs["api_base"] = url
        else:
            kwargs["api_key"] = os.environ["LITELLM_API_KEY"]
            kwargs["api_base"] = os.environ["LITELLM_BASE_URL"]

        response = litellm.completion(**kwargs)
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=response.choices[0].message.content,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
