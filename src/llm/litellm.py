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

import os

from .base import EmptyCompletionError, LLMBackend, LLMResult

_WATSONX_PREFIX = "watsonx/"


class LiteLLMBackend(LLMBackend):
    """LLM backend using the litellm library.

    Args:
        model_id: litellm model string with provider prefix, e.g.:
                  ``"watsonx/meta-llama/llama-3-3-70b-instruct"``
                  ``"litellm_proxy/GCP/claude-4-sonnet"``
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self.generate_with_usage(prompt, temperature).text

    def generate_with_usage(self, prompt: str, temperature: float = 0.0) -> LLMResult:
        import litellm

        kwargs: dict = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            # Overridable: 2048 is comfortable for a non-reasoning model and
            # too tight for one that emits reasoning_content, where the visible
            # answer is what is left after the thinking is paid for.
            "max_tokens": int(os.environ.get("AOB_LLM_MAX_TOKENS", "2048")),
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
        choice = response.choices[0]
        text = choice.message.content
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        # Fail where the fact is, not four frames later. `content` is None
        # whenever the model produced no visible text -- most often a reasoning
        # model that spent the whole cap on reasoning_content and stopped with
        # finish_reason='length'. Returning it lets a TypeError surface inside
        # parse_plan's regex, blaming the parser for the backend's result.
        if text is None:
            raise EmptyCompletionError(
                model=self._model_id,
                finish_reason=getattr(choice, "finish_reason", None),
                completion_tokens=completion_tokens,
                max_tokens=kwargs.get("max_tokens"),
            )

        return LLMResult(
            text=text,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=completion_tokens,
        )
