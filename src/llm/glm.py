"""Text generation with Z.ai's OpenAI-compatible GLM API."""

from __future__ import annotations

import os

from .base import LLMBackend, LLMResult

DEFAULT_GLM_MODEL = "glm-5.3"
_DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4/"
_DEFAULT_MAX_TOKENS = 8192


class GLMBackend(LLMBackend):
    """Use a Z.ai API key for scenario generation and repair.

    ``ZAI_API_KEY`` is required. ``ZAI_BASE_URL`` optionally overrides the
    standard API endpoint. GLM-5.3 is the default model and requires thinking,
    which is enabled automatically. Other models default to disabled thinking;
    ``ZAI_THINKING`` can explicitly select ``enabled`` or ``disabled``.
    """

    def __init__(
        self, model_id: str = DEFAULT_GLM_MODEL, *, timeout_seconds: float = 300,
    ) -> None:
        self._model_id = model_id.strip()
        if not self._model_id:
            raise ValueError("A GLM model ID is required")
        self._timeout_seconds = timeout_seconds

    def generate(
        self, prompt: str, temperature: float = 0.0, max_tokens: int | None = None,
    ) -> str:
        return self.generate_with_usage(prompt, temperature, max_tokens).text

    def generate_with_usage(
        self, prompt: str, temperature: float = 0.0, max_tokens: int | None = None,
    ) -> LLMResult:
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        api_key = os.environ.get("ZAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set ZAI_API_KEY in .env to use --backend glm")
        base_url = os.environ.get("ZAI_BASE_URL", "").strip() or _DEFAULT_BASE_URL
        requires_thinking = self._model_id.lower().startswith("glm-5.3")
        thinking = os.environ.get("ZAI_THINKING", "").strip().lower() or (
            "enabled" if requires_thinking else "disabled"
        )
        if thinking not in {"enabled", "disabled"}:
            raise ValueError("ZAI_THINKING must be enabled or disabled")
        if requires_thinking and thinking == "disabled":
            raise ValueError("GLM-5.3 requires ZAI_THINKING=enabled")

        from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

        try:
            with OpenAI(
                api_key=api_key, base_url=base_url,
                timeout=self._timeout_seconds, max_retries=2,
            ) as client:
                response = client.chat.completions.create(
                    model=self._model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens if max_tokens is not None else _DEFAULT_MAX_TOKENS,
                    extra_body={"thinking": {"type": thinking}},
                )
        except APITimeoutError:
            raise RuntimeError("Z.ai GLM request timed out after bounded retries") from None
        except APIConnectionError:
            raise RuntimeError("Could not connect to Z.ai; check ZAI_BASE_URL and network access") from None
        except APIStatusError as exc:
            # Do not echo provider response bodies or credentials into generation logs.
            raise RuntimeError(
                f"Z.ai GLM request failed (HTTP {exc.status_code}); "
                "check ZAI_API_KEY, model access, endpoint, and account quota"
            ) from None

        if not response.choices:
            raise RuntimeError("Z.ai GLM returned no response choices")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("Z.ai GLM response reached max_tokens before completion")
        if choice.finish_reason != "stop":
            raise RuntimeError(f"Z.ai GLM response did not complete: {choice.finish_reason}")
        result = choice.message.content
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("Z.ai GLM returned no final text")
        usage = response.usage
        return LLMResult(
            text=result,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
