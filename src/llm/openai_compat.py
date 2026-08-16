"""OpenAI-compatible LLM backend (no litellm dependency).

For gateways that expose the standard OpenAI Chat Completions API — such as
`TokenRouter <https://www.tokenrouter.com>`_ — we talk to them with the
``openai`` SDK directly instead of routing through litellm.  litellm only
earns its keep for providers that are *not* OpenAI-shaped (e.g. watsonx).

The prefix→endpoint mapping lives in :mod:`llm.routers` (shared with the
agent runners).  The bare model name is sent to the endpoint::

    tokenrouter/MiniMax-M3   →  POST {TOKENROUTER_BASE_URL}/chat/completions
                                with model="MiniMax-M3"
"""

from __future__ import annotations

from .base import LLMBackend, LLMResult
from .routers import is_openai_compat, resolve_model, resolve_router_creds

__all__ = ["OpenAICompatBackend", "is_openai_compat"]


class OpenAICompatBackend(LLMBackend):
    """LLM backend using the native ``openai`` SDK against a compatible router.

    Args:
        model_id: prefixed model string, e.g. ``"tokenrouter/MiniMax-M3"``.
    """

    def __init__(self, model_id: str) -> None:
        if not is_openai_compat(model_id):
            raise ValueError(f"unsupported OpenAI-compatible model id: {model_id!r}")
        self._model_id = model_id
        self._model_name = resolve_model(model_id)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self.generate_with_usage(prompt, temperature).text

    def generate_with_usage(self, prompt: str, temperature: float = 0.0) -> LLMResult:
        from openai import OpenAI

        creds = resolve_router_creds(self._model_id)  # strict: clear error if unset
        client = OpenAI(base_url=creds.base_url, api_key=creds.api_key)
        response = client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
        )
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=response.choices[0].message.content,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
