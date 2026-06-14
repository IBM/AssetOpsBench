"""OpenAI-compatible LLM backend (no litellm dependency).

For gateways that expose the standard OpenAI Chat Completions API — such as
`TokenRouter <https://www.tokenrouter.com>`_ — we talk to them with the
``openai`` SDK directly instead of routing through litellm.  litellm only
earns its keep for providers that are *not* OpenAI-shaped (e.g. watsonx); an
OpenAI-compatible router needs no such abstraction.

The provider is encoded in the model-string prefix and the bare model name is
sent to the endpoint, matching ordinary OpenAI SDK usage::

    tokenrouter/MiniMax-M3   →  POST {TOKENROUTER_BASE_URL}/chat/completions
                                with model="MiniMax-M3"

Credentials are read from the router's env vars (see ``_OPENAI_COMPAT_ROUTERS``).
"""

from __future__ import annotations

import os

from .base import LLMBackend, LLMResult

# prefix -> (base_url_env_var, api_key_env_var)
_OPENAI_COMPAT_ROUTERS: dict[str, tuple[str, str]] = {
    "tokenrouter/": ("TOKENROUTER_BASE_URL", "TOKENROUTER_API_KEY"),
}


def is_openai_compat(model_id: str) -> bool:
    """Return ``True`` if *model_id* targets a known OpenAI-compatible router."""
    return any(model_id.startswith(p) for p in _OPENAI_COMPAT_ROUTERS)


class OpenAICompatBackend(LLMBackend):
    """LLM backend using the native ``openai`` SDK against a compatible router.

    Args:
        model_id: prefixed model string, e.g. ``"tokenrouter/MiniMax-M3"``.
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        for prefix, (base_env, key_env) in _OPENAI_COMPAT_ROUTERS.items():
            if model_id.startswith(prefix):
                self._base_env = base_env
                self._key_env = key_env
                self._model_name = model_id[len(prefix):]
                break
        else:
            raise ValueError(
                f"unsupported OpenAI-compatible model id: {model_id!r} "
                f"(expected one of prefixes: {sorted(_OPENAI_COMPAT_ROUTERS)})"
            )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self.generate_with_usage(prompt, temperature).text

    def generate_with_usage(
        self, prompt: str, temperature: float = 0.0
    ) -> LLMResult:
        from openai import OpenAI

        client = OpenAI(
            base_url=os.environ[self._base_env],
            api_key=os.environ[self._key_env],
        )
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
