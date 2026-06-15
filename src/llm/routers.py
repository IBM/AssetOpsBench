"""Single source of truth for proxy/router model-string prefixes.

A *router* is a proxied endpoint selected by a model-string prefix.  This
module owns the prefix → env-var mapping and the resolve helpers so both
sides of the codebase agree on it:

* ``llm.OpenAICompatBackend`` — the native-``openai`` backend, and
* the agent SDK runners (claude / openai / deep / stirrup), which point their
  third-party SDK clients at the resolved endpoint.

Prefixes::

    litellm_proxy/<model>   LiteLLM proxy  (LITELLM_BASE_URL / LITELLM_API_KEY)
    tokenrouter/<model>     TokenRouter    (TOKENROUTER_BASE_URL / TOKENROUTER_API_KEY)
"""

from __future__ import annotations

import os
from typing import NamedTuple

LITELLM_PREFIX = "litellm_proxy/"
TOKENROUTER_PREFIX = "tokenrouter/"


class RouterCreds(NamedTuple):
    """Resolved endpoint + key for a proxy router."""

    prefix: str
    base_url: str
    api_key: str


# prefix -> (base_url_env_var, api_key_env_var)
PROXY_ROUTERS: dict[str, tuple[str, str]] = {
    LITELLM_PREFIX: ("LITELLM_BASE_URL", "LITELLM_API_KEY"),
    TOKENROUTER_PREFIX: ("TOKENROUTER_BASE_URL", "TOKENROUTER_API_KEY"),
}

# Prefixes whose endpoints speak the OpenAI Chat Completions API and can be
# driven by the native ``openai`` SDK (llm.OpenAICompatBackend).
OPENAI_COMPAT_PREFIXES: tuple[str, ...] = (TOKENROUTER_PREFIX,)


def router_prefix(model_id: str) -> str | None:
    """Return the proxy-router prefix matching *model_id*, else ``None``."""
    for prefix in PROXY_ROUTERS:
        if model_id.startswith(prefix):
            return prefix
    return None


def resolve_model(model_id: str) -> str:
    """Strip any known router prefix from *model_id* if present.

    ``"tokenrouter/MiniMax-M3"`` -> ``"MiniMax-M3"``;
    ``"anthropic/claude-sonnet-4-6"`` -> unchanged.
    """
    prefix = router_prefix(model_id)
    return model_id[len(prefix):] if prefix else model_id


def is_openai_compat(model_id: str) -> bool:
    """True if *model_id* targets an OpenAI-compatible router (e.g. TokenRouter)."""
    return model_id.startswith(OPENAI_COMPAT_PREFIXES)


def resolve_router_creds(
    model_id: str, *, strict: bool = True
) -> RouterCreds | None:
    """Resolve endpoint + key for *model_id*, or ``None`` if not proxied.

    Args:
        strict: when ``True`` (default) a matched prefix with missing env
            vars raises :class:`ValueError`; when ``False`` it returns
            ``None`` so the caller can fall back to its own defaults.
    """
    prefix = router_prefix(model_id)
    if prefix is None:
        return None
    base_env, key_env = PROXY_ROUTERS[prefix]
    base_url = os.environ.get(base_env)
    api_key = os.environ.get(key_env)
    if not base_url or not api_key:
        if strict:
            raise ValueError(
                f"{base_env} and {key_env} must be set "
                f"when using the {prefix!r} model prefix"
            )
        return None
    return RouterCreds(prefix=prefix, base_url=base_url, api_key=api_key)
