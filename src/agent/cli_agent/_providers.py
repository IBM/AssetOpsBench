"""Provider routing for the CLI coding-agent runners.

Generalises the ``litellm_proxy/`` prefix convention from
:mod:`agent._litellm` to the proxies the benchmark routes through — LiteLLM,
OpenRouter, and TokenRouter — plus a ``direct`` fallback that uses each CLI's
native provider auth.

A model id may carry a provider prefix::

    litellm_proxy/azure/gpt-5.4   -> litellm,     model "azure/gpt-5.4"
    openrouter/openai/gpt-5.4     -> openrouter,  model "openai/gpt-5.4"
    tokenrouter/gpt-5             -> tokenrouter, model "gpt-5"
    gpt-5                         -> direct,      model "gpt-5"

All three proxies are OpenAI-compatible at ``/v1`` (OpenRouter defaults to
``https://openrouter.ai/api/v1``; LiteLLM and TokenRouter are self-hosted, so
their base URL comes from an env var). Credentials are read at run time, so a
model id can be parsed without the environment being populated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .._litellm import LITELLM_PREFIX  # "litellm_proxy/"


@dataclass(frozen=True)
class _ProviderSpec:
    """Static routing facts for one provider."""

    name: str
    prefix: str
    base_url_env: str | None
    default_base_url: str | None
    api_key_env: str


# Known proxy providers, matched by model-id prefix.
_SPECS: tuple[_ProviderSpec, ...] = (
    _ProviderSpec(
        "litellm", LITELLM_PREFIX, "LITELLM_BASE_URL", None, "LITELLM_API_KEY"
    ),
    _ProviderSpec(
        "openrouter",
        "openrouter/",
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
    ),
    _ProviderSpec(
        "tokenrouter",
        "tokenrouter/",
        "TOKENROUTER_BASE_URL",
        None,
        "TOKENROUTER_API_KEY",
    ),
)

# Fallback: no recognised prefix -> the CLI's own native provider auth.
_DIRECT = _ProviderSpec("direct", "", None, None, "OPENAI_API_KEY")


@dataclass(frozen=True)
class ResolvedProvider:
    """A provider with its base URL + credentials resolved from the env."""

    name: str
    model: str
    base_url: str
    api_key: str
    api_key_env: str


def _match(model_id: str) -> tuple[_ProviderSpec, str]:
    for spec in _SPECS:
        if model_id.startswith(spec.prefix):
            return spec, model_id[len(spec.prefix):]
    return _DIRECT, model_id


def provider_name(model_id: str) -> str:
    """Provider a model id routes to (``"litellm"`` / ``"openrouter"`` / ...)."""
    return _match(model_id)[0].name


def resolve_model(model_id: str) -> str:
    """Strip the provider prefix, returning the wire model id."""
    return _match(model_id)[1]


def resolve_provider(model_id: str) -> ResolvedProvider:
    """Resolve *model_id* to a provider with base URL + key read from the env.

    Raises ``ValueError`` if a non-``direct`` provider is missing its base URL
    or API key env var.
    """
    spec, model = _match(model_id)

    base_url = ""
    if spec.base_url_env:
        base_url = os.environ.get(spec.base_url_env) or (spec.default_base_url or "")
    elif spec.default_base_url:
        base_url = spec.default_base_url
    api_key = os.environ.get(spec.api_key_env, "")

    if spec.name != "direct":
        missing = []
        if spec.base_url_env and not base_url:
            missing.append(spec.base_url_env)
        if not api_key:
            missing.append(spec.api_key_env)
        if missing:
            raise ValueError(
                f"{' and '.join(missing)} must be set when using the "
                f"{spec.prefix!r} ({spec.name}) provider prefix"
            )

    return ResolvedProvider(
        name=spec.name,
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=spec.api_key_env,
    )
