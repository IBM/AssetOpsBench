"""Shared LiteLLM-proxy helpers used by the SDK-driven runners.

The three SDK runners (:class:`~agent.claude_agent.ClaudeAgentRunner`,
:class:`~agent.openai_agent.OpenAIAgentRunner`,
:class:`~agent.deep_agent.DeepAgentRunner`) each accept model strings of
two shapes:

* ``litellm_proxy/<provider>/<model>`` — routed through the LiteLLM proxy;
  the prefix is stripped before handing the ID to the underlying SDK.
* ``<provider>/<model>`` — passed through as-is to the SDK's native
  provider integration.

This module centralises the prefix + stripping logic so the runners and
their tests share a single source of truth.
"""

from __future__ import annotations

LITELLM_PREFIX = "litellm_proxy/"


def resolve_model(model_id: str) -> str:
    """Strip the ``litellm_proxy/`` prefix from a model ID if present.

    Examples::

        "litellm_proxy/aws/claude-opus-4-6"  ->  "aws/claude-opus-4-6"
        "litellm_proxy/azure/gpt-5.4"        ->  "azure/gpt-5.4"
        "anthropic/claude-sonnet-4-6"        ->  "anthropic/claude-sonnet-4-6"
    """
    if model_id.startswith(LITELLM_PREFIX):
        return model_id[len(LITELLM_PREFIX):]
    return model_id
