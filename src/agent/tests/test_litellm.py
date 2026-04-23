"""Tests for the shared LiteLLM-proxy helpers in ``agent._litellm``."""

from __future__ import annotations

import pytest

from agent._litellm import LITELLM_PREFIX, resolve_model


def test_prefix_constant():
    assert LITELLM_PREFIX == "litellm_proxy/"


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("litellm_proxy/aws/claude-opus-4-6", "aws/claude-opus-4-6"),
        ("litellm_proxy/azure/gpt-5.4", "azure/gpt-5.4"),
        ("litellm_proxy/Azure/gpt-5-2025-08-07", "Azure/gpt-5-2025-08-07"),
        ("anthropic:claude-sonnet-4-6", "anthropic:claude-sonnet-4-6"),
        ("claude-opus-4-6", "claude-opus-4-6"),
        ("gpt-4o", "gpt-4o"),
        ("", ""),
    ],
)
def test_resolve_model(model_id, expected):
    assert resolve_model(model_id) == expected
