"""Tests for the shared router registry in ``llm.routers``."""

from __future__ import annotations

import pytest

from llm.routers import (
    LITELLM_PREFIX,
    TOKENROUTER_PREFIX,
    is_openai_compat,
    resolve_model,
    resolve_router_creds,
    router_prefix,
)


def test_prefix_constants():
    assert LITELLM_PREFIX == "litellm_proxy/"
    assert TOKENROUTER_PREFIX == "tokenrouter/"


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("litellm_proxy/aws/claude-opus-4-6", "aws/claude-opus-4-6"),
        ("tokenrouter/MiniMax-M3", "MiniMax-M3"),
        ("anthropic/claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"),
        ("gpt-4o", "gpt-4o"),
        ("", ""),
    ],
)
def test_resolve_model(model_id, expected):
    assert resolve_model(model_id) == expected


@pytest.mark.parametrize(
    "model_id,expected_prefix",
    [
        ("litellm_proxy/aws/claude-opus-4-6", "litellm_proxy/"),
        ("tokenrouter/MiniMax-M3", "tokenrouter/"),
        ("anthropic/claude-sonnet-4-6", None),
    ],
)
def test_router_prefix(model_id, expected_prefix):
    assert router_prefix(model_id) == expected_prefix


def test_is_openai_compat():
    assert is_openai_compat("tokenrouter/MiniMax-M3")
    assert not is_openai_compat("litellm_proxy/aws/claude-opus-4-6")
    assert not is_openai_compat("watsonx/meta-llama/x")


def test_resolve_router_creds_tokenrouter(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-key")
    creds = resolve_router_creds("tokenrouter/MiniMax-M3")
    assert creds.prefix == "tokenrouter/"
    assert creds.base_url == "https://api.tokenrouter.com/v1"
    assert creds.api_key == "tr-key"


def test_resolve_router_creds_native_passthrough():
    assert resolve_router_creds("anthropic/claude-sonnet-4-6") is None


def test_resolve_router_creds_strict_raises(monkeypatch):
    monkeypatch.delenv("TOKENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("TOKENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        resolve_router_creds("tokenrouter/MiniMax-M3")


def test_resolve_router_creds_lenient_returns_none(monkeypatch):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    assert resolve_router_creds("litellm_proxy/aws/x", strict=False) is None
