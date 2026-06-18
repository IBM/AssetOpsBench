"""Tests for backend selection and the OpenAI-compatible (TokenRouter) path."""

from __future__ import annotations

import sys
import types

import pytest

from llm import LiteLLMBackend, OpenAICompatBackend, is_openai_compat, make_backend


def _install_fake_openai(monkeypatch, captured: dict):
    """Install a stub ``openai`` module that records call kwargs."""

    def create(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content="hi"))
            ],
            usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    class OpenAI:
        def __init__(self, base_url=None, api_key=None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )

    fake = types.ModuleType("openai")
    fake.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)


def test_is_openai_compat():
    assert is_openai_compat("tokenrouter/MiniMax-M3")
    assert not is_openai_compat("litellm_proxy/aws/claude-opus-4-6")
    assert not is_openai_compat("watsonx/meta-llama/llama-3-3-70b-instruct")


def test_make_backend_dispatch():
    assert isinstance(make_backend("tokenrouter/MiniMax-M3"), OpenAICompatBackend)
    assert isinstance(make_backend("litellm_proxy/aws/claude-opus-4-6"), LiteLLMBackend)
    assert isinstance(make_backend("watsonx/meta-llama/x"), LiteLLMBackend)


def test_unsupported_prefix_raises():
    with pytest.raises(ValueError):
        OpenAICompatBackend("gpt-4o")


def test_tokenrouter_strips_prefix_and_routes(monkeypatch):
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured)
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-key")

    result = make_backend("tokenrouter/MiniMax-M3").generate_with_usage("hello")

    assert captured["model"] == "MiniMax-M3"  # bare name, prefix stripped
    assert captured["base_url"] == "https://api.tokenrouter.com/v1"
    assert captured["api_key"] == "tr-key"
    assert result.text == "hi"
    assert (result.input_tokens, result.output_tokens) == (3, 2)


def test_model_id_property_keeps_full_string():
    assert (
        OpenAICompatBackend("tokenrouter/MiniMax-M3").model_id
        == "tokenrouter/MiniMax-M3"
    )
