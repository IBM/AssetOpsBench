"""Provider request translation and response handling."""

from __future__ import annotations

import sys
import types

import pytest

from llm import LiteLLMBackend, OpenAICompatBackend, make_backend


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


def test_unsupported_prefix_raises():
    with pytest.raises(ValueError, match="unsupported OpenAI-compatible model id"):
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
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert result.text == "hi"
    assert (result.input_tokens, result.output_tokens) == (3, 2)


@pytest.mark.parametrize("max_tokens", [None, 8192])
def test_litellm_token_budget_preserves_text_and_usage_apis(monkeypatch, max_tokens):
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="hi"))],
            usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    fake = types.ModuleType("litellm")
    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.test/v1")
    backend = LiteLLMBackend("litellm_proxy/test")

    assert backend.generate("hello", max_tokens=max_tokens) == "hi"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["max_tokens"] == (max_tokens or 2048)
    result = backend.generate_with_usage("hello", max_tokens=max_tokens)
    assert result.text == "hi"
    assert (result.input_tokens, result.output_tokens) == (3, 2)
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["max_tokens"] == (max_tokens or 2048)
