"""Tests for backend selection and the OpenAI-compatible (TokenRouter) path."""

from __future__ import annotations

import sys
import types

import pytest

from llm import (
    EmptyCompletionError,
    LiteLLMBackend,
    OpenAICompatBackend,
    is_openai_compat,
    make_backend,
)
from llm.base import MAX_TOKENS_ENV, resolve_max_tokens


def _install_fake_openai(
    monkeypatch,
    captured: dict,
    content: str | None = "hi",
    finish_reason: str = "stop",
    completion_tokens: int = 2,
):
    """Install a stub ``openai`` module that records call kwargs."""

    def create(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=types.SimpleNamespace(
                prompt_tokens=3, completion_tokens=completion_tokens
            ),
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


def test_max_tokens_defaults_to_2048(monkeypatch):
    monkeypatch.delenv("AOB_LLM_MAX_TOKENS", raising=False)
    assert resolve_max_tokens() == 2048


def test_max_tokens_env_override(monkeypatch):
    monkeypatch.setenv("AOB_LLM_MAX_TOKENS", "8192")
    assert resolve_max_tokens() == 8192


@pytest.mark.parametrize("bad", ["nope", "0", "-1"])
def test_max_tokens_rejects_invalid(monkeypatch, bad):
    monkeypatch.setenv("AOB_LLM_MAX_TOKENS", bad)
    with pytest.raises(ValueError, match="AOB_LLM_MAX_TOKENS"):
        resolve_max_tokens()


def test_max_tokens_override_reaches_the_request(monkeypatch):
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured)
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-key")
    monkeypatch.setenv("AOB_LLM_MAX_TOKENS", "4096")

    make_backend("tokenrouter/MiniMax-M3").generate_with_usage("hello")

    assert captured["max_tokens"] == 4096


@pytest.mark.parametrize("content", [None, "", "   \n "])
def test_empty_completion_raises_at_the_backend(monkeypatch, content):
    """A budget-exhausted completion must fail here, not four frames later."""
    captured: dict = {}
    _install_fake_openai(
        monkeypatch,
        captured,
        content=content,
        finish_reason="length",
        completion_tokens=2048,
    )
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-key")
    monkeypatch.delenv("AOB_LLM_MAX_TOKENS", raising=False)

    with pytest.raises(EmptyCompletionError) as excinfo:
        make_backend("tokenrouter/MiniMax-M3").generate_with_usage("hello")

    err = excinfo.value
    assert err.model == "tokenrouter/MiniMax-M3"
    assert err.finish_reason == "length"
    assert (err.completion_tokens, err.max_tokens) == (2048, 2048)
    assert err.truncated
    assert f"Raise {MAX_TOKENS_ENV}" in str(err)


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_truncation_is_recognised_across_providers(monkeypatch, finish_reason):
    """Anthropic-family providers report ``max_tokens`` rather than ``length``."""
    captured: dict = {}
    _install_fake_openai(
        monkeypatch, captured, content=None, finish_reason=finish_reason
    )
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-key")

    with pytest.raises(EmptyCompletionError) as excinfo:
        make_backend("tokenrouter/MiniMax-M3").generate_with_usage("hello")

    assert excinfo.value.truncated


def test_untruncated_empty_completion_does_not_blame_the_budget(monkeypatch):
    """Misdirection is the defect being fixed — do not blame the cap wrongly."""
    captured: dict = {}
    _install_fake_openai(
        monkeypatch, captured, content=None, finish_reason="stop", completion_tokens=0
    )
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-key")

    with pytest.raises(EmptyCompletionError) as excinfo:
        make_backend("tokenrouter/MiniMax-M3").generate_with_usage("hello")

    err = excinfo.value
    assert not err.truncated
    assert "unlikely to help" in str(err)


def test_generate_surfaces_empty_completion(monkeypatch):
    captured: dict = {}
    _install_fake_openai(monkeypatch, captured, content=None, finish_reason="length")
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-key")

    with pytest.raises(EmptyCompletionError):
        make_backend("tokenrouter/MiniMax-M3").generate("hello")


def _install_fake_litellm(monkeypatch, captured: dict, content: str | None):
    def completion(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content),
                    finish_reason="length" if content is None else "stop",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=256, completion_tokens=2048),
        )

    fake = types.ModuleType("litellm")
    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)


def test_litellm_empty_completion_raises(monkeypatch):
    """The recorded failure: 256-token prompt, 2048-token completion, no content."""
    captured: dict = {}
    _install_fake_litellm(monkeypatch, captured, content=None)
    monkeypatch.setenv("LITELLM_API_KEY", "key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://proxy.example/v1")
    monkeypatch.delenv("AOB_LLM_MAX_TOKENS", raising=False)

    with pytest.raises(EmptyCompletionError) as excinfo:
        LiteLLMBackend("litellm_proxy/aws/claude-opus-4-6").generate_with_usage("hi")

    assert excinfo.value.finish_reason == "length"
    assert captured["max_tokens"] == 2048


def test_litellm_returns_usage_on_success(monkeypatch):
    captured: dict = {}
    _install_fake_litellm(monkeypatch, captured, content="a plan")
    monkeypatch.setenv("LITELLM_API_KEY", "key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("AOB_LLM_MAX_TOKENS", "16000")

    result = LiteLLMBackend("litellm_proxy/aws/claude-opus-4-6").generate_with_usage(
        "hi"
    )

    assert result.text == "a plan"
    assert (result.input_tokens, result.output_tokens) == (256, 2048)
    assert captured["max_tokens"] == 16000
