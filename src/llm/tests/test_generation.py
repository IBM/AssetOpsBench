"""Unit tests for GenerationParams, from_env, resolve_params, reasoning_supported."""

from __future__ import annotations

import pytest

from llm.generation import (
    GenerationParams,
    EFFORT_TO_OPENAI,
    from_env,
    reasoning_supported,
    resolve_params,
)


# ── reasoning_supported heuristics ───────────────────────────────────────────


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("watsonx/meta-llama/llama-3-3-70b-instruct", False),
        ("watsonx/ibm/granite-3-3-8b-instruct", False),
        ("litellm_proxy/aws/claude-opus-4-6", True),
        ("litellm_proxy/GCP/claude-4-sonnet", True),
        ("litellm_proxy/azure/gpt-5.4", True),
        ("anthropic/claude-sonnet-4-5", True),
        ("openai/o1-preview", True),
        ("openai/o3-mini", True),
        ("openai/gpt-5", True),
        ("some-unknown-model", False),
        ("ollama/llama3", False),
    ],
)
def test_reasoning_supported(model_id, expected):
    assert reasoning_supported(model_id) is expected


# ── effort vocab mapping ──────────────────────────────────────────────────────


def test_effort_to_openai_map_complete():
    for effort in ("none", "low", "medium", "high", "max"):
        assert effort in EFFORT_TO_OPENAI


def test_max_maps_to_xhigh():
    assert EFFORT_TO_OPENAI["max"] == "xhigh"


def test_none_maps_to_none():
    assert EFFORT_TO_OPENAI["none"] == "none"


# ── from_env ─────────────────────────────────────────────────────────────────


def test_from_env_defaults(monkeypatch):
    for var in (
        "LLM_MAX_TOKENS",
        "LLM_TEMPERATURE",
        "LLM_TOP_P",
        "LLM_REASONING_EFFORT",
        "LLM_THINKING_BUDGET_TOKENS",
        "LLM_STOP",
    ):
        monkeypatch.delenv(var, raising=False)
    p = from_env()
    assert p.max_tokens == 4096
    assert p.temperature == 0.0
    assert p.reasoning_effort == "none"
    assert p.top_p is None
    assert p.thinking_budget_tokens is None
    assert p.stop is None


def test_from_env_max_tokens(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
    assert from_env().max_tokens == 8192


def test_from_env_temperature(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    assert from_env().temperature == pytest.approx(0.7)


def test_from_env_top_p(monkeypatch):
    monkeypatch.setenv("LLM_TOP_P", "0.9")
    assert from_env().top_p == pytest.approx(0.9)


def test_from_env_reasoning_effort_valid(monkeypatch):
    for effort in ("none", "low", "medium", "high", "max"):
        monkeypatch.setenv("LLM_REASONING_EFFORT", effort)
        assert from_env().reasoning_effort == effort


def test_from_env_reasoning_effort_invalid_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "ultra")
    import logging

    with caplog.at_level(logging.WARNING, logger="llm.generation"):
        p = from_env()
    assert p.reasoning_effort == "none"
    assert "LLM_REASONING_EFFORT" in caplog.text


def test_from_env_thinking_budget_tokens(monkeypatch):
    monkeypatch.setenv("LLM_THINKING_BUDGET_TOKENS", "2048")
    assert from_env().thinking_budget_tokens == 2048


def test_from_env_stop_comma_separated(monkeypatch):
    monkeypatch.setenv("LLM_STOP", "</answer>,END, STOP ")
    p = from_env()
    assert p.stop == ("</answer>", "END", "STOP")


def test_from_env_stop_empty_skipped(monkeypatch):
    monkeypatch.setenv("LLM_STOP", "  ,  ")
    assert from_env().stop is None


# ── resolve_params ────────────────────────────────────────────────────────────


def test_resolve_params_no_override():
    base = GenerationParams(max_tokens=512, temperature=0.5)
    result = resolve_params(base)
    assert result is base


def test_resolve_params_temperature_always_wins():
    base = GenerationParams(temperature=0.5)
    result = resolve_params(base, temperature=0.9)
    assert result.temperature == pytest.approx(0.9)


def test_resolve_params_temperature_zero_wins():
    base = GenerationParams(temperature=0.8)
    result = resolve_params(base, temperature=0.0)
    assert result.temperature == pytest.approx(0.0)


def test_resolve_params_override_non_none_replaces():
    base = GenerationParams(max_tokens=1024)
    override = GenerationParams(max_tokens=8192)
    result = resolve_params(base, override=override)
    assert result.max_tokens == 8192


def test_resolve_params_none_field_does_not_replace():
    base = GenerationParams(top_p=0.95)
    override = GenerationParams()  # top_p=None by default
    result = resolve_params(base, override=override)
    assert result.top_p == pytest.approx(0.95)


def test_resolve_params_frozen():
    base = GenerationParams()
    result = resolve_params(base, temperature=0.5)
    assert result is not base
    with pytest.raises((AttributeError, TypeError)):
        result.temperature = 0.1  # type: ignore[misc]
