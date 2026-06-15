"""Unit tests for to_litellm_kwargs — kwargs emitted per model × effort matrix."""

from __future__ import annotations

import logging

import pytest

from llm.generation import GenerationParams
from llm.litellm import to_litellm_kwargs


# ── helpers ───────────────────────────────────────────────────────────────────

_CLAUDE_MODEL = "litellm_proxy/aws/claude-opus-4-6"
_WATSONX_MODEL = "watsonx/meta-llama/llama-3-3-70b-instruct"
_GPT_MODEL = "litellm_proxy/azure/gpt-5.4"
_UNKNOWN_MODEL = "some-custom-model"


# ── basic fields always present ───────────────────────────────────────────────


def test_max_tokens_present():
    p = GenerationParams(max_tokens=8192)
    kw = to_litellm_kwargs(_CLAUDE_MODEL, p)
    assert kw["max_tokens"] == 8192


def test_temperature_present():
    p = GenerationParams(temperature=0.7)
    kw = to_litellm_kwargs(_CLAUDE_MODEL, p)
    assert kw["temperature"] == pytest.approx(0.7)


def test_top_p_omitted_when_none():
    p = GenerationParams()
    kw = to_litellm_kwargs(_CLAUDE_MODEL, p)
    assert "top_p" not in kw


def test_top_p_included_when_set():
    p = GenerationParams(top_p=0.9)
    kw = to_litellm_kwargs(_CLAUDE_MODEL, p)
    assert kw["top_p"] == pytest.approx(0.9)


def test_stop_omitted_when_none():
    p = GenerationParams()
    kw = to_litellm_kwargs(_CLAUDE_MODEL, p)
    assert "stop" not in kw


def test_stop_included_when_set():
    p = GenerationParams(stop=("END", "</answer>"))
    kw = to_litellm_kwargs(_CLAUDE_MODEL, p)
    assert kw["stop"] == ["END", "</answer>"]


# ── thinking: supported model ─────────────────────────────────────────────────


def test_thinking_disabled_explicit_when_effort_none_on_claude():
    """effort=none on a Claude model → emit thinking=disabled to be deterministic."""
    p = GenerationParams(reasoning_effort="none")
    kw = to_litellm_kwargs(_CLAUDE_MODEL, p)
    assert kw.get("thinking") == {"type": "disabled"}
    assert "reasoning_effort" not in kw


def test_reasoning_effort_forwarded_on_claude():
    p = GenerationParams(reasoning_effort="medium")
    kw = to_litellm_kwargs(_CLAUDE_MODEL, p)
    assert kw["reasoning_effort"] == "medium"


def test_thinking_budget_tokens_sets_thinking_dict():
    p = GenerationParams(reasoning_effort="medium", thinking_budget_tokens=2048)
    kw = to_litellm_kwargs(_CLAUDE_MODEL, p)
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_effort_high_on_claude():
    p = GenerationParams(reasoning_effort="high")
    kw = to_litellm_kwargs(_CLAUDE_MODEL, p)
    assert kw["reasoning_effort"] == "high"
    assert "thinking" not in kw or kw.get("thinking") != {"type": "disabled"}


def test_effort_max_on_claude():
    p = GenerationParams(reasoning_effort="max")
    kw = to_litellm_kwargs(_CLAUDE_MODEL, p)
    assert kw["reasoning_effort"] == "max"


# ── thinking: unsupported model (WatsonX) ────────────────────────────────────


def test_no_thinking_kwargs_on_watsonx_when_none():
    p = GenerationParams(reasoning_effort="none")
    kw = to_litellm_kwargs(_WATSONX_MODEL, p)
    assert "thinking" not in kw
    assert "reasoning_effort" not in kw


def test_reasoning_stripped_with_warning_on_watsonx(caplog):
    p = GenerationParams(reasoning_effort="medium")
    with caplog.at_level(logging.WARNING, logger="llm.litellm"):
        kw = to_litellm_kwargs(_WATSONX_MODEL, p)
    assert "reasoning_effort" not in kw
    assert "thinking" not in kw
    assert "stripping" in caplog.text.lower()


# ── thinking: unknown model ───────────────────────────────────────────────────


def test_reasoning_stripped_with_warning_on_unknown_model(caplog):
    p = GenerationParams(reasoning_effort="low")
    with caplog.at_level(logging.WARNING, logger="llm.litellm"):
        kw = to_litellm_kwargs(_UNKNOWN_MODEL, p)
    assert "reasoning_effort" not in kw
    assert "stripping" in caplog.text.lower()


# ── watsonx: max_tokens still included ───────────────────────────────────────


def test_max_tokens_on_watsonx():
    p = GenerationParams(max_tokens=4096)
    kw = to_litellm_kwargs(_WATSONX_MODEL, p)
    assert kw["max_tokens"] == 4096


def test_temperature_on_watsonx():
    p = GenerationParams(temperature=0.3)
    kw = to_litellm_kwargs(_WATSONX_MODEL, p)
    assert kw["temperature"] == pytest.approx(0.3)


# ── proxy GPT ─────────────────────────────────────────────────────────────────


def test_reasoning_effort_forwarded_on_gpt():
    p = GenerationParams(reasoning_effort="high")
    kw = to_litellm_kwargs(_GPT_MODEL, p)
    assert kw["reasoning_effort"] == "high"
