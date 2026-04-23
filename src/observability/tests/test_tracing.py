"""Unit tests for ``observability.tracing`` and ``observability.runspan``.

Uses OTEL's :class:`InMemorySpanExporter` so tests run fully offline and
don't require a real collector.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from observability import agent_run_span, annotate_result, get_tracer, init_tracing
from observability import tracing as _tracing
from observability.runspan import _system_from_model


@pytest.fixture
def memory_exporter(monkeypatch):
    """Install a fresh InMemorySpanExporter as the global tracer provider.

    OTel protects ``set_tracer_provider`` with a one-shot guard; in tests we
    need to reset that guard between runs to install a fresh provider.  The
    private attributes used here are stable in OTel SDK >= 1.20.
    """
    _tracing._reset_for_tests()
    # Reset OTel's one-shot guard so set_tracer_provider actually installs.
    trace._TRACER_PROVIDER_SET_ONCE = type(trace._TRACER_PROVIDER_SET_ONCE)()  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    _tracing._reset_for_tests()


def test_init_tracing_noop_without_env(monkeypatch):
    """Without OTEL_EXPORTER_OTLP_ENDPOINT, init_tracing is a no-op."""
    _tracing._reset_for_tests()
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    init_tracing("test-service")
    assert _tracing._initialized is False


def test_init_tracing_skips_when_disabled(monkeypatch):
    """OTEL_SDK_DISABLED=true takes precedence even if endpoint set."""
    _tracing._reset_for_tests()
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    init_tracing("test-service")
    assert _tracing._initialized is False


def test_get_tracer_returns_tracer():
    """get_tracer() always returns a usable tracer-like object."""
    tracer = get_tracer()
    with tracer.start_as_current_span("test-span"):
        pass  # Should not raise regardless of init state.


def test_agent_run_span_emits_attributes(memory_exporter):
    """agent_run_span sets canonical attributes and ends span normally."""
    with agent_run_span(
        "plan-execute",
        model="litellm_proxy/aws/claude-opus-4-6",
        question="What sensors are on Chiller 6?",
    ) as span:
        span.set_attribute("custom.flag", True)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "agent.run plan-execute"
    assert s.attributes["agent.runner"] == "plan-execute"
    assert s.attributes["gen_ai.system"] == "anthropic"
    assert s.attributes["gen_ai.request.model"] == "litellm_proxy/aws/claude-opus-4-6"
    assert s.attributes["agent.question.length"] == len("What sensors are on Chiller 6?")
    assert s.attributes["custom.flag"] is True
    # OTEL default status is UNSET (not OK) — error status explicitly set below.
    assert s.status.status_code.name in ("UNSET", "OK")


def test_agent_run_span_records_exception(memory_exporter):
    """Exceptions propagate and are recorded on the span with ERROR status."""
    with pytest.raises(RuntimeError, match="boom"):
        with agent_run_span("claude-agent", model="aws/claude", question="q"):
            raise RuntimeError("boom")

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.status.status_code.name == "ERROR"
    assert any(e.name == "exception" for e in s.events)


def test_annotate_result_attaches_totals(memory_exporter):
    """annotate_result writes trajectory totals onto the span."""
    class _Traj:
        total_input_tokens = 1234
        total_output_tokens = 56
        turns = [object(), object()]
        all_tool_calls = [object()]

    with agent_run_span("deep-agent", model="aws/claude", question="q") as span:
        annotate_result(span, answer="final answer", trajectory=_Traj())

    spans = memory_exporter.get_finished_spans()
    s = spans[0]
    assert s.attributes["gen_ai.usage.input_tokens"] == 1234
    assert s.attributes["gen_ai.usage.output_tokens"] == 56
    assert s.attributes["agent.turns"] == 2
    assert s.attributes["agent.tool_calls"] == 1
    assert s.attributes["agent.answer.length"] == len("final answer")


def test_annotate_result_without_trajectory(memory_exporter):
    """annotate_result handles trajectory=None gracefully."""
    with agent_run_span("openai-agent", model="openai/gpt-5", question="q") as span:
        annotate_result(span, answer="ok", trajectory=None)

    s = memory_exporter.get_finished_spans()[0]
    assert s.attributes["agent.answer.length"] == 2
    assert "gen_ai.usage.input_tokens" not in s.attributes


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("litellm_proxy/aws/claude-opus-4-6", "anthropic"),
        ("litellm_proxy/azure/gpt-5.4", "openai"),
        ("watsonx/meta-llama/llama-3-3-70b-instruct", "watsonx"),
        ("anthropic/claude-sonnet-4-6", "anthropic"),
        ("openai/gpt-5", "openai"),
        ("", "unknown"),
    ],
)
def test_system_from_model(model_id, expected):
    assert _system_from_model(model_id) == expected
