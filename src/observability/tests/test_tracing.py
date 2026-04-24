"""Unit tests for ``observability.tracing`` and ``observability.runspan``.

Uses OTEL's :class:`InMemorySpanExporter` so tests run fully offline.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from observability import agent_run_span, get_tracer, init_tracing, set_run_context
from observability import tracing as _tracing
from observability import runspan as _runspan
from observability.runspan import _system_from_model


@pytest.fixture
def memory_exporter(monkeypatch):
    """Install a fresh InMemorySpanExporter as the global tracer provider."""
    monkeypatch.setattr(_tracing, "_initialized", False)
    # Reset OTel's one-shot guard so set_tracer_provider actually installs.
    trace._TRACER_PROVIDER_SET_ONCE = type(trace._TRACER_PROVIDER_SET_ONCE)()  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)
    yield exporter
    exporter.clear()


def test_init_tracing_noop_without_env(monkeypatch):
    monkeypatch.setattr(_tracing, "_initialized", False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    init_tracing("test-service")
    assert _tracing._initialized is False


def test_init_tracing_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(_tracing, "_initialized", False)
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    init_tracing("test-service")
    assert _tracing._initialized is False


def test_init_tracing_enables_with_file_only(monkeypatch, tmp_path):
    """OTEL_TRACES_FILE alone is enough to activate tracing."""
    monkeypatch.setattr(_tracing, "_initialized", False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("OTEL_TRACES_FILE", str(tmp_path / "traces.jsonl"))
    trace._TRACER_PROVIDER_SET_ONCE = type(trace._TRACER_PROVIDER_SET_ONCE)()  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]

    init_tracing("test-service")
    assert _tracing._initialized is True


def test_get_tracer_returns_tracer():
    """get_tracer() always returns a usable tracer; no-op spans work too."""
    tracer = get_tracer()
    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("k", "v")  # must not raise on non-recording span


def test_agent_run_span_emits_attributes(memory_exporter):
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


def test_agent_run_span_records_exception(memory_exporter):
    with pytest.raises(RuntimeError, match="boom"):
        with agent_run_span("claude-agent", model="aws/claude", question="q"):
            raise RuntimeError("boom")

    s = memory_exporter.get_finished_spans()[0]
    assert s.status.status_code.name == "ERROR"
    assert any(e.name == "exception" for e in s.events)


def test_set_run_context_seeds_span(memory_exporter):
    set_run_context(run_id="ctx-run", scenario_id="scn-9")
    with agent_run_span("claude-agent", model="anthropic/claude", question="q"):
        pass

    s = memory_exporter.get_finished_spans()[0]
    assert s.attributes["agent.run_id"] == "ctx-run"
    assert s.attributes["agent.scenario_id"] == "scn-9"


def test_run_ids_absent_by_default(memory_exporter):
    with agent_run_span("openai-agent", model="openai/gpt-5", question="q"):
        pass

    s = memory_exporter.get_finished_spans()[0]
    assert "agent.run_id" not in s.attributes
    assert "agent.scenario_id" not in s.attributes


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("litellm_proxy/aws/claude-opus-4-6", "anthropic"),
        ("litellm_proxy/azure/gpt-5.4", "azure"),
        ("watsonx/meta-llama/llama-3-3-70b-instruct", "watsonx"),
        ("anthropic/claude-sonnet-4-6", "anthropic"),
        ("openai/gpt-5", "openai"),
        ("", "unknown"),
    ],
)
def test_system_from_model(model_id, expected):
    assert _system_from_model(model_id) == expected
