"""Unit tests for :mod:`observability.file_exporter`."""

from __future__ import annotations

import json
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from observability import agent_run_span, set_run_context
from observability import runspan as _runspan
from observability.file_exporter import OTLPJsonFileExporter


def _reset_provider():
    """Swap in a fresh global TracerProvider for the next test."""
    trace._TRACER_PROVIDER_SET_ONCE = type(trace._TRACER_PROVIDER_SET_ONCE)()  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]


def _install_file_exporter(path: Path) -> TracerProvider:
    _reset_provider()
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(OTLPJsonFileExporter(path)))
    trace.set_tracer_provider(provider)
    return provider


def test_file_exporter_writes_jsonl(tmp_path: Path):
    out = tmp_path / "traces.jsonl"
    _install_file_exporter(out)

    set_run_context(run_id="run-1", scenario_id="scn-A")
    with agent_run_span("deep-agent", model="anthropic/claude", question="q"):
        pass

    content = out.read_text(encoding="utf-8")
    lines = [line for line in content.splitlines() if line]
    assert len(lines) == 1, f"expected 1 line, got {len(lines)}: {content!r}"

    payload = json.loads(lines[0])
    assert "resourceSpans" in payload
    (resource_span,) = payload["resourceSpans"]
    (scope_span,) = resource_span["scopeSpans"]
    (span,) = scope_span["spans"]
    assert span["name"] == "agent.run deep-agent"

    attrs = {a["key"]: a["value"] for a in span["attributes"]}
    assert attrs["agent.runner"]["stringValue"] == "deep-agent"
    assert attrs["agent.run_id"]["stringValue"] == "run-1"
    assert attrs["agent.scenario_id"]["stringValue"] == "scn-A"


def test_file_exporter_creates_parent_dir(tmp_path: Path):
    out = tmp_path / "nested" / "subdir" / "traces.jsonl"
    assert not out.parent.exists()
    OTLPJsonFileExporter(out)
    assert out.parent.is_dir()


def test_file_exporter_appends(tmp_path: Path):
    out = tmp_path / "traces.jsonl"
    _install_file_exporter(out)

    for i in range(3):
        with agent_run_span("deep-agent", model="anthropic/claude", question=f"q{i}"):
            pass

    lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3
    for line in lines:
        json.loads(line)  # each line must be valid JSON
