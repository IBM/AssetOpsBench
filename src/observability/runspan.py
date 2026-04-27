"""Root-span helper for agent ``run()`` calls.

Callers set the ambient ``run_id`` / ``scenario_id`` once at the CLI boundary
via :func:`set_run_context`, then invoke :func:`agent_run_span` inside each
runner.  The span carries GenAI semconv attributes plus the contextvar-sourced
benchmark identifiers.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from opentelemetry.trace import Status, StatusCode

from .tracing import get_tracer

_run_id_var: ContextVar[str | None] = ContextVar("agent_run_id", default=None)
_scenario_id_var: ContextVar[str | None] = ContextVar("agent_scenario_id", default=None)


def set_run_context(
    *, run_id: str | None = None, scenario_id: str | None = None
) -> None:
    """Set the ambient run/scenario IDs read by the next :func:`agent_run_span`.

    CLIs call this once before invoking ``runner.run(...)``; the runner itself
    does not need to know about these identifiers.
    """
    if run_id is not None:
        _run_id_var.set(run_id)
    if scenario_id is not None:
        _scenario_id_var.set(scenario_id)


def _system_from_model(model_id: str) -> str:
    """Best-effort provider family from a model ID; returns ``"unknown"``
    when the shape isn't recognized."""
    mid = model_id.removeprefix("litellm_proxy/")
    head, _, _ = mid.partition("/")
    # Only aliases actually emitted by this repo are mapped.
    if head.lower() == "aws":
        return "anthropic"
    return head.lower() or "unknown"


@contextmanager
def agent_run_span(
    runner_name: str,
    model: str,
    question: str,
) -> Iterator[Any]:
    """Start a root span for an agent ``run()`` call.

    Sets canonical attributes (runner name, GenAI system/model, question
    length, and any ambient ``agent.run_id`` / ``agent.scenario_id`` from
    :func:`set_run_context`) and records exceptions on the span before
    re-raising.
    """
    tracer = get_tracer()
    run_id = _run_id_var.get()
    scenario_id = _scenario_id_var.get()
    with tracer.start_as_current_span(f"agent.run {runner_name}") as span:
        span.set_attribute("agent.runner", runner_name)
        span.set_attribute("gen_ai.system", _system_from_model(model))
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("agent.question.length", len(question))
        if run_id:
            span.set_attribute("agent.run_id", run_id)
        if scenario_id:
            span.set_attribute("agent.scenario_id", scenario_id)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


