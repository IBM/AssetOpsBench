"""Helpers for wrapping an agent ``run()`` call in an OTEL span.

Usage::

    from observability.runspan import agent_run_span

    async def run(self, question: str) -> AgentResult:
        with agent_run_span("plan-execute", model=self._model_id,
                            question=question) as span:
            result = await self._do_run(question)
            annotate_result(span, result)
            return result
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from .attributes import (
    AGENT_ANSWER_LENGTH,
    AGENT_QUESTION_LENGTH,
    AGENT_RUN_ID,
    AGENT_RUNNER,
    AGENT_SCENARIO_ID,
    AGENT_TOOL_CALLS,
    AGENT_TURNS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from .tracing import get_tracer

_run_id_var: ContextVar[str | None] = ContextVar("agent_run_id", default=None)
_scenario_id_var: ContextVar[str | None] = ContextVar("agent_scenario_id", default=None)


def set_run_context(
    *, run_id: str | None = None, scenario_id: str | None = None
) -> None:
    """Set the ambient run/scenario IDs read by the next :func:`agent_run_span`.

    CLIs and scenario harnesses call this once before invoking
    ``runner.run(...)``; the runner itself does not need to know about these
    identifiers.  Values persist for the current task / thread until the
    process exits or the callee overwrites them.
    """
    if run_id is not None:
        _run_id_var.set(run_id)
    if scenario_id is not None:
        _scenario_id_var.set(scenario_id)


def _system_from_model(model_id: str) -> str:
    """Best-effort provider family extraction from a model ID.

    Handles ``litellm_proxy/<family>/...`` and ``<family>/...`` patterns.
    Returns ``"unknown"`` when the shape is unrecognized.
    """
    mid = model_id
    if mid.startswith("litellm_proxy/"):
        mid = mid[len("litellm_proxy/"):]
    head, _, _ = mid.partition("/")
    # Common aliases → canonical family
    aliases = {
        "aws": "anthropic",
        "azure": "openai",
        "gcp": "anthropic",
        "vertex_ai": "anthropic",
        "bedrock": "anthropic",
    }
    return aliases.get(head.lower(), head.lower() or "unknown")


@contextmanager
def agent_run_span(
    runner_name: str,
    model: str,
    question: str,
    *,
    run_id: str | None = None,
    scenario_id: str | None = None,
) -> Iterator[Any]:
    """Start a root span for an agent ``run()`` call.

    Sets canonical attributes (``agent.runner``, ``gen_ai.system``,
    ``gen_ai.request.model``, ``agent.question.length``, plus ``agent.run_id``
    and ``agent.scenario_id`` when provided) and records exceptions on the
    span before re-raising.

    Args:
        runner_name: Runner identifier for the ``agent.runner`` attribute.
        model: Full model ID, used to derive ``gen_ai.system`` and stored as
               ``gen_ai.request.model``.
        question: Incoming question (only its length is stored).
        run_id: Unique identifier for this invocation; persisted saved traces
                can be joined back to an evaluation record via this value.
        scenario_id: Optional benchmark scenario identifier when the runner
                     is executing against a known fixture.

    Yields:
        The underlying OTEL span (or a no-op shim) so callers can annotate
        it further via :func:`annotate_result` or ad-hoc ``set_attribute``.
    """
    tracer = get_tracer()
    effective_run_id = run_id or _run_id_var.get()
    effective_scenario_id = scenario_id or _scenario_id_var.get()
    with tracer.start_as_current_span(f"agent.run {runner_name}") as span:
        span.set_attribute(AGENT_RUNNER, runner_name)
        span.set_attribute(GEN_AI_SYSTEM, _system_from_model(model))
        span.set_attribute(GEN_AI_REQUEST_MODEL, model)
        span.set_attribute(AGENT_QUESTION_LENGTH, len(question))
        if effective_run_id:
            span.set_attribute(AGENT_RUN_ID, effective_run_id)
        if effective_scenario_id:
            span.set_attribute(AGENT_SCENARIO_ID, effective_scenario_id)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            _set_error_status(span, str(exc))
            raise


def annotate_result(span, *, answer: str, trajectory: Any = None) -> None:
    """Attach answer/trajectory stats to an in-progress run span.

    ``trajectory`` is duck-typed — anything exposing
    ``total_input_tokens`` / ``total_output_tokens`` / ``turns`` /
    ``all_tool_calls`` works, as do the runner-specific Trajectory types.
    Missing attributes are skipped silently.
    """
    span.set_attribute(AGENT_ANSWER_LENGTH, len(answer or ""))

    if trajectory is None:
        return

    input_tokens = _safe_getattr_int(trajectory, "total_input_tokens")
    output_tokens = _safe_getattr_int(trajectory, "total_output_tokens")
    if input_tokens is not None:
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    if output_tokens is not None:
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

    turns = getattr(trajectory, "turns", None)
    if turns is not None:
        try:
            span.set_attribute(AGENT_TURNS, len(turns))
        except TypeError:
            pass

    tool_calls = getattr(trajectory, "all_tool_calls", None)
    if tool_calls is not None:
        try:
            span.set_attribute(AGENT_TOOL_CALLS, len(tool_calls))
        except TypeError:
            pass


def _safe_getattr_int(obj: Any, name: str) -> int | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _set_error_status(span, message: str) -> None:
    """Best-effort error status — avoids importing StatusCode unconditionally."""
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, message))
    except ImportError:
        # No-op span or OTEL not installed.
        pass
