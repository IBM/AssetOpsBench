"""OpenTelemetry tracing setup for agent runners.

Public surface:
  - ``init_tracing(service_name)`` — one-shot setup of the global tracer
    provider + OTLP/HTTP exporter + httpx auto-instrumentation.
  - ``get_tracer()`` — returns a :class:`Tracer` (no-op when OTEL is not
    installed or tracing has been disabled).

Tracing is enabled iff:
  1. The ``opentelemetry`` packages are importable, AND
  2. ``OTEL_SDK_DISABLED`` is not set to ``"true"``, AND
  3. ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or the tracing-specific variant) is set.

When any precondition fails the module falls back to OpenTelemetry's built-in
no-op tracer, so runners can unconditionally call ``get_tracer()`` /
``start_as_current_span(...)`` without guarding.

HTTPX instrumentation is what propagates the ``traceparent`` header to the
LiteLLM proxy so its spans nest under the agent trace — all four runners
ultimately reach the proxy via ``httpx`` (LiteLLM, OpenAI SDK, LangChain
ChatOpenAI), so this single instrumentor covers them.
"""

from __future__ import annotations

import logging
import os
import threading

_log = logging.getLogger(__name__)

_NOOP_TRACER_NAME = "agent"

_initialized = False
_init_lock = threading.Lock()


def _tracing_enabled() -> bool:
    """Return True when OTEL env is configured and the SDK isn't disabled."""
    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return False
    return bool(
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )


def init_tracing(service_name: str) -> None:
    """Initialize the global OTEL tracer provider.

    Idempotent — subsequent calls are no-ops.  Silently does nothing when
    OTEL is disabled via environment (see module docstring for the rules),
    so it is safe to call unconditionally from CLI entry points.

    Args:
        service_name: Value for the ``service.name`` resource attribute
                      (e.g. ``"plan-execute"``, ``"deep-agent"``).
    """
    global _initialized
    if _initialized:
        return
    if not _tracing_enabled():
        _log.debug("OTEL tracing disabled (env not configured).")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        _log.warning("OTEL packages not installed; tracing disabled: %s", exc)
        return

    with _init_lock:
        if _initialized:
            return

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)

        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().instrument()
        except ImportError:
            _log.warning(
                "opentelemetry-instrumentation-httpx not installed — LiteLLM "
                "proxy calls will not be traced from the client side."
            )

        _initialized = True
        _log.info("OTEL tracing initialized (service=%s).", service_name)


def get_tracer(name: str = _NOOP_TRACER_NAME):
    """Return an OpenTelemetry :class:`Tracer`.

    When OpenTelemetry isn't installed, returns a lightweight shim exposing
    ``start_as_current_span`` as a no-op context manager so callers don't
    need to guard their instrumentation code.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        return _NoopTracer()

    return trace.get_tracer(name)


class _NoopSpan:
    """No-op span for environments without OpenTelemetry installed."""

    def set_attribute(self, key, value) -> None:  # noqa: D401
        return None

    def set_status(self, *args, **kwargs) -> None:
        return None

    def record_exception(self, *args, **kwargs) -> None:
        return None

    def add_event(self, *args, **kwargs) -> None:
        return None


class _NoopTracer:
    """Minimal tracer shim used when ``opentelemetry`` is not installed."""

    def start_as_current_span(self, name, **kwargs):
        class _NoopCm:
            def __enter__(self_inner):
                return _NoopSpan()

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _NoopCm()


def _reset_for_tests() -> None:
    """Reset the module's singleton state — test-only helper."""
    global _initialized
    _initialized = False
