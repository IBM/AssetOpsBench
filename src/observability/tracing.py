"""OpenTelemetry tracing setup for agent runners.

Tracing is enabled iff ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or its traces-specific
variant) is set and ``OTEL_SDK_DISABLED`` is not ``"true"``.  Otherwise
:func:`init_tracing` is a no-op and :func:`get_tracer` falls back to
OTel's built-in ``ProxyTracer`` (whose spans are non-recording), so
runner-side instrumentation code is safe to invoke unconditionally.

``BatchSpanProcessor`` buffers spans; an :func:`atexit` hook flushes the
provider on process exit so the final agent run's spans are not dropped.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading

from opentelemetry import trace

_log = logging.getLogger(__name__)

_initialized = False
_init_lock = threading.Lock()


def _tracing_enabled() -> bool:
    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return False
    return bool(
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )


def init_tracing(service_name: str) -> None:
    """Initialize the global OTEL tracer provider.

    Idempotent.  Silently does nothing when OTEL is disabled via env, so it
    is safe to call unconditionally from CLI entry points.
    """
    global _initialized
    if _initialized:
        return
    if not _tracing_enabled():
        return

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        _log.warning("OTEL SDK not installed; tracing disabled: %s", exc)
        return

    with _init_lock:
        if _initialized:
            return

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        # Flush buffered spans on exit so the last run's root span isn't lost.
        atexit.register(provider.shutdown)

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


def get_tracer(name: str = "agent"):
    """Return an OpenTelemetry :class:`Tracer`.

    When :func:`init_tracing` has not installed a provider, the returned
    tracer is OTel's built-in proxy whose spans are non-recording — callers
    can unconditionally ``start_as_current_span`` / ``set_attribute``.
    """
    return trace.get_tracer(name)
