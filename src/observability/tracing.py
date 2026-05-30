"""OpenTelemetry tracing setup for agent runners.

Two independent sinks are supported; either or both can be enabled:

* ``OTEL_TRACES_FILE`` — path to append OTLP-JSON lines to.  Writes happen
  in-process; no Docker or Collector required.  The output format is
  identical to the OpenTelemetry Collector's ``file`` exporter, so it can
  be replayed into any OTLP backend later.
* ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``)
  — ship spans over HTTP to a live collector (Jaeger, Tempo, Honeycomb, …).

When neither is set, :func:`init_tracing` is a no-op and :func:`get_tracer`
returns OTel's built-in proxy tracer (non-recording spans), so runner-side
instrumentation code is safe to invoke unconditionally.

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


def _traces_file_path() -> str | None:
    return os.environ.get("OTEL_TRACES_FILE") or None


def _http_endpoint_set() -> bool:
    return bool(
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )


def _tracing_enabled() -> bool:
    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return False
    return bool(_traces_file_path()) or _http_endpoint_set()


def init_tracing(service_name: str) -> None:
    """Initialize the global OTEL tracer provider.

    Idempotent.  No-op when tracing isn't configured, so callers can invoke
    unconditionally from CLI entry points.
    """
    global _initialized
    if _initialized:
        return
    if not _tracing_enabled():
        return

    try:
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

        if (path := _traces_file_path()) is not None:
            from .file_exporter import OTLPJsonFileExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPJsonFileExporter(path)))
            _log.info("OTEL file exporter enabled (path=%s).", path)

        if _http_endpoint_set():
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
            except ImportError:
                _log.warning(
                    "opentelemetry-exporter-otlp-proto-http not installed; "
                    "HTTP export disabled."
                )
            else:
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
                _log.info("OTEL HTTP exporter enabled.")

        trace.set_tracer_provider(provider)
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
