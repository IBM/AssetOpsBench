"""File-backed OTLP-JSON span exporter.

Writes one ``ExportTraceServiceRequest`` JSON object per line to the given
path, matching the on-disk format produced by the official OpenTelemetry
Collector ``file`` exporter.  Output is replayable into any OTLP backend
(Jaeger, Tempo, Honeycomb, Grafana Cloud, …) via the Collector's
``otlpjsonfile`` receiver, without requiring a Collector to *produce* it.

The default install contains no file-exporter primitive, so this module
composes one from the OTLP common encoder plus a tiny append-only writer.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

_log = logging.getLogger(__name__)


class OTLPJsonFileExporter(SpanExporter):
    """Append each span batch to *path* as one OTLP-JSON line.

    Thread-safe via a write lock so concurrent `export()` calls from the
    `BatchSpanProcessor` don't interleave bytes in the file.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        try:
            line = _encode_otlp_json(spans)
        except Exception:
            _log.exception("OTLPJsonFileExporter: encoding failed; dropping batch")
            return SpanExportResult.FAILURE
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            _log.exception("OTLPJsonFileExporter: write failed")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """No-op: file is opened per-export and closed immediately."""


def _encode_otlp_json(spans: Sequence[ReadableSpan]) -> str:
    """Encode *spans* as a single-line OTLP-JSON ``TracesData`` message.

    Isolated for testability; also keeps the `_internal` import contained to
    one call site so a future SDK refactor only touches this function.
    """
    # ``encode_spans`` produces a protobuf ``ExportTraceServiceRequest``; the
    # common encoder path is the same one used by ``OTLPSpanExporter`` itself,
    # so the wire format stays in sync with upstream.
    from google.protobuf.json_format import MessageToJson
    from opentelemetry.exporter.otlp.proto.common._internal.trace_encoder import (
        encode_spans,
    )

    request = encode_spans(spans)
    # OTLP-JSON wire format uses camelCase field names (spec default), which
    # is what ``MessageToJson`` emits out of the box.
    return MessageToJson(request, indent=None)
