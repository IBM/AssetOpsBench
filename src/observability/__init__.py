"""OpenTelemetry-based observability for agent runners.

Exports ``init_tracing`` and ``get_tracer``.  See
:mod:`observability.tracing` for details.
"""

from .runspan import agent_run_span, annotate_result, set_run_context
from .tracing import get_tracer, init_tracing

__all__ = [
    "agent_run_span",
    "annotate_result",
    "get_tracer",
    "init_tracing",
    "set_run_context",
]
