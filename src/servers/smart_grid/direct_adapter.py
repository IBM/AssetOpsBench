"""In-process direct-call adapter for the Smart Grid MCP servers.

Purpose
-------
Smart Grid Bench Experiment 1 measures the latency overhead of the MCP
JSON-RPC transport layer by running the same ReAct agent through three cells
that share the same tool set:

- **Cell A (Direct)** — ReAct calls the plain Python function in-process. Zero
  transport overhead. This module is that entry point.
- **Cell B (MCP baseline)** — ReAct speaks MCP JSON-RPC over stdio to the
  server processes. Transport cost = (B − A).
- **Cell C (MCP optimized)** — same as B with batching / connection reuse.

Using the same underlying functions across all three cells keeps the
comparison honest: any delta is transport, not algorithmic.

What this module does NOT do
----------------------------
- No ReAct loop. That's the Cell A runner's job.
- No serialization layer. Arguments go in as Python types, results come out
  as Python types.
- No schema validation. The MCP path already validates via :class:`FastMCP`.
- No logging. Latency instrumentation lives in the capture wrappers used by
  the runner so Cell A / B / C share it.

Layout
------
:func:`get_tools` — ordered list of :class:`ToolSpec` entries, one per tool.
:func:`get_tool` — ``ToolSpec`` lookup by qualified name like
    ``iot.get_sensor_readings``.
:func:`list_tool_specs_for_llm` — compact JSON-schema-ish list suitable for
    prompting an LLM (name, description, parameters).

The canonical tool set matches the four ``@mcp.tool()``-decorated functions
per server in :mod:`servers.smart_grid.{iot,fmsr,tsfm,wo}.main`.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    """A single tool in the direct-call registry."""

    name: str  # qualified name, e.g. "iot.get_sensor_readings"
    domain: str  # "iot" | "fmsr" | "tsfm" | "wo"
    bare_name: str  # "get_sensor_readings"
    fn: Callable[..., Any]  # the underlying Python function
    doc: str  # description extracted from the function's docstring

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def parameters(self) -> dict[str, dict[str, Any]]:
        """Return a minimal JSON-schema-ish parameter spec for LLM prompting."""
        sig = inspect.signature(self.fn)
        params: dict[str, dict[str, Any]] = {}
        for pname, p in sig.parameters.items():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            entry: dict[str, Any] = {}
            if p.annotation is not inspect.Parameter.empty:
                entry["type"] = _type_to_json_name(p.annotation)
            if p.default is not inspect.Parameter.empty:
                entry["default"] = _safe_json_value(p.default)
            else:
                entry["required"] = True
            params[pname] = entry
        return params


def _type_to_json_name(tp: Any) -> str:
    """Best-effort conversion from Python type hints to a JSON schema type."""
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _type_to_json_name(args[0])
        return "any"
    if tp is int:
        return "integer"
    if tp is float:
        return "number"
    if tp is bool:
        return "boolean"
    if tp is str:
        return "string"
    if tp in (list, dict):
        return tp.__name__
    if origin in (list, dict):
        return origin.__name__
    return "any"


def _safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _extract_doc(fn: Callable[..., Any]) -> str:
    doc = inspect.getdoc(fn) or ""
    # First paragraph only — keeps the prompt tight.
    for chunk in doc.split("\n\n"):
        chunk = chunk.strip()
        if chunk:
            return chunk
    return ""


def _build_registry() -> tuple[list[ToolSpec], dict[str, ToolSpec]]:
    """Import each Smart Grid server module and collect its
    ``@mcp.tool()``-decorated functions into the ToolSpec registry.

    FastMCP's ``@mcp.tool()`` decorator preserves the underlying Python
    function — module-level names remain callable in-process — so we just
    import them and call them directly. The ``mcp`` package itself still
    needs to be importable (for ``FastMCP()`` at module load); callers that
    run outside the serving venv should skip this module.
    """
    from servers.smart_grid.fmsr import main as fmsr
    from servers.smart_grid.iot import main as iot
    from servers.smart_grid.tsfm import main as tsfm
    from servers.smart_grid.wo import main as wo

    # Canonical tool set per domain, in the same order as the server files.
    # Keep this list in sync when servers add/remove tools.
    catalog: list[tuple[str, str, Callable[..., Any]]] = [
        ("iot", "list_assets", iot.list_assets),
        ("iot", "get_asset_metadata", iot.get_asset_metadata),
        ("iot", "list_sensors", iot.list_sensors),
        ("iot", "get_sensor_readings", iot.get_sensor_readings),
        ("fmsr", "list_failure_modes", fmsr.list_failure_modes),
        ("fmsr", "search_failure_modes", fmsr.search_failure_modes),
        ("fmsr", "get_sensor_correlation", fmsr.get_sensor_correlation),
        ("fmsr", "get_dga_record", fmsr.get_dga_record),
        ("fmsr", "analyze_dga", fmsr.analyze_dga),
        ("tsfm", "get_rul", tsfm.get_rul),
        ("tsfm", "forecast_rul", tsfm.forecast_rul),
        ("tsfm", "detect_anomalies", tsfm.detect_anomalies),
        ("tsfm", "trend_analysis", tsfm.trend_analysis),
        ("wo", "list_fault_records", wo.list_fault_records),
        ("wo", "get_fault_record", wo.get_fault_record),
        ("wo", "create_work_order", wo.create_work_order),
        ("wo", "list_work_orders", wo.list_work_orders),
        ("wo", "update_work_order", wo.update_work_order),
        ("wo", "estimate_downtime", wo.estimate_downtime),
    ]

    specs: list[ToolSpec] = []
    for domain, bare, fn in catalog:
        if not callable(fn):
            raise RuntimeError(
                f"Expected {domain}.{bare} to be callable after FastMCP "
                f"decoration; got {type(fn).__name__}. FastMCP may have "
                f"changed its decorator contract; adjust _build_registry."
            )
        specs.append(
            ToolSpec(
                name=f"{domain}.{bare}",
                domain=domain,
                bare_name=bare,
                fn=fn,
                doc=_extract_doc(fn),
            )
        )
    index = {s.name: s for s in specs}
    return specs, index


# Lazily initialized so that ``import servers.smart_grid.direct_adapter`` is
# cheap for callers that only want the types. Build on first use.
_TOOLS: list[ToolSpec] | None = None
_TOOLS_BY_NAME: dict[str, ToolSpec] | None = None


def _ensure_registry() -> None:
    global _TOOLS, _TOOLS_BY_NAME
    if _TOOLS is None:
        _TOOLS, _TOOLS_BY_NAME = _build_registry()


def get_tools() -> list[ToolSpec]:
    """Return the full ordered list of ToolSpec entries."""
    _ensure_registry()
    assert _TOOLS is not None
    return list(_TOOLS)


def get_tool(name: str) -> ToolSpec:
    """Lookup a ToolSpec by qualified name, e.g. ``iot.get_sensor_readings``."""
    _ensure_registry()
    assert _TOOLS_BY_NAME is not None
    if name not in _TOOLS_BY_NAME:
        raise KeyError(f"unknown tool {name!r}; available: {sorted(_TOOLS_BY_NAME)}")
    return _TOOLS_BY_NAME[name]


def list_tool_specs_for_llm() -> list[dict[str, Any]]:
    """Return a compact, JSON-serializable list of tool descriptors suitable
    for prompting an LLM. Intentionally minimal — a ReAct runner can enrich
    it further with few-shot examples if needed.
    """
    return [
        {
            "name": s.name,
            "description": s.doc,
            "parameters": s.parameters(),
        }
        for s in get_tools()
    ]
