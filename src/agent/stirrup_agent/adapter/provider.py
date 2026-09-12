"""The only framework-bound part of the adapter.

Everything that decides *what* changes lives in ``spec.py`` and ``apply.py``,
which are stdlib-only and fully tested without Stirrup, a container, or a live
server. This module is the thin binding that puts those decisions in front of
the agent, and it is kept small on purpose: the less logic here, the less of the
adapter escapes the test suite.

The seam is the one the repository already uses twice elsewhere. A provider
subclass returns the tool list from ``__aenter__``; the skill mount and the
workspace-preserving wrapper both hook the same lifecycle.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .spec import AdapterSpec

_log = logging.getLogger(__name__)


def build_mcp_provider_class(base_cls, spec: AdapterSpec):
    """Wrap an MCP provider class so its tools reach the agent adapted.

    Returns ``base_cls`` unchanged when the spec is empty, so the baseline arm
    runs the original code path rather than an adapter that happens to be a
    no-op. That is the same guarantee ``k0`` gives for skills, and it is what
    lets a reported gain be attributed to the adapter rather than the plumbing.
    """
    if spec.is_empty:
        return base_cls

    from .apply import adapt_tools, unknown_tool_names

    class _AdaptedMCPToolProvider(base_cls):
        async def __aenter__(self):
            tools = await super().__aenter__()
            missing = unknown_tool_names(tools, spec)
            if missing:
                raise ValueError(
                    f"adapter refers to tools no server offers: {missing}. "
                    "A patch against a nonexistent tool is a silent no-op, "
                    "which would pass the batch gate while changing nothing."
                )
            adapted = adapt_tools(tools, spec)
            _log.info(
                "adapter: %d server tools -> %d agent tools (%d composites)",
                len(tools),
                len(adapted),
                sum(1 for a in adapted if a.is_composite),
            )
            return [_bind(a, tools) for a in adapted]

    return _AdaptedMCPToolProvider


def _bind(adapted, server_tools):
    """Attach the adapted presentation to the underlying callable.

    Stirrup's ``Tool`` is a pydantic-described callable, so binding happens by
    rewriting the description and schema and wrapping the call. Kept in one
    place so a Stirrup API change touches one function.
    """
    by_name = {t.name: t for t in server_tools}

    if adapted.is_composite:
        return _CompositeTool(adapted, by_name)

    tool = by_name[adapted.target]
    tool.description = adapted.description
    if hasattr(tool, "set_input_schema"):
        tool.set_input_schema(adapted.schema)
    original = tool.run if hasattr(tool, "run") else None

    if original is not None:
        async def _run(arguments: dict, _original=original, _a=adapted):
            return await _original(_a.translate(dict(arguments)))

        tool.run = _run
    return tool


class _CompositeTool:
    """A derived tool: several server calls behind one name.

    Adds no endpoint and changes no server, which is what keeps New Tool
    Addition inside the client-side rule while remaining the highest-accepting
    patch type in AutoSaddler's Figure 3c.
    """

    def __init__(self, adapted, by_name: dict):
        self.name = adapted.name
        self.description = adapted.description
        self._schema = adapted.schema
        self._composite = adapted.composite
        self._by_name = by_name

    def input_schema(self) -> dict:
        return self._schema

    async def run(self, arguments: dict):
        from .apply import resolve_step_arguments

        results: list = []
        for step in self._composite.steps:
            target = self._by_name[step["tool"]]
            step_args = resolve_step_arguments(
                step.get("arguments", {}), dict(arguments), results
            )
            results.append(await target.run(step_args))
        return results[-1]


def load_spec(adapter_dir: Path | str | None) -> AdapterSpec:
    """Resolve ``--adapter-dir`` to a spec. Absent directory -> empty adapter."""
    return AdapterSpec.load(adapter_dir)
