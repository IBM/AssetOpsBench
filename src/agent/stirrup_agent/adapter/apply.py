"""Apply an :class:`AdapterSpec` to a tool list.

Framework-agnostic on purpose. The functions here take and return objects that
satisfy a small structural protocol, so they run under test with plain stubs and
under the runner with Stirrup's ``Tool``. The framework-bound glue is
``provider.py`` and it is deliberately tiny.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from .spec import AdapterSpec, Composite, ToolAdaptation


class ToolLike(Protocol):
    """The part of a tool this module needs."""

    name: str
    description: str

    def input_schema(self) -> dict: ...


@dataclass
class AdaptedTool:
    """A tool as the agent sees it, plus how to call the server underneath."""

    name: str
    description: str
    schema: dict
    #: maps the agent's arguments onto the server's
    translate: Callable[[dict], dict]
    #: text to surface before the call, or None
    hook: str | None = None
    #: the underlying server tool name; differs only for composites
    target: str | None = None
    composite: Composite | None = None

    @property
    def is_composite(self) -> bool:
        return self.composite is not None


def adapt_tool(tool: ToolLike, adaptation: ToolAdaptation) -> AdaptedTool:
    """Present one server tool through the adapter."""
    return AdaptedTool(
        name=tool.name,
        description=adaptation.description or tool.description,
        schema=adaptation.agent_schema(_schema_of(tool)),
        translate=adaptation.server_arguments,
        hook=adaptation.hook,
        target=tool.name,
    )


def composite_tool(composite: Composite) -> AdaptedTool:
    """Present a derived tool. It calls existing servers and adds no endpoint."""
    return AdaptedTool(
        name=composite.name,
        description=composite.description,
        schema=composite.input_schema or {"type": "object", "properties": {}},
        translate=lambda args: dict(args),
        target=None,
        composite=composite,
    )


def adapt_tools(tools: Iterable[ToolLike], spec: AdapterSpec) -> list[AdaptedTool]:
    """The full agent-facing tool list.

    Hidden tools are removed, remaining tools are adapted, and composites are
    appended. An empty spec returns the tools unchanged in name, description and
    schema, which is the property the baseline arm depends on.
    """
    out: list[AdaptedTool] = []
    seen: set[str] = set()
    for tool in tools:
        if tool.name in spec.hidden:
            continue
        out.append(adapt_tool(tool, spec.for_tool(tool.name)))
        seen.add(tool.name)

    for composite in spec.composites:
        if composite.name in seen:
            raise ValueError(
                f"composite {composite.name!r} collides with a server tool; "
                "rename it or hide the server tool"
            )
        out.append(composite_tool(composite))
        seen.add(composite.name)
    return out


def unknown_tool_names(tools: Iterable[ToolLike], spec: AdapterSpec) -> list[str]:
    """Names the spec adapts or hides that no server actually serves.

    An evolved patch that renames arguments on a tool which does not exist is a
    silent no-op, and a silent no-op that still passes the batch gate poisons
    the search. The loop calls this and rejects the patch.
    """
    available = {t.name for t in tools}
    referenced = set(spec.tools) | set(spec.hidden)
    for composite in spec.composites:
        referenced |= {step["tool"] for step in composite.steps}
    return sorted(referenced - available - {c.name for c in spec.composites})


def _schema_of(tool: ToolLike) -> dict:
    schema = tool.input_schema()
    if not isinstance(schema, dict):
        raise TypeError(f"tool {tool.name!r} returned a non-dict schema")
    return schema


def resolve_step_arguments(
    template: dict, inputs: dict, results: list[Any]
) -> dict:
    """Substitute ``$input.x`` and ``$<n>.x`` references in a composite step."""
    return {k: _resolve(v, inputs, results) for k, v in template.items()}


def _resolve(value: Any, inputs: dict, results: list[Any]) -> Any:
    if not isinstance(value, str) or not value.startswith("$"):
        return value
    source, _, path = value[1:].partition(".")
    if source == "input":
        if path not in inputs:
            raise KeyError(f"composite references unknown input {path!r}")
        return inputs[path]
    try:
        index = int(source)
    except ValueError as exc:
        raise KeyError(f"composite reference {value!r} is not $input or $<n>") from exc
    if not 0 <= index < len(results):
        raise KeyError(f"composite references step {index}, which has not run")
    return _dig(results[index], path) if path else results[index]


def _dig(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj[part]
        elif isinstance(obj, (list, tuple)):
            obj = obj[int(part)]
        else:
            raise KeyError(f"cannot read {part!r} from {type(obj).__name__}")
    return obj
