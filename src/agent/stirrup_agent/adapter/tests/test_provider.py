"""The framework binding, exercised against a fake provider.

Stirrup is not installable here (PyPI is blocked by policy), so these tests
drive the same lifecycle contract the real provider satisfies: an async context
manager whose ``__aenter__`` returns a list of tools.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from agent.stirrup_agent.adapter.provider import build_mcp_provider_class
from agent.stirrup_agent.adapter.spec import AdapterSpec


@dataclass
class FakeServerTool:
    name: str
    description: str = "server description"
    schema: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def input_schema(self) -> dict:
        return self.schema

    def set_input_schema(self, schema: dict) -> None:
        self.schema = schema

    async def run(self, arguments: dict):
        self.calls.append(dict(arguments))
        return {"ok": True, "echo": dict(arguments)}


class FakeProvider:
    """Mimics an MCP provider: tools arrive from __aenter__."""

    def __init__(self, tools):
        self._tools = tools

    async def __aenter__(self):
        return self._tools

    async def __aexit__(self, *exc):
        return None


def enter(cls, tools):
    async def go():
        async with cls(tools) as returned:
            return returned

    return asyncio.run(go())


def wo() -> FakeServerTool:
    return FakeServerTool(
        name="wo__query",
        schema={
            "type": "object",
            "properties": {"siteid": {"type": "string"}, "cursor": {"type": "string"}},
            "required": ["siteid"],
        },
    )


def test_an_empty_spec_returns_the_original_class_untouched() -> None:
    """The baseline arm must run the original code path, not a no-op wrapper."""
    cls = build_mcp_provider_class(FakeProvider, AdapterSpec.empty())

    assert cls is FakeProvider


def test_a_nonempty_spec_returns_a_subclass() -> None:
    spec = AdapterSpec.from_raw({"tools": {"wo__query": {"strip_empty": True}}})

    cls = build_mcp_provider_class(FakeProvider, spec)

    assert cls is not FakeProvider
    assert issubclass(cls, FakeProvider)


def test_the_adapted_description_and_schema_reach_the_agent() -> None:
    spec = AdapterSpec.from_raw(
        {"tools": {"wo__query": {"rename": {"site": "siteid"}, "description": "Use site."}}}
    )
    cls = build_mcp_provider_class(FakeProvider, spec)

    tools = enter(cls, [wo()])

    assert tools[0].description == "Use site."
    assert "site" in tools[0].input_schema()["properties"]


def test_arguments_are_translated_before_the_server_sees_them() -> None:
    spec = AdapterSpec.from_raw(
        {"tools": {"wo__query": {"rename": {"site": "siteid"}, "strip_empty": True}}}
    )
    tool = wo()
    cls = build_mcp_provider_class(FakeProvider, spec)

    adapted = enter(cls, [tool])[0]
    asyncio.run(adapted.run({"site": "MAIN", "cursor": None}))

    assert tool.calls == [{"siteid": "MAIN"}]


def test_a_hidden_tool_never_reaches_the_agent() -> None:
    spec = AdapterSpec.from_raw({"hidden": ["wo__query"]})
    cls = build_mcp_provider_class(FakeProvider, spec)

    assert enter(cls, [wo()]) == []


def test_an_adapter_naming_a_tool_no_server_offers_fails_loudly() -> None:
    """A silent no-op would pass the batch gate while changing nothing."""
    spec = AdapterSpec.from_raw({"tools": {"wo__typo": {"strip_empty": True}}})
    cls = build_mcp_provider_class(FakeProvider, spec)

    with pytest.raises(ValueError, match="no server offers"):
        enter(cls, [wo()])


def test_a_composite_runs_its_steps_in_order_and_returns_the_last() -> None:
    first = FakeServerTool(name="wo__query")
    second = FakeServerTool(name="iot__history")
    spec = AdapterSpec.from_raw(
        {
            "composites": [
                {
                    "name": "wo__with_history",
                    "description": "Work order plus its sensor history.",
                    "steps": [
                        {"tool": "wo__query", "arguments": {"siteid": "$input.site"}},
                        {"tool": "iot__history", "arguments": {"asset": "$0.echo.siteid"}},
                    ],
                }
            ]
        }
    )
    cls = build_mcp_provider_class(FakeProvider, spec)

    tools = enter(cls, [first, second])
    result = asyncio.run(tools[-1].run({"site": "MAIN"}))

    assert first.calls == [{"siteid": "MAIN"}]
    assert second.calls == [{"asset": "MAIN"}]
    assert result["echo"] == {"asset": "MAIN"}


def test_a_composite_appears_alongside_the_server_tools() -> None:
    spec = AdapterSpec.from_raw(
        {
            "composites": [
                {"name": "c", "description": "d", "steps": [{"tool": "wo__query"}]}
            ]
        }
    )
    cls = build_mcp_provider_class(FakeProvider, spec)

    names = [t.name for t in enter(cls, [wo()])]

    assert names == ["wo__query", "c"]
