"""Tests for the MCP gateway topology.

The retrieval and indexing half runs with no Stirrup import at all, which is
part of the point: BM25 ranking is deterministic and offline, so a gateway sweep
is replayable from the repository without a model or a network call. The
provider half needs ``stirrup.core.models`` for ``Tool`` and ``ToolResult`` and
skips when it is absent.
"""

from __future__ import annotations

import json

import pytest

from agent.stirrup_agent.gateway import (
    GATEWAY_CALL_TOOL,
    GATEWAY_DESCRIBE_TOOL,
    GATEWAY_SEARCH_TOOL,
    CallToolParams,
    DescribeToolsParams,
    SearchToolsParams,
    ToolCard,
    ToolIndex,
    _summarize,
    _tokenize,
)
from agent.stirrup_agent.trajectory import classify_tool

DOMAIN_SERVERS = {"iot", "fmsr", "tsfm", "wo", "vibration", "utilities"}


# -- tokenization ----------------------------------------------------------


def test_tokenizer_splits_snake_case_tool_names():
    # Tool names carry most of the retrieval signal in this catalogue, so
    # get_workorder_costs has to match a query saying "work order cost".
    assert _tokenize("wo__get_workorder_costs") == [
        "wo",
        "get",
        "workorder",
        "costs",
    ]


def test_tokenizer_splits_camel_case():
    assert _tokenize("computeFFTSpectrum") == ["compute", "fft", "spectrum"]


def test_summarize_takes_the_first_sentence():
    assert _summarize("List work orders. Supports filters.") == "List work orders."


def test_summarize_handles_missing_descriptions():
    assert _summarize("") == "(no description)"


# -- ranking ---------------------------------------------------------------


def _index() -> ToolIndex:
    return ToolIndex(
        [
            ToolCard(
                name="wo__list_workorders",
                server="wo",
                summary="List work orders.",
                description="List work orders with optional status and site filters.",
            ),
            ToolCard(
                name="wo__get_workorder_costs",
                server="wo",
                summary="Costs for one work order.",
                description="Return planned and actual costs for a single work order.",
            ),
            ToolCard(
                name="iot__history",
                server="iot",
                summary="Sensor history.",
                description="Fetch raw sensor readings over a time range for an asset.",
            ),
            ToolCard(
                name="tsfm__list_models",
                server="tsfm",
                summary="List forecasting models.",
                description="List available time series forecasting models.",
            ),
        ]
    )


def test_search_ranks_the_obvious_tool_first():
    hits = _index().search("what are the costs on this work order", k=3)
    assert hits[0][0].name == "wo__get_workorder_costs"


def test_search_respects_k():
    assert len(_index().search("work order", k=1)) == 1


def test_search_returns_nothing_for_an_unmatched_query():
    # An empty result must be visible to the agent as a failure rather than as
    # "there are no such tools", which is why the provider marks it success=False.
    assert _index().search("photosynthesis chlorophyll", k=3) == []


def test_ranking_is_deterministic_across_calls():
    index = _index()
    first = [c.name for c, _ in index.search("sensor readings for an asset", k=4)]
    second = [c.name for c, _ in index.search("sensor readings for an asset", k=4)]
    assert first == second


def test_manifest_groups_by_server_and_lists_every_tool():
    manifest = _index().manifest()
    assert "wo:" in manifest and "iot:" in manifest and "tsfm:" in manifest
    for name in (
        "wo__list_workorders",
        "wo__get_workorder_costs",
        "iot__history",
        "tsfm__list_models",
    ):
        assert name in manifest


# -- parameter coercion ----------------------------------------------------


def test_call_tool_accepts_arguments_as_a_json_string():
    # Models routinely send the arguments object as a string. Rejecting it would
    # surface as Stirrup's opaque "Tool arguments are not valid".
    params = CallToolParams.model_validate(
        {"name": "wo__list_workorders", "arguments": '{"site": "MAIN"}'}
    )
    assert params.arguments == {"site": "MAIN"}


def test_call_tool_tolerates_missing_and_malformed_arguments():
    assert CallToolParams(name="x").arguments == {}
    assert CallToolParams.model_validate({"name": "x", "arguments": "not json"}).arguments == {}


def test_describe_tools_accepts_a_bare_name():
    assert DescribeToolsParams.model_validate({"names": "iot__history"}).names == [
        "iot__history"
    ]


def test_search_params_bound_k():
    with pytest.raises(ValueError):
        SearchToolsParams(query="x", k=99)


# -- attribution -----------------------------------------------------------


def test_gateway_call_is_credited_to_the_underlying_server():
    # Without this the whole gateway arm would report zero domain tool calls and
    # tool_bypass would read true on every run.
    assert (
        classify_tool(
            GATEWAY_CALL_TOOL, DOMAIN_SERVERS, {"name": "tsfm__list_models"}
        )
        == "domain"
    )


def test_discovery_calls_are_not_domain_work():
    assert classify_tool(GATEWAY_SEARCH_TOOL, DOMAIN_SERVERS, {"query": "x"}) == "other"
    assert classify_tool(GATEWAY_DESCRIBE_TOOL, DOMAIN_SERVERS, {"names": []}) == "other"


def test_gateway_call_without_arguments_is_not_counted_as_domain():
    assert classify_tool(GATEWAY_CALL_TOOL, DOMAIN_SERVERS, None) == "other"


def test_flat_classification_is_unchanged():
    # The gateway must not disturb how a flat run is counted, or the two arms
    # stop being comparable.
    assert classify_tool("tsfm__list_models", DOMAIN_SERVERS) == "domain"
    assert classify_tool("code_exec", DOMAIN_SERVERS) == "code"
    assert classify_tool("finish", DOMAIN_SERVERS) == "other"


# -- provider --------------------------------------------------------------


def _fake_tools():
    """Two Stirrup Tools over trivial parameter models."""
    from pydantic import BaseModel
    from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata

    class ListParams(BaseModel):
        site: str

    class HistoryParams(BaseModel):
        asset_id: str
        sensor: str

    async def _list(params: ListParams):
        return ToolResult(
            content=f"orders at {params.site}", metadata=ToolUseCountMetadata()
        )

    async def _history(params: HistoryParams):
        return ToolResult(content="readings", metadata=ToolUseCountMetadata())

    return [
        Tool(
            name="wo__list_workorders",
            description="List work orders for a site.",
            parameters=ListParams,
            executor=_list,
        ),
        Tool(
            name="iot__history",
            description="Fetch raw sensor readings over a time range.",
            parameters=HistoryParams,
            executor=_history,
        ),
    ]


class _FakeInner:
    def __init__(self, tools):
        self._tools = tools
        self.exited = False

    async def __aenter__(self):
        return self._tools

    async def __aexit__(self, *exc):
        self.exited = True


async def _enter_gateway(mode: str):
    from agent.stirrup_agent.gateway import MCPGatewayToolProvider

    gateway = MCPGatewayToolProvider(_FakeInner(_fake_tools()), mode=mode)
    tools = await gateway.__aenter__()
    return gateway, {t.name: t for t in tools}


@pytest.mark.anyio
async def test_gateway_exposes_three_tools_not_the_catalogue():
    pytest.importorskip("stirrup.core.models")
    gateway, tools = await _enter_gateway("index")

    assert set(tools) == {GATEWAY_SEARCH_TOOL, GATEWAY_DESCRIBE_TOOL, GATEWAY_CALL_TOOL}
    assert gateway.tool_count == 2


@pytest.mark.anyio
async def test_index_mode_pins_the_catalogue_and_search_mode_does_not():
    pytest.importorskip("stirrup.core.models")
    _, index_tools = await _enter_gateway("index")
    _, search_tools = await _enter_gateway("search")

    # The manifest lives in a tool description, which is what makes it a fixed
    # per-turn cost. That is the whole difference between the two modes.
    assert "wo__list_workorders" in index_tools[GATEWAY_DESCRIBE_TOOL].description
    assert "wo__list_workorders" not in search_tools[GATEWAY_DESCRIBE_TOOL].description


@pytest.mark.anyio
async def test_schemas_are_deferred_until_described():
    pytest.importorskip("stirrup.core.models")
    gateway, tools = await _enter_gateway("index")

    # No parameter schema is present anywhere in the root's tool surface.
    surface = " ".join(t.description for t in tools.values())
    assert "asset_id" not in surface

    outcome = await tools[GATEWAY_DESCRIBE_TOOL].executor(
        DescribeToolsParams(names=["iot__history"])
    )
    described = json.loads(outcome.content)
    assert "asset_id" in described["iot__history"]["parameters"]["properties"]
    assert gateway.described == {"iot__history"}


@pytest.mark.anyio
async def test_call_tool_invokes_the_underlying_tool():
    pytest.importorskip("stirrup.core.models")
    gateway, tools = await _enter_gateway("index")

    outcome = await tools[GATEWAY_CALL_TOOL].executor(
        CallToolParams(name="wo__list_workorders", arguments={"site": "MAIN"})
    )

    assert outcome.content == "orders at MAIN"
    assert gateway.call_counts["wo__list_workorders"] == 1


@pytest.mark.anyio
async def test_bad_arguments_return_the_real_error_and_the_schema():
    pytest.importorskip("stirrup.core.models")
    _, tools = await _enter_gateway("index")

    outcome = await tools[GATEWAY_CALL_TOOL].executor(
        CallToolParams(name="iot__history", arguments={"asset_id": "Chiller6"})
    )

    # The gateway's own params always validate, so the inner failure reaches our
    # executor and we can say what was wrong instead of Stirrup's fixed string.
    assert outcome.success is False
    assert "sensor" in outcome.content
    assert "properties" in outcome.content


@pytest.mark.anyio
async def test_unknown_tool_name_suggests_alternatives():
    pytest.importorskip("stirrup.core.models")
    _, tools = await _enter_gateway("index")

    outcome = await tools[GATEWAY_CALL_TOOL].executor(
        CallToolParams(name="wo__list_work_order", arguments={})
    )

    assert outcome.success is False
    assert "wo__list_workorders" in outcome.content


@pytest.mark.anyio
async def test_search_returns_ranked_names():
    pytest.importorskip("stirrup.core.models")
    _, tools = await _enter_gateway("search")

    outcome = await tools[GATEWAY_SEARCH_TOOL].executor(
        SearchToolsParams(query="raw sensor readings for an asset", k=2)
    )

    hits = json.loads(outcome.content)
    assert hits[0]["name"] == "iot__history"
    assert hits[0]["server"] == "iot"


@pytest.mark.anyio
async def test_gateway_closes_the_inner_provider():
    pytest.importorskip("stirrup.core.models")
    from agent.stirrup_agent.gateway import MCPGatewayToolProvider

    inner = _FakeInner(_fake_tools())
    gateway = MCPGatewayToolProvider(inner, mode="index")
    await gateway.__aenter__()
    await gateway.__aexit__(None, None, None)

    assert inner.exited is True


def test_gateway_is_a_real_stirrup_tool_provider():
    # Agent.__init__ decides what to enter with `isinstance(t, ToolProvider)`.
    # If this regresses, the gateway is never connected and the agent starts
    # with no domain tools at all, silently.
    models = pytest.importorskip("stirrup.core.models")
    from agent.stirrup_agent.gateway import MCPGatewayToolProvider

    assert issubclass(MCPGatewayToolProvider, models.ToolProvider)
