"""Tests for the sub-agent topology.

Everything here runs without Stirrup, the MCP servers, Docker, or a model:
:mod:`agent.stirrup_agent.trajectory` has no Stirrup import at all, and
:mod:`agent.stirrup_agent.subagents` defers its Stirrup imports into the
functions that need them, so the manifests and the flattening logic are
testable on their own.

The stand-ins mimic Stirrup 0.2's block-based assistant messages (``blocks``
with ``kind`` discriminators) rather than the 0.1 channel fields, which is the
shape the runner now reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agent.stirrup_agent.subagents import (
    DOMAIN_MANIFESTS,
    ROOT_SERVERS,
    SUBAGENT_CONTEXT_WINDOW_TOKENS,
    SUBAGENT_MAX_OUTPUT_TOKENS,
    SUBAGENT_SERVERS,
    DomainFinishParams,
    SubAgentHistoryRecorder,
)
from agent.stirrup_agent.trajectory import build_trajectory, classify_tool


# -- stand-ins -------------------------------------------------------------


@dataclass
class _TextBlock:
    text: str
    kind: str = "text"


@dataclass
class _ToolCallBlock:
    name: str
    arguments: str = "{}"
    tool_call_id: str = ""
    kind: str = "tool_call"


@dataclass
class _Usage:
    input: int = 0
    output: int = 0


@dataclass
class _Assistant:
    blocks: list = field(default_factory=list)
    token_usage: _Usage = field(default_factory=_Usage)
    role: str = "assistant"
    request_start_time: float | None = None
    request_end_time: float | None = None


@dataclass
class _ToolMsg:
    tool_call_id: str
    content: str
    role: str = "tool"
    tool_start_time: float | None = None
    tool_end_time: float | None = None


def _assistant(text: str, calls: list[_ToolCallBlock], *, tin: int, tout: int):
    return _Assistant(
        blocks=[_TextBlock(text), *calls], token_usage=_Usage(input=tin, output=tout)
    )


# -- manifests -------------------------------------------------------------


def test_every_delegated_server_has_a_routing_manifest():
    # Under --topology subagent the manifest is the only thing the root agent
    # knows about a domain, so a missing one is a silently unroutable server.
    assert SUBAGENT_SERVERS <= set(DOMAIN_MANIFESTS)


def test_root_and_delegated_servers_do_not_overlap():
    assert not (SUBAGENT_SERVERS & ROOT_SERVERS)


def test_subagent_output_cap_fits_its_context_window():
    # Stirrup 0.2 validates max_tokens <= context_window_tokens in the client
    # constructor, so an inverted pair fails at construction, not at run time.
    assert SUBAGENT_MAX_OUTPUT_TOKENS <= SUBAGENT_CONTEXT_WINDOW_TOKENS


# -- finish params ---------------------------------------------------------


def test_domain_finish_params_reject_empty_answer():
    with pytest.raises(ValueError):
        DomainFinishParams(answer="   ")


def test_domain_finish_params_carry_artifacts_and_entities():
    params = DomainFinishParams(
        answer="12 sensors",
        entities={"asset_id": "Chiller6", "sensor_names": "SUPPLY_TEMP,RETURN_TEMP"},
        artifacts=[
            {
                "workspace_file": "mcp_results/iot__history_abc_def.json",
                "tool": "iot__history",
                "bytes": 4_200_000,
                "sha256": "deadbeef",
            }
        ],
    )
    dumped = params.model_dump()
    # Only this dump and the last assistant message cross back into the root's
    # context, so anything absent here is unrecoverable by the root.
    assert dumped["artifacts"][0]["workspace_file"].startswith("mcp_results/")
    assert dumped["entities"]["asset_id"] == "Chiller6"


# -- classification --------------------------------------------------------


def test_delegation_calls_are_not_counted_as_domain_calls():
    servers = {"iot", "tsfm", "wo", "fmsr", "vibration", "utilities"}
    # The real domain call, one level down.
    assert classify_tool("tsfm__list_models", servers) == "domain"
    # The delegation that produced it. Counting this as domain would double
    # count every delegated call against agent.domain_tool_calls.
    assert classify_tool("tsfm_agent", servers) == "other"
    assert classify_tool("code_exec", servers) == "code"


# -- flattening ------------------------------------------------------------


def test_subagent_turns_are_spliced_after_the_calling_root_turn():
    root_history = [
        [
            _assistant(
                "Delegating to tsfm.",
                [_ToolCallBlock(name="tsfm_agent", tool_call_id="c1")],
                tin=5_000,
                tout=100,
            ),
            _ToolMsg(tool_call_id="c1", content="<sub_agent_result>...</sub_agent_result>"),
        ],
        [_assistant("Done.", [], tin=6_000, tout=50)],
    ]
    sub_history = [
        [
            _assistant(
                "",
                [_ToolCallBlock(name="tsfm__list_models", tool_call_id="s1")],
                tin=9_000,
                tout=40,
            ),
            _ToolMsg(tool_call_id="s1", content="[...]"),
        ]
    ]

    traj = build_trajectory(root_history, sub_histories={"tsfm_agent": [sub_history]})

    assert [t.agent for t in traj.turns] == ["root", "tsfm_agent", "root"]
    assert [t.depth for t in traj.turns] == [0, 1, 0]
    assert [t.index for t in traj.turns] == [0, 1, 2]

    # Tool counts are tree-wide, so the domain call survives delegation.
    names = [tc.name for tc in traj.all_tool_calls]
    assert "tsfm__list_models" in names

    # Token totals are tree-wide; root-only accounting is a filter on depth.
    assert traj.total_input_tokens == 20_000
    assert sum(t.input_tokens for t in traj.turns if t.depth == 0) == 11_000


def test_repeated_delegations_consume_their_histories_in_call_order():
    root_history = [
        [_assistant("", [_ToolCallBlock(name="iot_agent", tool_call_id="a")], tin=1, tout=1)],
        [_assistant("", [_ToolCallBlock(name="iot_agent", tool_call_id="b")], tin=1, tout=1)],
    ]
    first = [[_assistant("first", [], tin=1, tout=1)]]
    second = [[_assistant("second", [], tin=1, tout=1)]]

    traj = build_trajectory(root_history, sub_histories={"iot_agent": [first, second]})

    nested = [t.text for t in traj.turns if t.depth == 1]
    assert nested == ["first", "second"]


def test_missing_subagent_history_degrades_instead_of_raising():
    # A sub-agent that errored before returning leaves no recorded history. A
    # partial trajectory beats losing the whole run's telemetry.
    root_history = [
        [_assistant("", [_ToolCallBlock(name="wo_agent", tool_call_id="a")], tin=1, tout=1)],
        [_assistant("", [_ToolCallBlock(name="wo_agent", tool_call_id="b")], tin=1, tout=1)],
    ]
    only_one = [[_assistant("first", [], tin=1, tout=1)]]

    traj = build_trajectory(root_history, sub_histories={"wo_agent": [only_one]})

    assert [t.depth for t in traj.turns] == [0, 1, 0]


def test_flat_run_is_unchanged_without_sub_histories():
    history = [
        [
            _assistant(
                "Looking up sensors.",
                [_ToolCallBlock(name="iot__measured_sensors", tool_call_id="c1")],
                tin=1_000,
                tout=20,
            ),
            _ToolMsg(tool_call_id="c1", content="SUPPLY_TEMP"),
        ]
    ]

    traj = build_trajectory(history)

    assert len(traj.turns) == 1
    assert traj.turns[0].agent == "root"
    assert traj.turns[0].depth == 0
    assert traj.turns[0].tool_calls[0].output == "SUPPLY_TEMP"


# -- recorder --------------------------------------------------------------


def test_recorder_counts_calls_across_domains():
    recorder = SubAgentHistoryRecorder()
    recorder.record("iot_agent", [["h1"]])
    recorder.record("iot_agent", [["h2"]])
    recorder.record("tsfm_agent", [["h3"]])

    assert recorder.call_count == 3
    assert len(recorder.histories["iot_agent"]) == 2