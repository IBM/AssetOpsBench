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
    build_domain_finish_tool,
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


def test_domain_finish_params_coerce_rather_than_raise():
    # The model deliberately never fails validation (see its docstring): every
    # field defaults and coerces so a malformed call reaches the executor, which
    # can name the keys the model actually sent. The real contract lives there.
    assert DomainFinishParams(answer="   ").answer == "   "


@pytest.mark.anyio
async def test_domain_finish_tool_rejects_empty_answer():
    tool = build_domain_finish_tool(None)

    result = await tool.executor(DomainFinishParams(answer="   "))

    assert result.success is False
    assert "`answer` was empty" in result.content


@pytest.mark.anyio
async def test_domain_finish_tool_names_unrecognised_fields():
    tool = build_domain_finish_tool(None)

    result = await tool.executor(DomainFinishParams(result="12 sensors"))

    assert result.success is False
    assert "result" in result.content


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

# -- run metrics -----------------------------------------------------------
#
# The root/sub split is the experiment: the topology buys root context headroom
# by re-paying system prompts and tool schemas inside every delegation, so cost
# and context move in opposite directions. These figures used to exist only as
# OpenTelemetry span attributes, which made a sweep unreadable without a
# tracing backend. They are now persisted in the trajectory record too, and
# both readers take them from one computation so they cannot drift apart.


def _runner_with_turns(turns):
    from agent.models import Trajectory
    from agent.stirrup_agent.runner import StirrupAgentRunner

    runner = StirrupAgentRunner(code_backend="local", topology="subagent")
    trajectory = Trajectory()
    trajectory.turns.extend(turns)
    return runner, trajectory


def _turn(index, *, agent, depth, tokens_in, tokens_out=0, tools=()):
    from agent.models import ToolCall, TurnRecord

    return TurnRecord(
        index=index,
        text="",
        tool_calls=[ToolCall(name=n, input={}, id=f"c{index}-{i}")
                    for i, n in enumerate(tools)],
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        agent=agent,
        depth=depth,
    )


def test_run_metrics_split_root_from_subagent_tokens():
    runner, trajectory = _runner_with_turns([
        _turn(0, agent="root", depth=0, tokens_in=1_000, tools=("wo_agent",)),
        _turn(1, agent="wo_agent", depth=1, tokens_in=9_000, tools=("wo__list",)),
        _turn(2, agent="wo_agent", depth=1, tokens_in=7_000),
        _turn(3, agent="root", depth=0, tokens_in=3_000, tools=("code_exec",)),
    ])

    m = runner._run_metrics(trajectory, answer="done", started=0.0)

    assert m["input_tokens"] == 20_000
    assert m["root_input_tokens"] == 4_000
    assert m["subagent_input_tokens"] == 16_000
    # The context axis is the root's worst single turn, not the total.
    assert m["root_peak_context_tokens"] == 3_000
    assert m["root_turns"] == 2
    assert m["subagent_turns"] == 2
    assert m["topology"] == "subagent"


def test_run_metrics_root_and_subagent_tokens_sum_to_the_total():
    # The split is a filter on depth, never a separate accumulator, so it can
    # never disagree with the turns it describes.
    runner, trajectory = _runner_with_turns([
        _turn(0, agent="root", depth=0, tokens_in=500),
        _turn(1, agent="tsfm_agent", depth=1, tokens_in=1_500),
        _turn(2, agent="wo_agent", depth=1, tokens_in=2_500),
    ])

    m = runner._run_metrics(trajectory, answer="", started=0.0)

    assert m["root_input_tokens"] + m["subagent_input_tokens"] == m["input_tokens"]


def test_run_metrics_break_down_by_agent():
    runner, trajectory = _runner_with_turns([
        _turn(0, agent="root", depth=0, tokens_in=100, tools=("wo_agent",)),
        _turn(1, agent="wo_agent", depth=1, tokens_in=200, tools=("wo__list", "wo__get")),
    ])

    by_agent = runner._run_metrics(trajectory, answer="", started=0.0)["by_agent"]

    assert by_agent["root"]["turns"] == 1
    assert by_agent["wo_agent"]["turns"] == 1
    assert by_agent["wo_agent"]["tool_calls"] == 2
    assert by_agent["wo_agent"]["input_tokens"] == 200


def test_delegation_calls_are_not_counted_as_domain_work_in_metrics():
    # A delegation is named `{server}_agent` with no `__`, so it must fall
    # through to "other" while the real domain calls beneath it count once.
    runner, trajectory = _runner_with_turns([
        _turn(0, agent="root", depth=0, tokens_in=10, tools=("wo_agent",)),
        _turn(1, agent="wo_agent", depth=1, tokens_in=10, tools=("wo__list_workorders",)),
    ])

    m = runner._run_metrics(trajectory, answer="", started=0.0)

    assert m["domain_tool_calls"] == 1
    assert m["other_tool_calls"] == 1
