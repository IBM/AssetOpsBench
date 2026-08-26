"""Unit tests for the Stirrup -> AssetOpsBench trajectory mapping.

These use lightweight stand-ins that mimic Stirrup's message attribute surface
(``role``, ``content``, ``tool_calls``, ``token_usage``, ``tool_call_id``), so
they run without Stirrup, the MCP servers, Docker, or a model.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent._prompts import AGENT_SYSTEM_PROMPT
from agent.stirrup_agent.finish_tool import ASSETOPS_FINISH_TOOL
from agent.stirrup_agent.runner import (
    StirrupAgentRunner,
    _CONTEXT_SUMMARIZATION_CUTOFF,
    _ROOT_CONTEXT_WINDOW_TOKENS,
    _ROOT_MAX_OUTPUT_TOKENS,
    _build_full_summary_logger,
    _copy_workspace_contents,
)
from agent.stirrup_agent.trajectory import (
    build_trajectory,
    classify_tool,
    final_answer,
)
from agent.stirrup_agent.gateway import MCPGatewayToolProvider
from agent.stirrup_agent.workspace_bridge import WorkspaceBridgedMCPToolProvider

_DOMAIN = {"iot", "utilities", "fmsr", "tsfm", "wo", "vibration"}


@dataclass
class _Usage:
    input: int = 0
    answer: int = 0
    reasoning: int = 0

    @property
    def output(self) -> int:
        return self.answer + self.reasoning


@dataclass
class _TC:
    name: str
    arguments: str
    tool_call_id: str


@dataclass
class _Assistant:
    content: str
    tool_calls: list = field(default_factory=list)
    token_usage: _Usage = field(default_factory=_Usage)
    request_start_time: float | None = None
    request_end_time: float | None = None
    role: str = "assistant"


@dataclass
class _Tool:
    content: str
    tool_call_id: str
    name: str
    tool_start_time: float | None = None
    tool_end_time: float | None = None
    role: str = "tool"


@dataclass
class _Finish:
    reason: str


@dataclass
class _StructuredFinish:
    answer: str
    reason: str = ""
    paths: list[str] = field(default_factory=list)


def test_classify_tool():
    assert classify_tool("wo__get_work_order", _DOMAIN) == "domain"
    assert classify_tool("vibration__compute_fft", _DOMAIN) == "domain"
    assert classify_tool("code_exec", _DOMAIN) == "code"
    assert classify_tool("web_search", _DOMAIN) == "other"
    assert classify_tool("calculator", _DOMAIN) == "other"


def test_copy_workspace_contents_preserves_nested_outputs(tmp_path: Path):
    destination = tmp_path / "stirrup_agent_1001"
    source = destination / "docker_exec_env_abc"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "answer.txt").write_text("done", encoding="utf-8")
    (nested / "plot.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    _copy_workspace_contents(source, destination)

    assert (destination / "answer.txt").read_text(encoding="utf-8") == "done"
    assert (destination / "nested" / "plot.csv").exists()


def test_stirrup_runner_requires_workspace_when_preserving():
    with pytest.raises(ValueError, match="workspace_dir"):
        StirrupAgentRunner(preserve_workspace=True)


def test_stirrup_runner_rejects_unsupported_code_backend():
    with pytest.raises(ValueError, match="code_backend"):
        StirrupAgentRunner(code_backend="e2b")


def test_stirrup_runner_bridges_mcp_results_when_code_is_enabled():
    runner = StirrupAgentRunner(code_backend="local")

    # The code track is [code_provider, *handoff_tools, mcp_provider]; the
    # handoff tools in the middle are why this cannot unpack to a fixed pair.
    code_provider, *rest = runner._build_tools()
    mcp_provider = rest[-1]

    assert isinstance(mcp_provider, WorkspaceBridgedMCPToolProvider)
    assert mcp_provider._exec_env is code_provider
    # Flat topology connects every registered server through one provider.
    assert mcp_provider._server_names is None


def test_gateway_topology_wraps_the_bridged_provider():
    runner = StirrupAgentRunner(code_backend="local", topology="gateway")

    code_provider, *rest = runner._build_tools()
    gateway = rest[-1]

    # The gateway must sit on top of the bridge, not replace it, or oversized
    # MCP results stop spilling into the workspace and land in the context.
    assert isinstance(gateway, MCPGatewayToolProvider)
    assert isinstance(gateway._inner, WorkspaceBridgedMCPToolProvider)
    assert gateway._inner._exec_env is code_provider


def test_gateway_topology_works_without_code_execution():
    # Unlike a delegated topology, the gateway needs no execution environment,
    # so it can be compared with flat on the same track as the other runners.
    runner = StirrupAgentRunner(code_enabled=False, topology="gateway")

    (gateway,) = runner._build_tools()

    assert isinstance(gateway, MCPGatewayToolProvider)


def test_unknown_topology_is_rejected_at_construction():
    with pytest.raises(ValueError, match="topology"):
        StirrupAgentRunner(topology="subagent")


def test_stirrup_runner_uses_shared_prompt_when_code_is_disabled():
    runner = StirrupAgentRunner(code_enabled=False)

    assert runner._build_system_prompt() == AGENT_SYSTEM_PROMPT


def test_stirrup_runner_forwards_temperature_to_litellm_client():
    runner = StirrupAgentRunner(
        model="watsonx/ibm/granite-4-h-small",
        temperature=0.2,
        reasoning_effort="high",
    )

    client = runner._build_client()

    assert client._kwargs == {"temperature": 0.2}
    assert client._reasoning_effort == "high"
    assert client.max_tokens == 64_000
    assert client.context_window_tokens == 100_000


def test_stirrup_runner_forwards_temperature_to_router_client(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TOKENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.example/v1")

    runner = StirrupAgentRunner(
        model="tokenrouter/MiniMax-M3",
        temperature=0.2,
        reasoning_effort="medium",
    )

    client = runner._build_client()

    from stirrup.clients.chat_completions_client import ChatCompletionsClient

    assert isinstance(client, ChatCompletionsClient)
    assert client._kwargs == {"temperature": 0.2}
    assert client._reasoning_effort == "medium"
    assert client.max_tokens == 64_000
    assert client.context_window_tokens == 100_000


def test_stirrup_runner_uses_75k_summarization_trigger():
    assert _ROOT_CONTEXT_WINDOW_TOKENS == 100_000
    assert _CONTEXT_SUMMARIZATION_CUTOFF == 0.75
    assert _ROOT_CONTEXT_WINDOW_TOKENS * _CONTEXT_SUMMARIZATION_CUTOFF == 75_000
    # Stirrup validates this pair in the client constructor.
    assert _ROOT_MAX_OUTPUT_TOKENS <= _ROOT_CONTEXT_WINDOW_TOKENS


def test_full_summary_logger_does_not_truncate(capsys: pytest.CaptureFixture[str]):
    marker = "SUMMARY_END_MARKER_1234567890"
    summary = "x" * 900 + marker

    logger = _build_full_summary_logger()
    logger.context_summarization_complete(summary, "unused bridge")

    assert marker in capsys.readouterr().out


def test_build_trajectory_maps_turns_calls_and_outputs():
    # history is list[list[ChatMessage]] (per-turn grouping)
    history = [
        [
            _Assistant(
                content="let me check work orders",
                tool_calls=[_TC("wo__get_work_order", '{"asset": "CWC04013"}', "t1")],
                token_usage=_Usage(input=20, answer=8, reasoning=2),
                request_start_time=1.0,
                request_end_time=2.5,
            ),
            _Tool(
                content="[{'wo': 7}]",
                tool_call_id="t1",
                name="wo__get_work_order",
                tool_start_time=2.5,
                tool_end_time=3.0,
            ),
        ],
        [
            _Assistant(
                content="there are 7 open work orders",
                token_usage=_Usage(input=5, answer=6),
            ),
        ],
    ]
    traj = build_trajectory(history)

    assert len(traj.turns) == 2
    assert traj.total_input_tokens == 25
    assert traj.total_output_tokens == 16  # (8+2) + (6+0)

    call = traj.all_tool_calls[0]
    assert call.name == "wo__get_work_order"
    assert call.input == {"asset": "CWC04013"}  # JSON string parsed
    assert call.output == "[{'wo': 7}]"
    assert call.duration_ms == 500.0  # (3.0 - 2.5) * 1000
    assert traj.turns[0].duration_ms == 1500.0  # (2.5 - 1.0) * 1000

    assert final_answer(history, None) == "there are 7 open work orders"


def test_final_answer_prefers_finish_turn_content_over_finish_reason():
    history = [
        [
            _Assistant(
                content='{"response":"FORMAT_OK"}',
                tool_calls=[_TC("finish", '{"reason":"done","paths":[]}', "f1")],
            )
        ]
    ]
    assert (
        final_answer(
            history,
            _Finish(
                "Returned the exact required JSON object without calling any tools."
            ),
        )
        == '{"response":"FORMAT_OK"}'
    )


def test_final_answer_prefers_structured_finish_answer_over_turn_content():
    history = [
        [
            _Assistant(
                content="Task complete.",
                tool_calls=[_TC("finish", '{"answer":"42"}', "f1")],
            )
        ]
    ]

    assert final_answer(history, _StructuredFinish(answer="42")) == "42"


def test_final_answer_uses_finish_reason_when_finish_turn_content_is_empty():
    history = [
        [_Assistant(content="I will inspect the work orders.")],
        [
            _Assistant(
                content="",
                tool_calls=[_TC("finish", '{"reason":"done","paths":[]}', "f1")],
            )
        ],
    ]
    assert (
        final_answer(history, _Finish("computed RUL = 142 days"))
        == "computed RUL = 142 days"
    )


def test_final_answer_falls_back_to_assistant_text_without_finish_reason():
    history = [[_Assistant(content="The pump is healthy.")]]
    assert final_answer(history, None) == "The pump is healthy."


def test_arguments_parsed_when_already_dict():
    history = [
        [
            _Assistant(
                content="x", tool_calls=[_TC("iot__get_sensors", {"asset": "CH6"}, "a")]
            )
        ]
    ]
    traj = build_trajectory(history)
    assert traj.all_tool_calls[0].input == {"asset": "CH6"}


@pytest.mark.anyio
async def test_run_persists_legacy_final_answer(
    monkeypatch: pytest.MonkeyPatch,
):
    runner_module = importlib.import_module("agent.stirrup_agent.runner")
    client = object()
    history = [
        [
            _Assistant(
                content="calculating",
                tool_calls=[_TC("code_exec", '{"cmd":"calculate"}', "c1")],
            ),
            _Tool(content="FINAL=7", tool_call_id="c1", name="code_exec"),
        ],
        [
            _Assistant(
                content='The requested result is {"count":7}.',
                tool_calls=[_TC("finish", '{"reason":"done"}', "f1")],
            )
        ],
    ]

    class _FakeAgent:
        def __init__(self, **kwargs):
            assert kwargs["client"] is client
            assert kwargs["finish_tool"] is ASSETOPS_FINISH_TOOL

        def session(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def run(self, question):
            assert question == "Return a JSON object."
            return _Finish("run complete"), history, {}

    persist = MagicMock()
    monkeypatch.setattr("stirrup.Agent", _FakeAgent)
    monkeypatch.setattr(runner_module, "persist_trajectory", persist)

    runner = StirrupAgentRunner(server_paths={}, code_enabled=False)
    monkeypatch.setattr(runner, "_build_client", lambda: client)
    monkeypatch.setattr(runner, "_build_tools", lambda: [])

    result = await runner.run("Return a JSON object.")

    assert result.answer == 'The requested result is {"count":7}.'
    persist.assert_called_once()
    assert persist.call_args.kwargs["answer"] == result.answer


@pytest.mark.anyio
async def test_run_uses_structured_finish_without_repair_call(
    monkeypatch: pytest.MonkeyPatch,
):
    runner_module = importlib.import_module("agent.stirrup_agent.runner")
    client = object()
    history = [
        [
            _Assistant(
                content="Task complete.",
                tool_calls=[_TC("finish", '{"answer":"[1,2]"}', "f1")],
            )
        ]
    ]

    class _FakeAgent:
        def __init__(self, **kwargs):
            assert kwargs["client"] is client
            assert kwargs["finish_tool"] is ASSETOPS_FINISH_TOOL

        def session(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def run(self, question):
            return _StructuredFinish(answer="[1,2]"), history, {}

    persist = MagicMock()
    monkeypatch.setattr("stirrup.Agent", _FakeAgent)
    monkeypatch.setattr(runner_module, "persist_trajectory", persist)

    runner = StirrupAgentRunner(server_paths={}, code_enabled=False)
    monkeypatch.setattr(runner, "_build_client", lambda: client)
    monkeypatch.setattr(runner, "_build_tools", lambda: [])

    result = await runner.run("Return a JSON array.")

    assert result.answer == "[1,2]"
    assert persist.call_args.kwargs["answer"] == "[1,2]"
