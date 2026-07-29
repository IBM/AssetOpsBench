"""Unit tests for the Stirrup -> AssetOpsBench trajectory mapping.

These use lightweight stand-ins that mimic Stirrup's message attribute surface
(``role``, ``content``, ``tool_calls``, ``token_usage``, ``tool_call_id``), so
they run without Stirrup, the MCP servers, Docker, or a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agent._prompts import AGENT_SYSTEM_PROMPT
from agent.stirrup_agent.runner import (
    StirrupAgentRunner,
    _CONTEXT_SUMMARIZATION_CUTOFF,
    _WORKING_CONTEXT_BUDGET,
    _build_full_summary_logger,
    _copy_workspace_contents,
)
from agent.stirrup_agent.trajectory import (
    build_trajectory,
    classify_tool,
    final_answer,
)
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

    code_provider, mcp_provider = runner._build_tools()

    assert isinstance(mcp_provider, WorkspaceBridgedMCPToolProvider)
    assert mcp_provider._exec_env is code_provider


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

    provider_client = client._client
    assert provider_client._kwargs == {"temperature": 0.2}
    assert provider_client._reasoning_effort == "high"
    assert provider_client.max_tokens == 64_000
    assert client.max_tokens == 100_000


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

    router_client = client._client
    assert isinstance(router_client, ChatCompletionsClient)
    assert router_client._kwargs == {"temperature": 0.2}
    assert router_client._reasoning_effort == "medium"
    assert router_client.max_tokens == 64_000
    assert client.max_tokens == 100_000


def test_stirrup_runner_uses_50k_summarization_trigger():
    assert _WORKING_CONTEXT_BUDGET == 100_000
    assert _CONTEXT_SUMMARIZATION_CUTOFF == 0.50
    assert _WORKING_CONTEXT_BUDGET * _CONTEXT_SUMMARIZATION_CUTOFF == 50_000


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
