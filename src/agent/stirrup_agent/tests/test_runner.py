"""Unit tests for the Stirrup -> AssetOpsBench trajectory mapping.

These use lightweight stand-ins that mimic Stirrup's message attribute surface
(``role``, ``content``, ``tool_calls``, ``token_usage``, ``tool_call_id``), so
they run without Stirrup, the MCP servers, Docker, or a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agent.stirrup_agent.runner import (
    StirrupAgentRunner,
    _ResponsesThenChatClient,
    _copy_workspace_contents,
)
from agent.stirrup_agent.trajectory import (
    build_trajectory,
    classify_tool,
    final_answer,
)

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
    from stirrup.clients.open_responses_client import OpenResponsesClient

    assert isinstance(client._responses_client, OpenResponsesClient)
    assert isinstance(client._chat_client, ChatCompletionsClient)
    assert client._responses_client._kwargs == {"temperature": 0.2}
    assert client._chat_client._kwargs == {"temperature": 0.2}
    assert client._responses_client._reasoning_effort == "medium"
    assert client._chat_client._reasoning_effort == "medium"
    assert client.max_tokens == 64_000


class _FakeClient:
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0
        self.max_tokens = 64_000
        self.model_slug = "test-model"

    async def generate(self, messages, tools):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _LastAttempt:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def exception(self) -> Exception:
        return self._error


class _RetryError(Exception):
    def __init__(self, error: Exception) -> None:
        super().__init__("retries exhausted")
        self.last_attempt = _LastAttempt(error)


@pytest.mark.anyio
async def test_router_client_prefers_open_responses():
    responses = _FakeClient(result="responses result")
    chat = _FakeClient(result="chat response")
    client = _ResponsesThenChatClient(responses, chat)

    assert await client.generate([], {}) == "responses result"
    assert responses.calls == 1
    assert chat.calls == 0


@pytest.mark.anyio
async def test_router_client_falls_back_to_chat_and_sticks_to_it():
    responses = _FakeClient(error=_StatusError(404))
    chat = _FakeClient(result="chat response")
    client = _ResponsesThenChatClient(responses, chat)

    assert await client.generate([], {}) == "chat response"
    assert await client.generate([], {}) == "chat response"
    assert responses.calls == 1
    assert chat.calls == 2


@pytest.mark.anyio
async def test_router_client_falls_back_after_retried_server_error():
    responses = _FakeClient(error=_RetryError(_StatusError(500)))
    chat = _FakeClient(result="chat response")
    client = _ResponsesThenChatClient(responses, chat)

    assert await client.generate([], {}) == "chat response"
    assert responses.calls == 1
    assert chat.calls == 1


@pytest.mark.anyio
async def test_router_client_does_not_mask_non_interface_errors():
    responses = _FakeClient(error=RuntimeError("request failed"))
    chat = _FakeClient(result="chat response")
    client = _ResponsesThenChatClient(responses, chat)

    with pytest.raises(RuntimeError, match="request failed"):
        await client.generate([], {})
    assert chat.calls == 0


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


def test_final_answer_prefers_finish_reason_over_earlier_assistant_text():
    history = [
        [_Assistant(content="I will inspect the work orders.")],
        [_Assistant(content="")],
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
