"""Unit tests for the Stirrup -> AssetOpsBench trajectory mapping.

These use lightweight stand-ins that mimic Stirrup's message attribute surface
(``role``, ``content``, ``tool_calls``, ``token_usage``, ``tool_call_id``), so
they run without Stirrup, the MCP servers, Docker, or a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
            _Tool(content="[{'wo': 7}]", tool_call_id="t1", name="wo__get_work_order",
                  tool_start_time=2.5, tool_end_time=3.0),
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
    assert call.input == {"asset": "CWC04013"}        # JSON string parsed
    assert call.output == "[{'wo': 7}]"
    assert call.duration_ms == 500.0                   # (3.0 - 2.5) * 1000
    assert traj.turns[0].duration_ms == 1500.0         # (2.5 - 1.0) * 1000

    assert final_answer(history, _Finish("done")) == "there are 7 open work orders"


def test_final_answer_falls_back_to_finish_reason():
    history = [[_Assistant(content="")]]
    assert final_answer(history, _Finish("computed RUL = 142 days")) == "computed RUL = 142 days"


def test_arguments_parsed_when_already_dict():
    history = [[_Assistant(content="x", tool_calls=[_TC("iot__get_sensors", {"asset": "CH6"}, "a")])]]
    traj = build_trajectory(history)
    assert traj.all_tool_calls[0].input == {"asset": "CH6"}