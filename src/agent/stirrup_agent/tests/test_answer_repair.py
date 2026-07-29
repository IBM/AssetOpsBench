"""Tests for Stirrup's post-run answer-format repair pass."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.models import ToolCall, Trajectory, TurnRecord
from agent.stirrup_agent.answer_repair import (
    _MAX_EVIDENCE_CHARS,
    build_repair_evidence,
    repair_answer,
)


def _trajectory(*, output: object = "FINAL=7") -> Trajectory:
    return Trajectory(
        turns=[
            TurnRecord(
                index=0,
                text="working",
                tool_calls=[ToolCall(name="code_exec", input={}, output="OLD=3")],
            ),
            TurnRecord(
                index=1,
                text="computed result",
                tool_calls=[
                    ToolCall(name="code_exec", input={}, output="INTERMEDIATE=5"),
                    ToolCall(name="code_exec", input={}, output=output),
                ],
            ),
            TurnRecord(index=2, text='The requested answer is {"count": 7}.'),
        ]
    )


class _Client:
    def __init__(self, content: str = '{"count": 7}', error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[tuple[list, dict]] = []

    async def generate(self, messages, tools):
        self.calls.append((messages, tools))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.content)


def test_build_repair_evidence_uses_only_requested_turn_content():
    evidence = build_repair_evidence(
        question="Return JSON with count.",
        answer="Finished the calculation.",
        trajectory=_trajectory(),
    )

    assert evidence == {
        "question": "Return JSON with count.",
        "original_answer": "Finished the calculation.",
        "last_turn_text": 'The requested answer is {"count": 7}.',
        "second_to_last_turn_last_tool_output": "FINAL=7",
    }


def test_build_repair_evidence_returns_none_for_null_tool_output():
    assert (
        build_repair_evidence(
            question="q",
            answer="original",
            trajectory=_trajectory(output=None),
        )
        is None
    )


def test_build_repair_evidence_returns_none_without_preceding_tool_call():
    trajectory = Trajectory(
        turns=[TurnRecord(index=0, text="work"), TurnRecord(index=1, text="answer")]
    )

    assert (
        build_repair_evidence(
            question="q", answer="original", trajectory=trajectory
        )
        is None
    )


def test_build_repair_evidence_bounds_large_values_and_keeps_tail():
    output = "HEAD" + ("x" * (_MAX_EVIDENCE_CHARS + 100)) + "FINAL=7"
    evidence = build_repair_evidence(
        question="q", answer="a", trajectory=_trajectory(output=output)
    )

    bounded = evidence["second_to_last_turn_last_tool_output"]
    assert len(bounded) == _MAX_EVIDENCE_CHARS
    assert bounded.startswith("HEAD")
    assert bounded.endswith("FINAL=7")
    assert "middle omitted" in bounded


@pytest.mark.anyio
async def test_repair_answer_uses_tools_disabled_call_and_returns_response():
    client = _Client()

    result = await repair_answer(
        client,
        question="Return JSON with count.",
        answer="Finished the calculation.",
        trajectory=_trajectory(),
    )

    assert result == '{"count": 7}'
    assert len(client.calls) == 1
    messages, tools = client.calls[0]
    assert tools == {}
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    evidence = json.loads(messages[1].content.split("\n", 1)[1])
    assert evidence["last_turn_text"] == 'The requested answer is {"count": 7}.'
    assert evidence["second_to_last_turn_last_tool_output"] == "FINAL=7"


@pytest.mark.anyio
async def test_repair_answer_copies_original_without_call_for_null_output():
    client = _Client(content="invented")

    result = await repair_answer(
        client,
        question="q",
        answer="original answer",
        trajectory=_trajectory(output=None),
    )

    assert result == "original answer"
    assert client.calls == []


@pytest.mark.anyio
async def test_repair_answer_empty_response_falls_back_to_original():
    client = _Client(content="  \n")

    result = await repair_answer(
        client,
        question="q",
        answer="original answer",
        trajectory=_trajectory(),
    )

    assert result == "original answer"


@pytest.mark.anyio
async def test_repair_answer_client_error_falls_back_to_original():
    client = _Client(error=RuntimeError("model unavailable"))

    result = await repair_answer(
        client,
        question="q",
        answer="original answer",
        trajectory=_trajectory(),
    )

    assert result == "original answer"
