"""Top-level data models for the agent orchestration layer.

The trajectory types (:class:`ToolCall`, :class:`TurnRecord`,
:class:`Trajectory`) are shared across every SDK-driven runner
(:class:`~agent.claude_agent.ClaudeAgentRunner`,
:class:`~agent.openai_agent.OpenAIAgentRunner`,
:class:`~agent.deep_agent.DeepAgentRunner`) because each SDK reports the
same per-turn shape: some text, zero or more tool calls, and token usage.
The plan-execute runner uses its own plan-shaped models in
:mod:`agent.plan_execute.models`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation made by the agent."""

    name: str
    input: dict
    id: str = ""
    output: object = None
    duration_ms: float | None = None
    """Wall-clock time spent inside the tool.  ``None`` when the runner's
    SDK does not expose per-tool timing hooks."""


@dataclass
class TurnRecord:
    """One assistant turn: text output, tool calls, and token usage."""

    index: int
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float | None = None
    """Wall-clock time from turn start to turn end.  ``None`` when the
    runner cannot observe per-turn boundaries cleanly."""


@dataclass
class Trajectory:
    """Full execution trace across all agent turns."""

    turns: list[TurnRecord] = field(default_factory=list)
    started_at: str | None = None
    """ISO-8601 UTC timestamp of when ``run()`` began, for replay
    alignment with the corresponding trace.  Populated by the runner."""

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def all_tool_calls(self) -> list[ToolCall]:
        return [tc for turn in self.turns for tc in turn.tool_calls]


@dataclass
class AgentResult:
    """Result returned by any AgentRunner."""

    question: str
    answer: str
    trajectory: Any


# ---------------------------------------------------------------------------
# Uniform final-result extraction (framework-agnostic).
#
# Every runner records tool calls the same way (Trajectory.all_tool_calls), so the most reliable,
# framework-independent way to get the final answer is to have the agent call the utilities
# `record_final_answer` tool with the formatted answer just before finishing, and read it back from
# the trajectory. MCP tools arrive server-prefixed (e.g. `utilities__record_final_answer`), so match
# on the suffix. Returns None when the agent did not call it, so callers can fall back to their
# framework-native extraction.
# ---------------------------------------------------------------------------
FINAL_ANSWER_TOOL = "record_final_answer"


def final_answer_from_trajectory(trajectory: "Trajectory") -> str | None:
    """The last `record_final_answer` tool-call argument, or None if the agent never called it."""
    for tc in reversed(trajectory.all_tool_calls):
        name = tc.name or ""
        if name == FINAL_ANSWER_TOOL or name.endswith("__" + FINAL_ANSWER_TOOL):
            arg = tc.input
            value = arg.get("answer") if isinstance(arg, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None
