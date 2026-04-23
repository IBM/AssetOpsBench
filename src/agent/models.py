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


@dataclass
class TurnRecord:
    """One assistant turn: text output, tool calls, and token usage."""

    index: int
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Trajectory:
    """Full execution trace across all agent turns."""

    turns: list[TurnRecord] = field(default_factory=list)

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
