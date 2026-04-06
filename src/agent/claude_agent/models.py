"""Trajectory data models for ClaudeAgentRunner."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A single tool invocation made by the agent."""

    name: str
    input: dict
    id: str = ""


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
