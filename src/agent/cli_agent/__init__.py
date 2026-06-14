"""CLI coding-agent runner subpackage (Codex, Claude Code, Gemini, ...)."""

from .base import CliCodingAgentRunner
from .claude_code import ClaudeCodeRunner
from .codex import CodexCliRunner
from .gemini import GeminiCliRunner

__all__ = [
    "CliCodingAgentRunner",
    "ClaudeCodeRunner",
    "CodexCliRunner",
    "GeminiCliRunner",
    "CLI_AGENT_RUNNERS",
]

#: Registry name -> runner class, for the benchmark's ``--runner`` dispatch.
CLI_AGENT_RUNNERS = {
    CodexCliRunner.agent_name: CodexCliRunner,
    ClaudeCodeRunner.agent_name: ClaudeCodeRunner,
    GeminiCliRunner.agent_name: GeminiCliRunner,
}
