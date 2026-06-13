"""CLI coding-agent runners for AssetOpsBench.

A runner family that benchmarks terminal coding agents (Codex, Claude Code,
Gemini CLI, ...) against the AssetOpsBench MCP servers, using the per-agent
adapter pattern from ``rdi-berkeley/agents-last-exam`` ported onto this repo's
:class:`~agent.runner.AgentRunner` contract.

Add a new agent by subclassing :class:`CliCodingAgentRunner` and implementing
``_write_config`` / ``_build_command`` / ``_handle_event``.  See ``README.md``.
"""

from .base import CliCodingAgentRunner
from .claude_code import ClaudeCodeRunner
from .codex import CodexCliRunner
from .gemini import GeminiCliRunner

__all__ = [
    "CliCodingAgentRunner",
    "ClaudeCodeRunner",
    "CodexCliRunner",
    "GeminiCliRunner",
]

#: Registry name -> runner class, for the benchmark's ``--runner`` dispatch.
CLI_AGENT_RUNNERS = {
    CodexCliRunner.agent_name: CodexCliRunner,
    ClaudeCodeRunner.agent_name: ClaudeCodeRunner,
    GeminiCliRunner.agent_name: GeminiCliRunner,
}
