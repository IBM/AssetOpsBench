"""AgentRunner implementation backed by the claude-agent-sdk.

Each registered MCP server is connected as a stdio MCP server so Claude can
call IoT / FMSR / TSFM / utilities tools directly without a custom plan loop.

Usage::

    import anyio
    from agent.claude_agent import ClaudeAgentRunner

    runner = ClaudeAgentRunner()
    result = anyio.run(runner.run, "What sensors are on Chiller 6?")
    print(result.answer)
"""

from __future__ import annotations

import logging
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from ..models import AgentResult
from ..plan_execute.executor import DEFAULT_SERVER_PATHS
from ..runner import AgentRunner

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-opus-4-6"

_SYSTEM_PROMPT = """\
You are an industrial asset operations assistant with access to MCP tools for
querying IoT sensor data, failure mode and symptom records, time-series
forecasting models, and work order management.

Answer the user's question concisely and accurately using the available tools.
When you retrieve data, include the key numbers or names in your answer.
"""


def _build_mcp_servers(
    server_paths: dict[str, Path | str],
) -> dict[str, dict]:
    """Convert server_paths entries into claude-agent-sdk mcp_servers dicts.

    Entry-point names (str without path separators) become
    ``{"command": "uv", "args": ["run", name]}``.
    Path objects become ``{"command": "uv", "args": ["run", str(path)]}``.
    """
    mcp: dict[str, dict] = {}
    for name, spec in server_paths.items():
        if isinstance(spec, Path):
            mcp[name] = {"command": "uv", "args": ["run", str(spec)]}
        else:
            # uv entry-point name, e.g. "iot-mcp-server"
            mcp[name] = {"command": "uv", "args": ["run", spec]}
    return mcp


class ClaudeAgentRunner(AgentRunner):
    """Agent runner that delegates to the claude-agent-sdk agentic loop.

    The sdk handles tool discovery, invocation, and multi-turn conversation
    against the registered MCP servers.  ``history`` is ``None`` until a
    structured history type is decided.

    Args:
        llm: Unused — ClaudeAgentRunner uses the claude-agent-sdk directly.
             Accepted for interface compatibility with ``AgentRunner``.
        server_paths: MCP server specs identical to ``PlanExecuteRunner``.
                      Defaults to all registered servers.
        model: Claude model ID to use (default: ``claude-opus-4-6``).
        max_turns: Maximum agentic loop turns (default: 30).
        permission_mode: claude-agent-sdk permission mode (default: ``"default"``).
    """

    def __init__(
        self,
        llm=None,
        server_paths: dict[str, Path | str] | None = None,
        model: str = _DEFAULT_MODEL,
        max_turns: int = 30,
        permission_mode: str = "default",
    ) -> None:
        super().__init__(llm, server_paths)
        self._model = model
        self._max_turns = max_turns
        self._permission_mode = permission_mode
        self._resolved_server_paths: dict[str, Path | str] = (
            server_paths if server_paths is not None else dict(DEFAULT_SERVER_PATHS)
        )

    async def run(self, question: str) -> AgentResult:
        """Run the claude-agent-sdk loop for *question*.

        Args:
            question: Natural-language question to answer.

        Returns:
            AgentResult where ``answer`` holds the sdk's final response
            and ``history`` is ``None`` (type TBD).
        """
        mcp_servers = _build_mcp_servers(self._resolved_server_paths)

        options = ClaudeAgentOptions(
            model=self._model,
            system_prompt=_SYSTEM_PROMPT,
            mcp_servers=mcp_servers,
            max_turns=self._max_turns,
            permission_mode=self._permission_mode,
        )

        _log.info("ClaudeAgentRunner: starting query (model=%s)", self._model)
        answer = ""
        async for message in query(prompt=question, options=options):
            if isinstance(message, ResultMessage):
                answer = message.result or ""
                _log.info(
                    "ClaudeAgentRunner: done (stop_reason=%s)", message.stop_reason
                )

        return AgentResult(question=question, answer=answer, history=None)
