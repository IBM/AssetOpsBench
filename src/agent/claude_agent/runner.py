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

import datetime as _dt
import logging
import os
import time
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, HookMatcher, ResultMessage, query
from claude_agent_sdk import TextBlock, ToolUseBlock

from observability import agent_run_span, persist_trajectory

from .._litellm import LITELLM_PREFIX, resolve_model
from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, ToolCall, Trajectory, TurnRecord
from ..runner import AgentRunner

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "litellm_proxy/aws/claude-opus-4-6"


def _sdk_env(model_id: str) -> dict[str, str] | None:
    """Build env overrides for the claude-agent-sdk subprocess.

    When routing through a LiteLLM proxy the SDK needs the proxy URL and key
    under its own env var names.  We derive them from the LITELLM_* vars so
    the user never has to set SDK-internal vars directly.
    """
    if not model_id.startswith(LITELLM_PREFIX):
        return None
    env: dict[str, str] = {}
    if base_url := os.environ.get("LITELLM_BASE_URL"):
        env["ANTHROPIC_BASE_URL"] = base_url
    if api_key := os.environ.get("LITELLM_API_KEY"):
        env["ANTHROPIC_API_KEY"] = api_key
    return env or None


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
    against the registered MCP servers.

    Args:
        llm: Unused — ClaudeAgentRunner uses the claude-agent-sdk directly.
             Accepted for interface compatibility with ``AgentRunner``.
        server_paths: MCP server specs identical to ``PlanExecuteRunner``.
                      Defaults to all registered servers.
        model: Claude model ID to use (default: ``litellm_proxy/aws/claude-opus-4-6``).
        max_turns: Maximum agentic loop turns (default: 30).
        permission_mode: claude-agent-sdk permission mode (default: ``"default"``).
    """

    def __init__(
        self,
        llm=None,
        server_paths: dict[str, Path | str] | None = None,
        model: str = _DEFAULT_MODEL,
        max_turns: int = 30,
        permission_mode: str = "bypassPermissions",
    ) -> None:
        super().__init__(llm, server_paths)
        self._model = resolve_model(model)
        self._sdk_env = _sdk_env(model)
        self._max_turns = max_turns
        self._permission_mode = permission_mode
        self._mcp_servers = _build_mcp_servers(self._server_paths)

    async def run(self, question: str) -> AgentResult:
        """Run the claude-agent-sdk loop for *question*.

        Args:
            question: Natural-language question to answer.

        Returns:
            AgentResult with the final answer and full execution trajectory.
        """
        with agent_run_span(
            "claude-agent", model=self._model, question=question
        ) as span:
            options = ClaudeAgentOptions(
                model=self._model,
                system_prompt=AGENT_SYSTEM_PROMPT,
                mcp_servers=self._mcp_servers,
                max_turns=self._max_turns,
                permission_mode=self._permission_mode,
                env=self._sdk_env,
            )

            _log.info("ClaudeAgentRunner: starting query (model=%s)", self._model)
            answer = ""
            run_started = time.perf_counter()
            trajectory = Trajectory(started_at=_dt.datetime.now(_dt.UTC).isoformat())
            turn_index = 0
            last_turn_start = run_started
            tool_outputs: dict[str, object] = {}

            async def _capture_tool_output(input_data, tool_use_id: str, context) -> dict:
                resp = input_data.get("tool_response") if isinstance(input_data, dict) else input_data
                if isinstance(resp, dict):
                    tool_outputs[tool_use_id] = resp.get("content", resp)
                else:
                    tool_outputs[tool_use_id] = resp
                return {}

            # Only PostToolUse is registered.  Adding PreToolUse made older
            # ``@anthropic-ai/claude-code`` CLI binaries exit on config parse;
            # per-tool duration for claude-agent is therefore not captured
            # (matches openai-agent / deep-agent).
            options.hooks = {
                "PostToolUse": [HookMatcher(matcher=".*", hooks=[_capture_tool_output])],
            }

            def _flush_tool_outputs() -> None:
                """Patch any pending hook outputs onto the last turn's tool calls."""
                if not trajectory.turns:
                    return
                for tc in trajectory.turns[-1].tool_calls:
                    if tc.id in tool_outputs:
                        tc.output = tool_outputs.pop(tc.id)

            async for message in query(prompt=question, options=options):
                if isinstance(message, AssistantMessage):
                    _flush_tool_outputs()
                    now = time.perf_counter()
                    turn_duration_ms = (now - last_turn_start) * 1000
                    last_turn_start = now
                    text = ""
                    tool_calls: list[ToolCall] = []
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text += block.text
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append(
                                ToolCall(name=block.name, input=block.input, id=block.id)
                            )
                    usage = message.usage or {}
                    trajectory.turns.append(
                        TurnRecord(
                            index=turn_index,
                            text=text,
                            tool_calls=tool_calls,
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                            duration_ms=turn_duration_ms,
                        )
                    )
                    turn_index += 1
                elif isinstance(message, ResultMessage):
                    _flush_tool_outputs()
                    answer = message.result or ""
                    _log.info(
                        "ClaudeAgentRunner: done (stop_reason=%s, turns=%d, "
                        "input_tokens=%d, output_tokens=%d)",
                        message.stop_reason,
                        len(trajectory.turns),
                        trajectory.total_input_tokens,
                        trajectory.total_output_tokens,
                    )

            duration_ms = (time.perf_counter() - run_started) * 1000
            span.set_attribute("agent.answer.length", len(answer))
            span.set_attribute("gen_ai.usage.input_tokens", trajectory.total_input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", trajectory.total_output_tokens)
            span.set_attribute("agent.turns", len(trajectory.turns))
            span.set_attribute("agent.tool_calls", len(trajectory.all_tool_calls))
            span.set_attribute("agent.duration_ms", duration_ms)
            persist_trajectory(
                runner_name="claude-agent",
                model=self._model,
                question=question,
                answer=answer,
                trajectory=trajectory,
            )
            return AgentResult(question=question, answer=answer, trajectory=trajectory)
