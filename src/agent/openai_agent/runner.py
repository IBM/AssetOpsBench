"""AgentRunner implementation backed by the OpenAI Agents SDK.

Each registered MCP server is connected as a stdio MCP server so the OpenAI
agent can call IoT / FMSR / TSFM / utilities tools directly via MCP.

Usage::

    import anyio
    from agent.openai_agent import OpenAIAgentRunner

    runner = OpenAIAgentRunner(model="litellm_proxy/azure/gpt-5.4")
    result = anyio.run(runner.run, "What sensors are on Chiller 6?")
    print(result.answer)
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time
from contextlib import AsyncExitStack
from pathlib import Path

from agents import (
    Agent,
    OpenAIProvider,
    RunConfig,
    Runner,
)
from agents.mcp import MCPServerStdio

from observability import agent_run_span, persist_trajectory

from llm.routers import resolve_model, resolve_router_creds
from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, ToolCall, Trajectory, TurnRecord
from ..runner import AgentRunner

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "litellm_proxy/azure/gpt-5.4"
_TOKENROUTER_OPENAI_GPT5_PREFIX = "tokenrouter/openai/gpt-5."


def _uses_responses_api(model_id: str) -> bool:
    """Return whether *model_id* should use the OpenAI Responses API."""
    return model_id.startswith(_TOKENROUTER_OPENAI_GPT5_PREFIX)


def _build_run_config(model_id: str) -> RunConfig:
    """Build a RunConfig that selects the requested OpenAI API.

    When *model_id* starts with a proxy-router prefix (``litellm_proxy/`` or
    ``tokenrouter/``), configures an :class:`OpenAIProvider` for that router's
    OpenAI-compatible endpoint and credentials.

    ``tokenrouter/openai/gpt-5.*`` models use the Responses API. All other
    model IDs use Chat Completions, including direct OpenAI API usage.
    """
    creds = resolve_router_creds(model_id)
    use_responses = _uses_responses_api(model_id)
    provider = (
        OpenAIProvider(
            base_url=creds.base_url,
            api_key=creds.api_key,
            use_responses=use_responses,
        )
        if creds is not None
        else OpenAIProvider(use_responses=use_responses)
    )

    return RunConfig(
        model_provider=provider,
        # Router credentials cannot authenticate with the OpenAI traces API.
        # Keep this run-scoped so other Agents SDK users retain their setting.
        tracing_disabled=creds is not None,
        workflow_name="AssetOps Assistant",
    )


def _build_mcp_servers(
    server_paths: dict[str, Path | str],
) -> list[MCPServerStdio]:
    """Convert server_paths entries into MCPServerStdio instances.

    Entry-point names (str without path separators) become
    ``MCPServerStdio(command="uv", args=["run", name])``.
    Path objects become ``MCPServerStdio(command="uv", args=["run", str(path)])``.
    """
    servers: list[MCPServerStdio] = []
    for name, spec in server_paths.items():
        cmd_arg = str(spec) if isinstance(spec, Path) else spec
        servers.append(
            MCPServerStdio(
                name=name,
                params={
                    "command": "uv",
                    "args": ["run", cmd_arg],
                },
                cache_tools_list=True,
            )
        )
    return servers


def _build_trajectory(result) -> Trajectory:
    """Extract a Trajectory from a Runner.run result.

    Walks ``result.new_items`` to collect text messages, tool calls, and
    tool outputs.  Token usage is pulled from ``result.raw_responses``.
    """
    trajectory = Trajectory()
    turn_index = 0
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    saw_tool_output = False

    def _flush() -> None:
        nonlocal text_parts, tool_calls, turn_index, saw_tool_output
        if not text_parts and not tool_calls:
            return
        trajectory.turns.append(
            TurnRecord(
                index=turn_index,
                text="".join(text_parts),
                tool_calls=list(tool_calls),
            )
        )
        turn_index += 1
        text_parts = []
        tool_calls = []
        saw_tool_output = False

    def _start_model_item() -> None:
        # Tool outputs separate one model response from the next. This keeps a
        # response containing both preamble text and tool calls in one turn.
        if saw_tool_output:
            _flush()

    def _field(value, name: str, default=None):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    for item in result.new_items:
        item_type = getattr(item, "type", "")
        if item_type == "message_output_item":
            _start_model_item()
            raw = getattr(item, "raw_item", None)
            if raw:
                content = _field(raw, "content", None) or []
                for part in content:
                    text = _field(part, "text")
                    if text:
                        text_parts.append(text)
        elif item_type == "tool_call_item":
            _start_model_item()
            raw = getattr(item, "raw_item", None)
            if raw:
                tc_name = _field(raw, "name", "") or ""
                tc_id = _field(raw, "call_id", "") or _field(raw, "id", "") or ""
                tc_args = _field(raw, "arguments", "{}") or "{}"
                try:
                    tc_input = (
                        json.loads(tc_args) if isinstance(tc_args, str) else tc_args
                    )
                except (json.JSONDecodeError, TypeError):
                    tc_input = {"raw": tc_args}
                tool_calls.append(ToolCall(name=tc_name, input=tc_input, id=tc_id))
        elif item_type == "tool_call_output_item":
            output = getattr(item, "output", None)
            raw = getattr(item, "raw_item", None)
            output_call_id = _field(raw, "call_id", "") if raw else ""
            matching_call = next(
                (call for call in reversed(tool_calls) if call.id == output_call_id),
                None,
            )
            if matching_call is None:
                matching_call = next(
                    (call for call in reversed(tool_calls) if call.output is None),
                    None,
                )
            if matching_call is not None:
                matching_call.output = output
            saw_tool_output = True

    # Flush remaining
    _flush()

    # Distribute token usage from raw_responses across turns
    raw_responses = getattr(result, "raw_responses", []) or []
    while len(trajectory.turns) < len(raw_responses):
        trajectory.turns.append(TurnRecord(index=len(trajectory.turns), text=""))
    for i, resp in enumerate(raw_responses):
        usage = getattr(resp, "usage", None)
        if usage:
            trajectory.turns[i].input_tokens = getattr(usage, "input_tokens", 0) or 0
            trajectory.turns[i].output_tokens = getattr(usage, "output_tokens", 0) or 0

    return trajectory


class OpenAIAgentRunner(AgentRunner):
    """Agent runner that delegates to the OpenAI Agents SDK agentic loop.

    The SDK handles tool discovery, invocation, and multi-turn conversation
    against the registered MCP servers.

    Router-prefixed models use the matching proxy endpoint and credentials.
    ``tokenrouter/openai/gpt-5.*`` uses the Responses API; all other model IDs
    use Chat Completions.

    Args:
        llm: Unused — OpenAIAgentRunner uses the OpenAI Agents SDK directly.
             Accepted for interface compatibility with ``AgentRunner``.
        server_paths: MCP server specs identical to ``PlanExecuteRunner``.
                      Defaults to all registered servers.
        model: Model ID, optionally prefixed with ``litellm_proxy/`` or
               ``tokenrouter/`` (default: ``litellm_proxy/azure/gpt-5.4``).
        max_turns: Maximum agentic loop turns (default: 30).
    """

    def __init__(
        self,
        llm=None,
        server_paths: dict[str, Path | str] | None = None,
        model: str = _DEFAULT_MODEL,
        max_turns: int = 30,
    ) -> None:
        super().__init__(llm, server_paths)
        self._model_id = model
        self._model = resolve_model(model)
        self._run_config = _build_run_config(model)
        self._max_turns = max_turns

    async def run(self, question: str) -> AgentResult:
        """Run the OpenAI Agents SDK loop for *question*.

        Args:
            question: Natural-language question to answer.

        Returns:
            AgentResult with the final answer and full execution trajectory.
        """
        with agent_run_span(
            "openai-agent", model=self._model_id, question=question
        ) as span:
            run_started = time.perf_counter()
            started_at = _dt.datetime.now(_dt.UTC).isoformat()
            mcp_servers = _build_mcp_servers(self._server_paths)

            # AsyncExitStack enters every server and closes them in LIFO order
            # on exit (success or exception).
            async with AsyncExitStack() as stack:
                active_servers = [
                    await stack.enter_async_context(s) for s in mcp_servers
                ]
                agent = Agent(
                    name="AssetOps Assistant",
                    instructions=AGENT_SYSTEM_PROMPT,
                    mcp_servers=active_servers,
                    model=self._model,
                )

                _log.info(
                    "OpenAIAgentRunner: starting query (model=%s, servers=%d)",
                    self._model,
                    len(active_servers),
                )

                result = await Runner.run(
                    agent,
                    question,
                    max_turns=self._max_turns,
                    run_config=self._run_config,
                )

                answer = result.final_output or ""
                trajectory = _build_trajectory(result)
                trajectory.started_at = started_at

                _log.info(
                    "OpenAIAgentRunner: done (turns=%d, input_tokens=%d, "
                    "output_tokens=%d)",
                    len(trajectory.turns),
                    trajectory.total_input_tokens,
                    trajectory.total_output_tokens,
                )

                span.set_attribute("agent.answer.length", len(answer))
                span.set_attribute(
                    "gen_ai.usage.input_tokens", trajectory.total_input_tokens
                )
                span.set_attribute(
                    "gen_ai.usage.output_tokens", trajectory.total_output_tokens
                )
                span.set_attribute("agent.turns", len(trajectory.turns))
                span.set_attribute("agent.tool_calls", len(trajectory.all_tool_calls))
                span.set_attribute(
                    "agent.duration_ms", (time.perf_counter() - run_started) * 1000
                )
                persist_trajectory(
                    runner_name="openai-agent",
                    model=self._model_id,
                    question=question,
                    answer=answer,
                    trajectory=trajectory,
                )
                return AgentResult(
                    question=question,
                    answer=answer,
                    trajectory=trajectory,
                )
