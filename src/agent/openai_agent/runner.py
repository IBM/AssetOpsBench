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

from openai import AsyncOpenAI

from agents import (
    Agent,
    ModelProvider,
    OpenAIChatCompletionsModel,
    OpenAIProvider,
    RunConfig,
    Runner,
)
from agents.mcp import MCPServerStdio

from observability import agent_run_span, persist_trajectory

from llm.routers import TOKENROUTER_PREFIX, resolve_model, resolve_router_creds
from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, ToolCall, Trajectory, TurnRecord
from ..runner import AgentRunner

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "litellm_proxy/azure/gpt-5.4"


def _build_run_config(model_id: str) -> RunConfig | None:
    """Build a RunConfig with a router-specific model provider when needed.

    When *model_id* starts with a proxy-router prefix (``litellm_proxy/`` or
    ``tokenrouter/``), creates an :class:`AsyncOpenAI` client pointing at that
    router's OpenAI-compatible endpoint using credentials from the router's
    environment variables. All models routed through TokenRouter use the
    Responses API; LiteLLM models use Chat Completions.

    Returns ``None`` for direct OpenAI API usage.
    """
    creds = resolve_router_creds(model_id)
    if creds is None:
        return None

    resolved = resolve_model(model_id)
    client = AsyncOpenAI(base_url=creds.base_url, api_key=creds.api_key)

    if creds.prefix == TOKENROUTER_PREFIX:
        return RunConfig(
            model_provider=OpenAIProvider(
                openai_client=client,
                use_responses=True,
            ),
            tracing_disabled=True,
        )

    class _ChatCompletionsModelProvider(ModelProvider):
        def get_model(self, model_name: str | None):
            return OpenAIChatCompletionsModel(
                model=model_name or resolved,
                openai_client=client,
            )

    return RunConfig(
        model_provider=_ChatCompletionsModelProvider(),
        tracing_disabled=True,
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


def _item_value(item, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _tool_call_id(raw_item) -> str:
    return (_item_value(raw_item, "call_id", "") or _item_value(raw_item, "id", ""))


def _tool_call_from_raw(raw_item) -> ToolCall:
    tc_args = _item_value(raw_item, "arguments", "{}") or "{}"
    try:
        tc_input = json.loads(tc_args) if isinstance(tc_args, str) else tc_args
    except (json.JSONDecodeError, TypeError):
        tc_input = {"raw": tc_args}
    return ToolCall(
        name=_item_value(raw_item, "name", "") or "",
        input=tc_input,
        id=_tool_call_id(raw_item),
    )


def _message_text(raw_item) -> str:
    text_parts: list[str] = []
    for part in _item_value(raw_item, "content", None) or []:
        text = _item_value(part, "text", None)
        if text:
            text_parts.append(text)
    return "".join(text_parts)


def _collect_tool_calls(result) -> dict[str, ToolCall]:
    calls_by_id: dict[str, ToolCall] = {}
    for item in getattr(result, "new_items", []) or []:
        if getattr(item, "type", "") != "tool_call_item":
            continue
        raw_item = getattr(item, "raw_item", None)
        if raw_item is None:
            continue
        tool_call = _tool_call_from_raw(raw_item)
        if tool_call.id:
            calls_by_id[tool_call.id] = tool_call

    for item in getattr(result, "new_items", []) or []:
        if getattr(item, "type", "") != "tool_call_output_item":
            continue
        raw_item = getattr(item, "raw_item", None)
        call_id = _tool_call_id(raw_item)
        if call_id in calls_by_id:
            calls_by_id[call_id].output = getattr(item, "output", None)

    return calls_by_id


def _build_trajectory_from_responses(result, raw_responses) -> Trajectory:
    trajectory = Trajectory()
    calls_by_id = _collect_tool_calls(result)

    for index, response in enumerate(raw_responses):
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for raw_item in getattr(response, "output", None) or []:
            if _item_value(raw_item, "type", "") == "message":
                text_parts.append(_message_text(raw_item))
            call_id = _tool_call_id(raw_item)
            if call_id in calls_by_id:
                tool_calls.append(calls_by_id[call_id])

        usage = getattr(response, "usage", None)
        trajectory.turns.append(
            TurnRecord(
                index=index,
                text="".join(text_parts),
                tool_calls=tool_calls,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            )
        )

    return trajectory


def _build_trajectory_from_items(result, raw_responses) -> Trajectory:
    """Extract a Trajectory from a Runner.run result.

    Fallback for mocked or legacy results whose raw responses do not expose
    their output items.
    """
    trajectory = Trajectory()
    turn_index = 0
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_calls_by_id: dict[str, ToolCall] = {}

    def _flush() -> None:
        nonlocal text_parts, tool_calls, tool_calls_by_id, turn_index
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
        tool_calls_by_id = {}

    for item in result.new_items:
        item_type = getattr(item, "type", "")
        if item_type == "message_output_item":
            # Flush any pending tool calls from previous turn
            _flush()
            raw = getattr(item, "raw_item", None)
            if raw:
                content = getattr(raw, "content", None) or []
                for part in content:
                    if hasattr(part, "text"):
                        text_parts.append(part.text)
        elif item_type == "tool_call_item":
            raw = getattr(item, "raw_item", None)
            if raw:
                tool_call = _tool_call_from_raw(raw)
                tool_calls.append(tool_call)
                if tool_call.id:
                    tool_calls_by_id[tool_call.id] = tool_call
        elif item_type == "tool_call_output_item":
            output = getattr(item, "output", None)
            call_id = _tool_call_id(getattr(item, "raw_item", None))
            matching_call = tool_calls_by_id.get(call_id)
            if matching_call is None:
                matching_call = next(
                    (call for call in reversed(tool_calls) if call.output is None), None
                )
            if matching_call is not None:
                matching_call.output = output

    # Flush remaining
    _flush()

    # Distribute token usage from raw_responses across turns
    for i, resp in enumerate(raw_responses):
        usage = getattr(resp, "usage", None)
        if not usage:
            continue
        if i >= len(trajectory.turns):
            trajectory.turns.append(TurnRecord(index=i, text=""))
        trajectory.turns[i].input_tokens = getattr(usage, "input_tokens", 0) or 0
        trajectory.turns[i].output_tokens = getattr(usage, "output_tokens", 0) or 0

    return trajectory


def _build_trajectory(result) -> Trajectory:
    """Extract text, tool calls, tool outputs, and usage from a run result."""
    raw_responses = getattr(result, "raw_responses", []) or []
    if raw_responses and all(hasattr(response, "output") for response in raw_responses):
        return _build_trajectory_from_responses(result, raw_responses)
    return _build_trajectory_from_items(result, raw_responses)


class OpenAIAgentRunner(AgentRunner):
    """Agent runner that delegates to the OpenAI Agents SDK agentic loop.

    The SDK handles tool discovery, invocation, and multi-turn conversation
    against the registered MCP servers.

    Routes prefixed models through either LiteLLM or TokenRouter using the
    matching ``*_BASE_URL`` / ``*_API_KEY`` environment variables.

    Args:
        llm: Unused — OpenAIAgentRunner uses the OpenAI Agents SDK directly.
             Accepted for interface compatibility with ``AgentRunner``.
        server_paths: MCP server specs identical to ``PlanExecuteRunner``.
                      Defaults to all registered servers.
        model: Direct model ID or a ``litellm_proxy/`` / ``tokenrouter/``
               model ID (default: ``litellm_proxy/azure/gpt-5.4``).
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

                run_kwargs: dict = dict(max_turns=self._max_turns)
                if self._run_config is not None:
                    run_kwargs["run_config"] = self._run_config

                result = await Runner.run(
                    agent,
                    question,
                    **run_kwargs,
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
