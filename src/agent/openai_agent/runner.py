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
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

from openai import AsyncOpenAI

from agents import (
    Agent,
    ModelProvider,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStdio

from observability import agent_run_span, persist_trajectory
from observability.benchmark_trace import emit, instrument_model, plain, tool_result_error
import uuid

from llm.routers import resolve_model, resolve_router_creds
from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, ToolCall, Trajectory, TurnRecord
from ..runner import AgentRunner

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "litellm_proxy/azure/gpt-5.4"


def _build_run_config(model_id: str) -> RunConfig | None:
    """Build a RunConfig with a LiteLLM model provider when needed.

    When *model_id* starts with a proxy-router prefix (``litellm_proxy/`` or
    ``tokenrouter/``), creates an :class:`AsyncOpenAI` client pointing at that
    router's OpenAI-compatible endpoint (credentials from the router's env
    vars) and wraps it in :class:`OpenAIChatCompletionsModel`.

    Returns ``None`` for direct OpenAI API usage.
    """
    creds = resolve_router_creds(model_id)
    if creds is None:
        return None

    resolved = resolve_model(model_id)
    client = AsyncOpenAI(base_url=creds.base_url, api_key=creds.api_key)
    set_tracing_disabled(disabled=True)

    class _LiteLLMModelProvider(ModelProvider):
        def get_model(self, model_name: str | None):
            return instrument_model(OpenAIChatCompletionsModel(
                model=model_name or resolved,
                openai_client=client,
            ))

    settings = None
    effort = os.environ.get("GLM_REASONING_EFFORT") if model_id.startswith("zai/") else None
    if effort:
        if effort not in {"low", "high", "max"}:
            raise ValueError("GLM_REASONING_EFFORT must be low, high or max")
        settings = ModelSettings(extra_body={"thinking": {"type": "enabled"},
                                             "reasoning_effort": effort})
    return RunConfig(model_provider=_LiteLLMModelProvider(), model_settings=settings)


class MeasuredMCPServer(MCPServerStdio):
    async def call_tool(self, tool_name, arguments, meta=None):
        identifier = str(uuid.uuid4())
        start = time.perf_counter()
        emit('tool_start', id=identifier, name=tool_name, server=self.name, arguments=arguments)
        try:
            output = await super().call_tool(tool_name, arguments, meta)
        except BaseException as error:
            emit('tool_end', id=identifier, name=tool_name, server=self.name, arguments=arguments,
                 duration_ms=(time.perf_counter()-start)*1000,
                 error={'type': type(error).__name__, 'message': str(error)})
            raise
        emit('tool_end', id=identifier, name=tool_name, server=self.name, arguments=arguments,
             duration_ms=(time.perf_counter()-start)*1000, output=output,
             error=tool_result_error(output))
        return output


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
            MeasuredMCPServer(
                name=name,
                params={
                    "command": "uv",
                    "args": ["run", cmd_arg],
                    "env": {key: os.environ[key] for key in (
                        "COUCHDB_URL", "COUCHDB_USERNAME", "COUCHDB_PASSWORD"
                    ) if key in os.environ},
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
    calls_by_id = {}

    def _flush() -> None:
        nonlocal text_parts, tool_calls, turn_index
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
                tc_name = getattr(raw, "name", "") or ""
                tc_id = getattr(raw, "call_id", "") or getattr(raw, "id", "") or ""
                tc_args = getattr(raw, "arguments", "{}") or "{}"
                try:
                    tc_input = (
                        json.loads(tc_args) if isinstance(tc_args, str) else tc_args
                    )
                except (json.JSONDecodeError, TypeError):
                    tc_input = {"raw": tc_args}
                call = ToolCall(name=tc_name, input=tc_input, id=tc_id)
                tool_calls.append(call)
                if tc_id: calls_by_id[tc_id] = call
        elif item_type == "tool_call_output_item":
            output = getattr(item, "output", None)
            raw = getattr(item, 'raw_item', None)
            identifier = raw.get('call_id') if isinstance(raw,dict) else getattr(raw,'call_id',None)
            matching = calls_by_id.get(identifier) if identifier else (tool_calls[-1] if tool_calls else None)
            if matching is not None:
                matching.output = output

    # Flush remaining
    _flush()

    # Distribute token usage from raw_responses across turns
    raw_responses = getattr(result, "raw_responses", []) or []
    for i, resp in enumerate(raw_responses):
        usage = getattr(resp, "usage", None)
        if usage and i < len(trajectory.turns):
            trajectory.turns[i].input_tokens = getattr(usage, "input_tokens", 0) or 0
            trajectory.turns[i].output_tokens = getattr(usage, "output_tokens", 0) or 0

    return trajectory


class OpenAIAgentRunner(AgentRunner):
    """Agent runner that delegates to the OpenAI Agents SDK agentic loop.

    The SDK handles tool discovery, invocation, and multi-turn conversation
    against the registered MCP servers.

    Routes all requests through a LiteLLM proxy via the ``litellm_proxy/``
    proxy-router prefix ``litellm_proxy/`` or ``tokenrouter/`` (requires the
    matching ``*_BASE_URL`` / ``*_API_KEY`` env vars).

    Args:
        llm: Unused — OpenAIAgentRunner uses the OpenAI Agents SDK directly.
             Accepted for interface compatibility with ``AgentRunner``.
        server_paths: MCP server specs identical to ``PlanExecuteRunner``.
                      Defaults to all registered servers.
        model: LiteLLM model string with ``litellm_proxy/`` prefix
               (default: ``litellm_proxy/azure/gpt-5.4``).
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
            emit('capabilities', tool_events=True, request_timings=True, compaction_events=False)
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

                emit('sdk_result', payload={'items':[{'type':i.type, 'raw_item':plain(getattr(i,'raw_item',None)), 'output':plain(getattr(i,'output',None))} for i in result.new_items], 'responses':result.raw_responses})
                emit('provider_summary', payload={'usage':plain(getattr(getattr(result,'context_wrapper',None),'usage',None))})
                answer = result.final_output or ""
                emit("final_answer", answer=answer)
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
