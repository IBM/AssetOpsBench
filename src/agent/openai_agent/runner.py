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

import json
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI

from agents import Agent, ModelProvider, OpenAIChatCompletionsModel, RunConfig, Runner, set_tracing_disabled
from agents.mcp import MCPServerStdio

from ..models import AgentResult
from ..plan_execute.executor import DEFAULT_SERVER_PATHS
from ..runner import AgentRunner
from .models import ToolCall, Trajectory, TurnRecord

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "litellm_proxy/azure/gpt-5.4"
_LITELLM_PREFIX = "litellm_proxy/"

def _resolve_model(model_id: str) -> str:
    """Strip the ``litellm_proxy/`` prefix from a model ID.

    Examples::

        "litellm_proxy/Azure/gpt-5-2025-08-07"  ->  "Azure/gpt-5-2025-08-07"
        "gpt-4o"                                 ->  "gpt-4o"
    """
    if model_id.startswith(_LITELLM_PREFIX):
        return model_id[len(_LITELLM_PREFIX):]
    return model_id


def _build_run_config(model_id: str) -> RunConfig | None:
    """Build a RunConfig with a LiteLLM model provider when needed.

    When *model_id* starts with ``litellm_proxy/``, creates an
    :class:`AsyncOpenAI` client pointing at the LiteLLM proxy (using
    ``LITELLM_BASE_URL`` and ``LITELLM_API_KEY``) and wraps it in
    :class:`OpenAIChatCompletionsModel`.

    Returns ``None`` for direct OpenAI API usage.
    """
    if not model_id.startswith(_LITELLM_PREFIX):
        return None

    base_url = os.environ.get("LITELLM_BASE_URL")
    api_key = os.environ.get("LITELLM_API_KEY")
    if not base_url or not api_key:
        raise ValueError(
            "LITELLM_BASE_URL and LITELLM_API_KEY must be set "
            f"when using {_LITELLM_PREFIX!r} model prefix"
        )

    resolved = _resolve_model(model_id)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    set_tracing_disabled(disabled=True)

    class _LiteLLMModelProvider(ModelProvider):
        def get_model(self, model_name: str | None):
            return OpenAIChatCompletionsModel(
                model=model_name or resolved,
                openai_client=client,
            )

    return RunConfig(model_provider=_LiteLLMModelProvider())


_SYSTEM_PROMPT = """\
You are an industrial asset operations assistant with access to MCP tools for
querying IoT sensor data, failure mode and symptom records, time-series
forecasting models, and work order management.

Answer the user's question concisely and accurately using the available tools.
When you retrieve data, include the key numbers or names in your answer.
"""


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
                    tc_input = json.loads(tc_args) if isinstance(tc_args, str) else tc_args
                except (json.JSONDecodeError, TypeError):
                    tc_input = {"raw": tc_args}
                tool_calls.append(ToolCall(name=tc_name, input=tc_input, id=tc_id))
        elif item_type == "tool_call_output_item":
            output = getattr(item, "output", None)
            # Attach output to the last matching tool call
            if tool_calls:
                tool_calls[-1].output = output

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
    model ID prefix (requires ``LITELLM_BASE_URL`` and ``LITELLM_API_KEY``).

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
        self._model = _resolve_model(model)
        self._run_config = _build_run_config(model)
        self._max_turns = max_turns
        self._resolved_server_paths: dict[str, Path | str] = (
            server_paths if server_paths is not None else dict(DEFAULT_SERVER_PATHS)
        )

    async def run(self, question: str) -> AgentResult:
        """Run the OpenAI Agents SDK loop for *question*.

        Args:
            question: Natural-language question to answer.

        Returns:
            AgentResult with the final answer and full execution trajectory.
        """
        mcp_servers = _build_mcp_servers(self._resolved_server_paths)

        # Use async context managers to manage MCP server lifecycle
        async with _managed_servers(mcp_servers) as active_servers:
            agent = Agent(
                name="AssetOps Assistant",
                instructions=_SYSTEM_PROMPT,
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

            _log.info(
                "OpenAIAgentRunner: done (turns=%d, input_tokens=%d, "
                "output_tokens=%d)",
                len(trajectory.turns),
                trajectory.total_input_tokens,
                trajectory.total_output_tokens,
            )

            return AgentResult(
                question=question,
                answer=answer,
                trajectory=trajectory,
            )


class _managed_servers:
    """Async context manager that enters all MCP server contexts."""

    def __init__(self, servers: list[MCPServerStdio]) -> None:
        self._servers = servers
        self._entered: list[MCPServerStdio] = []

    async def __aenter__(self) -> list[MCPServerStdio]:
        for server in self._servers:
            await server.__aenter__()
            self._entered.append(server)
        return self._entered

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        for server in reversed(self._entered):
            try:
                await server.__aexit__(exc_type, exc_val, exc_tb)
            except Exception:
                _log.warning("Failed to close MCP server %s", server.name, exc_info=True)
