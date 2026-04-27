"""AgentRunner implementation backed by the LangChain deep-agents framework.

Each registered MCP server is connected via ``langchain-mcp-adapters`` so its
tools are exposed to the deep agent as native LangChain tools.  The deep agent
handles planning, tool invocation, and multi-turn conversation.

Usage::

    import anyio
    from agent.deep_agent import DeepAgentRunner

    runner = DeepAgentRunner()
    result = anyio.run(runner.run, "What sensors are on Chiller 6?")
    print(result.answer)
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import time
from functools import cached_property
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from observability import agent_run_span, persist_trajectory

from .._litellm import LITELLM_PREFIX, resolve_model
from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, ToolCall, Trajectory, TurnRecord
from ..runner import AgentRunner

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent

_DEFAULT_MODEL = "litellm_proxy/aws/claude-opus-4-6"


def _build_chat_model(model_id: str):
    """Construct a LangChain chat model for *model_id*.

    When the ID uses the ``litellm_proxy/`` prefix, a :class:`ChatOpenAI`
    instance is pointed at the LiteLLM proxy (using ``LITELLM_BASE_URL`` and
    ``LITELLM_API_KEY``).  Otherwise the model string is passed to
    ``init_chat_model`` so any provider supported by LangChain can be used.
    """
    if model_id.startswith(LITELLM_PREFIX):
        base_url = os.environ.get("LITELLM_BASE_URL")
        api_key = os.environ.get("LITELLM_API_KEY")
        if not base_url or not api_key:
            raise ValueError(
                "LITELLM_BASE_URL and LITELLM_API_KEY must be set "
                f"when using {LITELLM_PREFIX!r} model prefix"
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=resolve_model(model_id),
            base_url=base_url,
            api_key=api_key,
        )

    from langchain.chat_models import init_chat_model

    return init_chat_model(model_id)


def _build_mcp_connections(
    server_paths: dict[str, Path | str],
) -> dict[str, dict]:
    """Convert ``server_paths`` entries into ``MultiServerMCPClient`` specs.

    Entry-point names (``str``) become ``{"command": "uv", "args": ["run", name]}``.
    ``Path`` objects become ``{"command": "uv", "args": ["run", str(path)]}``.
    Both run with ``cwd`` set to the repo root so ``uv run`` resolves scripts.
    """
    connections: dict[str, dict] = {}
    for name, spec in server_paths.items():
        cmd_arg = str(spec) if isinstance(spec, Path) else spec
        connections[name] = {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", cmd_arg],
            "cwd": str(_REPO_ROOT),
        }
    return connections


def _build_trajectory(messages) -> Trajectory:
    """Extract a :class:`Trajectory` from the deep-agent message list.

    Walks through ``AIMessage`` / ``ToolMessage`` entries in order, grouping
    each ``AIMessage`` into its own turn and attaching subsequent
    ``ToolMessage`` outputs to the matching tool call by ``tool_call_id``.
    """
    trajectory = Trajectory()
    turn_index = 0
    tool_call_index: dict[str, ToolCall] = {}

    for msg in messages:
        if isinstance(msg, AIMessage):
            text = msg.content if isinstance(msg.content, str) else ""
            if not text and isinstance(msg.content, list):
                parts: list[str] = []
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        parts.append(part)
                text = "".join(parts)

            tool_calls: list[ToolCall] = []
            for tc in msg.tool_calls or []:
                call = ToolCall(
                    name=tc.get("name", ""),
                    input=tc.get("args", {}) or {},
                    id=tc.get("id", "") or "",
                )
                tool_calls.append(call)
                if call.id:
                    tool_call_index[call.id] = call

            usage = msg.usage_metadata or {}
            trajectory.turns.append(
                TurnRecord(
                    index=turn_index,
                    text=text,
                    tool_calls=tool_calls,
                    input_tokens=int(usage.get("input_tokens", 0) or 0),
                    output_tokens=int(usage.get("output_tokens", 0) or 0),
                )
            )
            turn_index += 1
        elif isinstance(msg, ToolMessage):
            call = tool_call_index.get(msg.tool_call_id)
            if call is not None:
                call.output = msg.content

    return trajectory


class DeepAgentRunner(AgentRunner):
    """Agent runner that delegates to LangChain's deep-agents framework.

    The deep agent handles planning, tool invocation, and multi-turn
    conversation against the MCP-bridged LangChain tools.

    Routes LLM calls through a LiteLLM proxy when the model ID uses the
    ``litellm_proxy/`` prefix (requires ``LITELLM_BASE_URL`` and
    ``LITELLM_API_KEY``).

    Args:
        llm: Unused — DeepAgentRunner uses the deep-agents framework directly.
             Accepted for interface compatibility with ``AgentRunner``.
        server_paths: MCP server specs identical to ``PlanExecuteRunner``.
                      Defaults to all registered servers.
        model: LiteLLM-prefixed or native provider model string
               (default: ``litellm_proxy/aws/claude-opus-4-6``).
        recursion_limit: Maximum graph recursion steps (default: 100).
    """

    def __init__(
        self,
        llm=None,
        server_paths: dict[str, Path | str] | None = None,
        model: str = _DEFAULT_MODEL,
        recursion_limit: int = 100,
    ) -> None:
        super().__init__(llm, server_paths)
        self._model_id = model
        self._recursion_limit = recursion_limit

    @cached_property
    def _chat_model(self):
        """LangChain chat model, built once per runner instance."""
        return _build_chat_model(self._model_id)

    async def run(self, question: str) -> AgentResult:
        """Run the deep-agents loop for *question*.

        Args:
            question: Natural-language question to answer.

        Returns:
            :class:`AgentResult` with the final answer and full trajectory.
        """
        with agent_run_span(
            "deep-agent", model=self._model_id, question=question
        ) as span:
            run_started = time.perf_counter()
            started_at = _dt.datetime.now(_dt.UTC).isoformat()
            from deepagents import create_deep_agent
            from langchain_mcp_adapters.client import MultiServerMCPClient

            connections = _build_mcp_connections(self._server_paths)
            client = MultiServerMCPClient(connections) if connections else None
            tools = await client.get_tools() if client is not None else []

            agent = create_deep_agent(
                model=self._chat_model,
                tools=tools,
                system_prompt=AGENT_SYSTEM_PROMPT,
            )

            _log.info(
                "DeepAgentRunner: starting query (model=%s, tools=%d)",
                self._model_id,
                len(tools),
            )

            state = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": self._recursion_limit},
            )

            messages = state.get("messages", []) if isinstance(state, dict) else []
            trajectory = _build_trajectory(messages)
            trajectory.started_at = started_at

            answer = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    if isinstance(msg.content, str) and msg.content.strip():
                        answer = msg.content
                        break
                    if isinstance(msg.content, list):
                        parts = [
                            p.get("text", "")
                            for p in msg.content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        joined = "".join(parts).strip()
                        if joined:
                            answer = joined
                            break

            _log.info(
                "DeepAgentRunner: done (turns=%d, input_tokens=%d, output_tokens=%d)",
                len(trajectory.turns),
                trajectory.total_input_tokens,
                trajectory.total_output_tokens,
            )

            span.set_attribute("agent.answer.length", len(answer))
            span.set_attribute("gen_ai.usage.input_tokens", trajectory.total_input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", trajectory.total_output_tokens)
            span.set_attribute("agent.turns", len(trajectory.turns))
            span.set_attribute("agent.tool_calls", len(trajectory.all_tool_calls))
            span.set_attribute(
                "agent.duration_ms", (time.perf_counter() - run_started) * 1000
            )
            persist_trajectory(
                runner_name="deep-agent",
                model=self._model_id,
                question=question,
                answer=answer,
                trajectory=trajectory,
            )
            return AgentResult(question=question, answer=answer, trajectory=trajectory)
