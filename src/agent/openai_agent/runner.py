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

import asyncio
import datetime as _dt
import json
import logging
import time
from collections.abc import Collection, Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Self

from agents import (
    Agent,
    OpenAIProvider,
    RunConfig,
    Runner,
)
from agents.mcp import MCPServerStdio, create_static_tool_filter

from llm.routers import resolve_model, resolve_router_creds
from observability import agent_run_span, persist_trajectory

from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, ToolCall, Trajectory, TurnRecord
from ..runner import AgentRunner
from .workspace_tools import WorkspaceToolFactory

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "litellm_proxy/azure/gpt-5.4"
_TOKENROUTER_OPENAI_GPT5_PREFIX = "tokenrouter/openai/gpt-5."

MCPToolAllowlist = Mapping[str, Collection[str]]


def _uses_responses_api(model_id: str) -> bool:
    """Return whether *model_id* should use the OpenAI Responses API."""
    return model_id.startswith(_TOKENROUTER_OPENAI_GPT5_PREFIX)


def _build_permissions(
    *,
    allow_bash: bool = False,
    allow_edit: bool = False,
    allow_web: bool = False,
    allow_files: bool = False,
) -> dict[str, bool]:
    """Build benchmark-safe OpenAI-agent capability permissions.

    MCP access is always available through the separately configured server and
    tool allowlists. Local workspace and web tools are denied unless explicitly
    enabled. Bash also enables workspace edits, matching the OpenCode runner.
    """
    return {
        "mcp": True,
        "files": allow_files,
        "bash": allow_bash,
        "edit": allow_edit or allow_bash,
        "web": allow_web,
    }


def _resolve_run_dir(
    *,
    workspace_dir: Path | str | None,
    permissions: Mapping[str, bool],
) -> Path | None:
    """Resolve the optional workspace required by local file/edit/bash tools."""
    workspace_requested = any(
        permissions[capability] for capability in ("files", "bash", "edit")
    )
    if workspace_requested and workspace_dir is None:
        raise ValueError(
            "workspace_dir is required when enabling files, edits, or bash"
        )
    if workspace_dir is None:
        return None

    run_dir = Path(workspace_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_run_config(model_id: str) -> RunConfig:
    """Build a RunConfig that selects the requested OpenAI API.

    When *model_id* starts with a proxy-router prefix (``litellm_proxy/`` or
    ``tokenrouter/``), configures an :class:`OpenAIProvider` for that router's
    OpenAI-compatible endpoint and credentials.

    ``tokenrouter/openai/gpt-5.*`` models use the Responses API. All other
    router-backed model IDs use Chat Completions. Unprefixed model IDs are
    rejected so this runner never falls back to direct OpenAI credentials.
    """
    creds = resolve_router_creds(model_id)
    if creds is None:
        raise ValueError(
            "OpenAIAgentRunner model IDs must start with "
            "'litellm_proxy/' or 'tokenrouter/'"
        )

    use_responses = _uses_responses_api(model_id)
    provider = OpenAIProvider(
        base_url=creds.base_url,
        api_key=creds.api_key,
        use_responses=use_responses,
    )

    return RunConfig(
        model_provider=provider,
        # Router credentials cannot authenticate with the OpenAI traces API.
        # Keep this run-scoped so other Agents SDK users retain their setting.
        tracing_disabled=True,
        workflow_name="AssetOps Assistant",
    )


def _normalize_mcp_tool_allowlist(
    server_paths: Mapping[str, Path | str],
    mcp_tool_allowlist: MCPToolAllowlist | None,
) -> dict[str, tuple[str, ...]] | None:
    """Validate and copy an optional per-server MCP tool allowlist.

    Supplying any allowlist enables fail-closed mode: configured servers omitted
    from the mapping expose no tools. Unknown servers and invalid tool names are
    rejected so a typo cannot silently widen or misdirect permissions.
    """
    if mcp_tool_allowlist is None:
        return None

    unknown_servers = sorted(set(mcp_tool_allowlist) - set(server_paths))
    if unknown_servers:
        raise ValueError(
            "MCP tool allowlist contains unknown servers: " + ", ".join(unknown_servers)
        )

    normalized: dict[str, tuple[str, ...]] = {}
    for server_name in server_paths:
        tool_names = mcp_tool_allowlist.get(server_name, ())
        if isinstance(tool_names, str):
            raise TypeError(
                f"MCP tool allowlist for {server_name!r} must be a collection "
                "of tool names, not a string"
            )

        invalid_names = [
            tool_name
            for tool_name in tool_names
            if not isinstance(tool_name, str) or not tool_name.strip()
        ]
        if invalid_names:
            raise ValueError(
                f"MCP tool allowlist for {server_name!r} contains invalid tool "
                f"names: {invalid_names!r}"
            )
        normalized[server_name] = tuple(sorted(set(tool_names)))

    return normalized


def _build_mcp_servers(
    server_paths: dict[str, Path | str],
    *,
    mcp_tool_allowlist: MCPToolAllowlist | None = None,
) -> list[MCPServerStdio]:
    """Convert server_paths entries into MCPServerStdio instances.

    Entry-point names (str without path separators) become
    ``MCPServerStdio(command="uv", args=["run", name])``.
    Path objects become ``MCPServerStdio(command="uv", args=["run", str(path)])``.

    The runner exposes MCP tools only. When ``mcp_tool_allowlist`` is provided,
    each server receives an SDK-native static allowlist; omitted servers expose
    no tools. Allowed MCP calls run without interactive approval, matching the
    non-interactive benchmark behavior of the OpenCode runner.
    """
    normalized_allowlist = _normalize_mcp_tool_allowlist(
        server_paths, mcp_tool_allowlist
    )
    servers: list[MCPServerStdio] = []
    for name, spec in server_paths.items():
        if normalized_allowlist is not None and not normalized_allowlist[name]:
            continue
        cmd_arg = str(spec) if isinstance(spec, Path) else spec
        tool_filter = (
            create_static_tool_filter(
                allowed_tool_names=list(normalized_allowlist[name])
            )
            if normalized_allowlist is not None
            else None
        )
        servers.append(
            MCPServerStdio(
                name=name,
                params={
                    "command": "uv",
                    "args": ["run", cmd_arg],
                },
                cache_tools_list=True,
                tool_filter=tool_filter,
                require_approval="never",
            )
        )
    return servers


async def _enter_mcp_servers(
    stack: AsyncExitStack,
    servers: list[MCPServerStdio],
) -> list[MCPServerStdio]:
    """Connect all MCP servers concurrently and register them with *stack*."""
    async with asyncio.TaskGroup() as group:
        tasks = [
            group.create_task(stack.enter_async_context(server)) for server in servers
        ]
    return [task.result() for task in tasks]


def _build_trajectory(result) -> Trajectory:
    """Extract a Trajectory from a Runner.run result.

    Each raw model response becomes exactly one trajectory turn. Tool outputs
    are then joined from ``result.new_items`` by call ID.
    """
    trajectory = Trajectory()

    def _field(value, name: str, default=None):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    tool_calls_by_id: dict[str, ToolCall] = {}
    all_tool_calls: list[ToolCall] = []

    for turn_index, response in enumerate(getattr(result, "raw_responses", []) or []):
        text_parts: list[str] = []
        turn_tool_calls: list[ToolCall] = []

        for raw in _field(response, "output", []) or []:
            raw_type = _field(raw, "type", "")
            if raw_type == "message":
                for part in _field(raw, "content", []) or []:
                    text = _field(part, "text")
                    if text:
                        text_parts.append(text)
            elif raw_type == "function_call":
                tc_name = _field(raw, "name", "") or ""
                tc_id = _field(raw, "call_id", "") or _field(raw, "id", "") or ""
                tc_args = _field(raw, "arguments", "{}") or "{}"
                try:
                    tc_input = (
                        json.loads(tc_args) if isinstance(tc_args, str) else tc_args
                    )
                except (json.JSONDecodeError, TypeError):
                    tc_input = {"raw": tc_args}
                tool_call = ToolCall(name=tc_name, input=tc_input, id=tc_id)
                turn_tool_calls.append(tool_call)
                all_tool_calls.append(tool_call)
                if tc_id:
                    tool_calls_by_id[tc_id] = tool_call

        usage = _field(response, "usage")
        trajectory.turns.append(
            TurnRecord(
                index=turn_index,
                text="".join(text_parts),
                tool_calls=turn_tool_calls,
                input_tokens=_field(usage, "input_tokens", 0) or 0,
                output_tokens=_field(usage, "output_tokens", 0) or 0,
            )
        )

    assigned_calls: set[int] = set()
    for item in getattr(result, "new_items", []) or []:
        if getattr(item, "type", "") == "tool_call_output_item":
            output = getattr(item, "output", None)
            raw = getattr(item, "raw_item", None)
            output_call_id = _field(raw, "call_id", "") if raw else ""
            matching_call = tool_calls_by_id.get(output_call_id)
            if matching_call is None:
                matching_call = next(
                    (call for call in all_tool_calls if id(call) not in assigned_calls),
                    None,
                )
            if matching_call is not None:
                matching_call.output = output
                assigned_calls.add(id(matching_call))

    return trajectory


class OpenAIAgentRunner(AgentRunner):
    """Agent runner that delegates to the OpenAI Agents SDK agentic loop.

    The SDK handles tool discovery, invocation, and multi-turn conversation
    against the registered MCP servers.

    Local file, edit, Bash, and web function tools are denied by default. They
    can be enabled independently for a dedicated workspace. These are ordinary
    function tools so they work with both Responses and Chat Completions models.

    A one-shot :meth:`run` connects and closes MCP servers automatically. For
    repeated calls, use the runner as an async context manager to connect once
    and reuse the active servers until :meth:`aclose`.

    Router-prefixed models use the matching proxy endpoint and credentials.
    ``tokenrouter/openai/gpt-5.*`` uses the Responses API; all other
    router-backed model IDs use Chat Completions. Unprefixed IDs are rejected.

    Args:
        llm: Unused — OpenAIAgentRunner uses the OpenAI Agents SDK directly.
             Accepted for interface compatibility with ``AgentRunner``.
        server_paths: MCP server specs identical to ``PlanExecuteRunner``.
                      Defaults to all registered servers.
        model: Model ID prefixed with ``litellm_proxy/`` or ``tokenrouter/``
               (default: ``litellm_proxy/azure/gpt-5.4``).
        max_turns: Maximum agentic loop turns (default: 30).
        mcp_tool_allowlist: Optional mapping of MCP server names to allowed tool
                            names. Supplying it enables fail-closed filtering:
                            servers omitted from the mapping expose no tools.
        allow_files: Allow workspace file listing, reading, and search tools.
        allow_bash: Allow Bash commands and workspace edits. This is not an OS
                    sandbox; commands can reference host paths explicitly.
        allow_edit: Allow workspace write, replace, and delete tools.
        allow_web: Allow public web search and fetch tools.
        workspace_dir: Dedicated workspace required by files, Bash, or edits.
    """

    def __init__(
        self,
        llm=None,
        server_paths: dict[str, Path | str] | None = None,
        model: str = _DEFAULT_MODEL,
        max_turns: int = 30,
        mcp_tool_allowlist: MCPToolAllowlist | None = None,
        allow_bash: bool = False,
        allow_edit: bool = False,
        allow_web: bool = False,
        allow_files: bool = False,
        workspace_dir: Path | str | None = None,
    ) -> None:
        super().__init__(llm, server_paths)
        self._model_id = model
        self._model = resolve_model(model)
        self._run_config = _build_run_config(model)
        self._max_turns = max_turns
        self._mcp_tool_allowlist = _normalize_mcp_tool_allowlist(
            self._server_paths, mcp_tool_allowlist
        )
        self._permissions = _build_permissions(
            allow_bash=allow_bash,
            allow_edit=allow_edit,
            allow_web=allow_web,
            allow_files=allow_files,
        )
        self._run_dir = _resolve_run_dir(
            workspace_dir=workspace_dir,
            permissions=self._permissions,
        )
        self._local_tools = WorkspaceToolFactory(self._run_dir).build_tools(
            allow_bash=allow_bash,
            allow_edit=allow_edit,
            allow_web=allow_web,
            allow_files=allow_files,
        )
        self._mcp_stack: AsyncExitStack | None = None
        self._active_mcp_servers: list[MCPServerStdio] | None = None
        self._mcp_lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        await self._ensure_persistent_mcp_servers()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _ensure_persistent_mcp_servers(self) -> list[MCPServerStdio]:
        async with self._mcp_lock:
            if self._active_mcp_servers is not None:
                return self._active_mcp_servers

            stack = AsyncExitStack()
            await stack.__aenter__()
            try:
                active_servers = await _enter_mcp_servers(
                    stack,
                    _build_mcp_servers(
                        self._server_paths,
                        mcp_tool_allowlist=self._mcp_tool_allowlist,
                    ),
                )
            except BaseException:
                await stack.aclose()
                raise

            self._mcp_stack = stack
            self._active_mcp_servers = active_servers
            return active_servers

    async def aclose(self) -> None:
        """Close MCP servers opened by the async context manager."""
        async with self._mcp_lock:
            stack = self._mcp_stack
            self._mcp_stack = None
            self._active_mcp_servers = None
            if stack is not None:
                await stack.aclose()

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

            async def _execute(active_servers: list[MCPServerStdio]) -> AgentResult:
                agent = Agent(
                    name="AssetOps Assistant",
                    instructions=AGENT_SYSTEM_PROMPT,
                    tools=self._local_tools,
                    mcp_servers=active_servers,
                    mcp_config={"include_server_in_tool_names": True},
                    model=self._model,
                )

                _log.info(
                    "OpenAIAgentRunner: starting query "
                    "(model=%s, servers=%d, workspace=%s, permissions=%s)",
                    self._model,
                    len(active_servers),
                    self._run_dir or "<disabled>",
                    self._permissions,
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

            if self._active_mcp_servers is not None:
                return await _execute(self._active_mcp_servers)

            # One-shot runs connect concurrently and close on success or error.
            async with AsyncExitStack() as stack:
                active_servers = await _enter_mcp_servers(
                    stack,
                    _build_mcp_servers(
                        self._server_paths,
                        mcp_tool_allowlist=self._mcp_tool_allowlist,
                    ),
                )
                return await _execute(active_servers)
