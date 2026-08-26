"""AgentRunner backed by Artificial Analysis' Stirrup framework.

Stirrup is an in-process Python library (not a separate binary), so this
runner is structurally the same as :class:`~agent.deep_agent.DeepAgentRunner`:
build a client, attach the AssetOpsBench MCP servers as a tool provider, run
the agent loop, and map the returned message history onto the shared
:class:`~agent.models.Trajectory`.

Model routing:
  * OpenAI-compatible router prefixes -> Stirrup ``ChatCompletionsClient``.
  * ``<provider>/<model>``     -> Stirrup ``LiteLLMClient``, which reaches
    Anthropic, watsonx, Bedrock, etc. natively through LiteLLM.  This means
    ``watsonx/...`` models work directly here, without the proxy detour Goose
    needed.

Tracks (the code switch):
  * ``code_enabled=False`` -> tools are *only* the MCP servers; directly
    comparable to claude-agent / openai-agent / deep-agent.
  * ``code_enabled=True``  -> a sandboxed code-execution tool (Docker by
    default) is added, so the agent may solve a scenario by writing code.
    Report on its own leaderboard track; the bypass metric records whether it
    did so instead of calling the domain tools.

Topology (the tool-surface switch):
  * ``topology="flat"``    -> every MCP server is attached to the root agent, so
    all domain tool schemas sit in the root context on every turn. The default,
    and the shape the other runners use.
  * ``topology="gateway"`` -> every server sits behind three routing tools
    (``search_tools`` / ``describe_tools`` / ``call_tool``), so the root carries
    a handful of entries instead of the full manifest while keeping one context
    and one trajectory. See :mod:`.gateway`. Works on both tracks: on the code
    track the gateway wraps the workspace-bridged provider, so oversized results
    still spill into ``mcp_results/``.

Stirrup's web/default tools are deliberately NOT attached: the environment
under test is the MCP servers (plus, on the code track, code execution), so
adding web search would contaminate the benchmark.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from observability import agent_run_span, persist_trajectory

from llm.routers import resolve_model, resolve_router_creds
from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, Trajectory
from ..runner import AgentRunner
from .finish_tool import ASSETOPS_FINISH_TOOL
from .gateway import (
    DEFAULT_TOP_K,
    GATEWAY_DISCOVERY_TOOLS,
    GATEWAY_MODES,
    MCPGatewayToolProvider,
)
from .trajectory import build_trajectory, classify_tool, final_answer
from .handoff_tools import build_handoff_tools

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
# A code-track image needs the scientific stack the WO/vibration analyses use.
_DEFAULT_CODE_IMAGE = os.environ.get("STIRRUP_CODE_IMAGE", "assetops-code")
TOPOLOGIES = ("flat", "gateway")

# Stirrup 0.2.0 takes the working-context budget as an explicit client argument
# (``context_window_tokens``) instead of reading ``max_tokens`` for both the
# provider output cap and the summarization trigger, so the adapter this module
# used to carry is gone. ``max_tokens`` must stay <= ``context_window_tokens``;
# the client constructor validates the pair.
_ROOT_CONTEXT_WINDOW_TOKENS = 100_000
_ROOT_MAX_OUTPUT_TOKENS = 64_000
_CONTEXT_SUMMARIZATION_CUTOFF = 0.75
_CODE_EXEC_SYSTEM_PROMPT = """\
Code execution:
- MCP tools and their definitions are authoritative for domain data and semantics.
  Never use code to query backing services or bypass an available MCP tool.
- Do not overuse code_exec. Answer directly from MCP results, domain knowledge,
  and basic reasoning or arithmetic when sufficient. Use code_exec only for
  necessary computation, data processing, workspace inspection, or validation.
  Never use it for planning, comments, placeholders, or empty scripts.
- Prefer one complete script that inspects, analyzes, and verifies. Do not repeat
  equivalent experiments; correct failures directly.
- Stay inside the execution workspace and use relative paths. Workspace state
  persists across code_exec calls.
- For artifacts, inspect only the schema, counts, a small sample, or the specific
  rows or fields needed, then process in place. If an artifact exceeds 200 KiB,
  never print it in full; extract and process the relevant subset in bounded
  batches. Avoid large record lists and verbose diagnostics. Reuse snapshots
  unless domain state has changed.
"""
_DOCKER_CODE_EXEC_SYSTEM_PROMPT = """\
The Docker execution workspace is /workspace. Host filesystem paths are not
available inside the container. NumPy, pandas, and SciPy are installed; check
availability before using other packages.
"""
_GATEWAY_SYSTEM_PROMPT = """\
Tool routing:
- Your domain tools are behind a gateway. You do not see them until you ask.
- search_tools(query) ranks the catalogue against what you are doing now.
  describe_tools(names) returns full parameter schemas. call_tool(name,
  arguments) runs one.
- Describe a tool once, then call it as often as you need; do not re-describe a
  schema you already have in this conversation.
- If call_tool rejects your arguments it returns the real error and the schema.
  Correct them from that rather than guessing or searching again.
"""
_LOCAL_CODE_EXEC_SYSTEM_PROMPT = """\
The local execution workspace is a temporary directory, but commands run on the
host with the current user's permissions. Keep all reads and writes inside the
workspace and use relative paths.
"""


def _build_full_summary_logger():
    """Return a Stirrup logger that displays generated summaries without truncation."""
    from rich.text import Text
    from stirrup.utils.logging import AgentLogger, console

    class _FullSummaryLogger(AgentLogger):
        def context_summarization_complete(
            self, summary: str, bridge: str
        ) -> None:
            console.print(Text("✓ Summary Generated", style="bold green"))
            console.print(summary, markup=False, soft_wrap=True)

    return _FullSummaryLogger()


def _copy_workspace_contents(source: Path, destination: Path) -> None:
    """Copy code-exec workspace contents out of Stirrup's temp child dir."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _preserving_provider_class(provider_cls):
    class _PreservingCodeExecToolProvider(provider_cls):
        def __init__(self, *args, preserve_dir: Path, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._assetops_preserve_dir = preserve_dir

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            temp_dir = self.temp_dir
            if temp_dir is not None and temp_dir.exists():
                try:
                    fix_ownership = getattr(self, "_fix_file_ownership", None)
                    if fix_ownership is not None:
                        await fix_ownership()
                    _copy_workspace_contents(temp_dir, self._assetops_preserve_dir)
                    _log.info(
                        "Preserved Stirrup code workspace %s -> %s",
                        temp_dir,
                        self._assetops_preserve_dir,
                    )
                except Exception:
                    _log.warning(
                        "Failed to preserve Stirrup code workspace %s",
                        temp_dir,
                        exc_info=True,
                    )
            return await super().__aexit__(exc_type, exc_val, exc_tb)

    return _PreservingCodeExecToolProvider


class StirrupAgentRunner(AgentRunner):
    """Run a question through a Stirrup agent against the MCP servers.

    Args:
        llm: Unused; accepted for :class:`AgentRunner` interface parity.
        server_paths: MCP server specs (defaults to all registered servers).
        model: ``litellm_proxy/<provider>/<model>`` or native ``<provider>/<model>``.
        code_enabled: Add a sandboxed code-execution tool (the code track).
        code_backend: ``"docker"`` (sandboxed, default) or ``"local"``.
        topology: ``"flat"`` (all servers on the root) or ``"gateway"`` (all
            servers behind routing tools, single context).
        gateway_mode: ``"index"`` pins a compact catalogue into the root
            context and defers only schemas; ``"search"`` withholds the
            catalogue too. Ignored unless ``topology="gateway"``.
        gateway_top_k: Default candidates returned by ``search_tools``.
        workspace_dir: Optional host base directory for Docker/local code execution.
        preserve_workspace: Copy final code-execution files back into ``workspace_dir``.
        max_turns: Stirrup agent loop bound.
        temperature: Optional sampling temperature passed to the Stirrup client.
        reasoning_effort: Optional reasoning effort passed to the Stirrup client.
    """

    def __init__(
        self,
        llm=None,
        server_paths=None,
        model: str = _DEFAULT_MODEL,
        code_enabled: bool = True,
        code_backend: str = "docker",
        topology: str = "flat",
        gateway_mode: str = "index",
        gateway_top_k: int = DEFAULT_TOP_K,
        workspace_dir: Path | str | None = None,
        preserve_workspace: bool = False,
        max_turns: int = 30,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(llm, server_paths)
        if code_backend not in {"docker", "local"}:
            raise ValueError("code_backend must be 'docker' or 'local'")
        if topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {TOPOLOGIES}")
        if gateway_mode not in GATEWAY_MODES:
            raise ValueError(f"gateway_mode must be one of {GATEWAY_MODES}")
        self._model_id = model
        self._code_enabled = code_enabled
        self._code_backend = code_backend
        self._topology = topology
        self._gateway_mode = gateway_mode
        self._gateway_top_k = gateway_top_k
        self._gateway: MCPGatewayToolProvider | None = None
        self._workspace_dir = (
            Path(workspace_dir).expanduser().resolve()
            if workspace_dir is not None
            else None
        )
        if preserve_workspace and self._workspace_dir is None:
            raise ValueError("workspace_dir is required when preserve_workspace is enabled")
        if preserve_workspace and code_backend not in {"docker", "local"}:
            raise ValueError(
                "preserve_workspace is only supported with docker or local code backends"
            )
        self._preserve_workspace = preserve_workspace
        self._max_turns = max_turns
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort

    # -- client / tools ----------------------------------------------------

    def _build_client(
        self,
        *,
        context_window_tokens: int = _ROOT_CONTEXT_WINDOW_TOKENS,
        max_output_tokens: int = _ROOT_MAX_OUTPUT_TOKENS,
    ):
        """Build a Stirrup LLM client for the configured model id.

        ``context_window_tokens`` is the working-context budget the agent loop
        uses to decide when to summarize; it is intentionally lower than the
        provider's real window so long runs compact early.
        """
        client_kwargs = (
            {"temperature": self._temperature}
            if self._temperature is not None
            else None
        )

        creds = resolve_router_creds(self._model_id)
        if creds is not None:
            from stirrup.clients.chat_completions_client import ChatCompletionsClient

            return ChatCompletionsClient(
                model=resolve_model(self._model_id),
                max_tokens=max_output_tokens,
                context_window_tokens=context_window_tokens,
                base_url=creds.base_url.rstrip("/"),
                api_key=creds.api_key,
                reasoning_effort=self._reasoning_effort,
                kwargs=client_kwargs,
            )

        from stirrup.clients.litellm_client import LiteLLMClient

        return LiteLLMClient(
            model=self._model_id,
            max_tokens=max_output_tokens,
            context_window_tokens=context_window_tokens,
            reasoning_effort=self._reasoning_effort,
            kwargs=client_kwargs,
        )

    def _build_mcp_config(self):
        """Build the Stirrup MCP configuration for AssetOpsBench servers.

        Each server is a stdio process launched exactly as the other runners
        launch it: ``uv run --directory <repo> <entry-point>``.
        """
        from stirrup.tools.mcp import MCPConfig

        servers: dict[str, dict] = {}
        for name, spec in self._server_paths.items():
            cmd_arg = str(spec)
            servers[name] = {
                "command": "uv",
                "args": ["run", "--directory", str(_REPO_ROOT), cmd_arg],
                "cwd": str(_REPO_ROOT),
            }
        return MCPConfig.model_validate({"mcpServers": servers})

    def _build_mcp_provider(self, *, exec_env=None, server_names: list[str] | None = None):
        """Build the MCP provider, bridging large results when code is enabled."""
        config = self._build_mcp_config()
        if exec_env is None:
            from stirrup.tools.mcp import MCPToolProvider

            return MCPToolProvider(config=config, server_names=server_names)

        from .workspace_bridge import WorkspaceBridgedMCPToolProvider

        return WorkspaceBridgedMCPToolProvider(
            config=config,
            exec_env=exec_env,
            server_names=server_names,
        )

    def _wrap_in_gateway(self, provider):
        """Put every MCP server behind the routing gateway.

        The gateway wraps the provider the track already built, so on the code
        track it wraps the workspace-bridged provider and oversized results keep
        spilling into ``mcp_results/`` exactly as they do under ``flat``.
        """
        self._gateway = MCPGatewayToolProvider(
            provider, mode=self._gateway_mode, top_k=self._gateway_top_k
        )
        return self._gateway

    def _build_code_provider(self):
        """Build the sandboxed code-execution provider for the code track."""
        if self._code_backend == "local":
            from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

            provider_cls = LocalCodeExecToolProvider
            kwargs = {"temp_base_dir": self._workspace_dir}
            if self._preserve_workspace:
                provider_cls = _preserving_provider_class(provider_cls)
                kwargs["preserve_dir"] = self._workspace_dir
            return provider_cls(**kwargs)
        from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider

        if self._preserve_workspace:
            provider_cls = _preserving_provider_class(DockerCodeExecToolProvider)
            return provider_cls(
                _DEFAULT_CODE_IMAGE,
                is_dockerfile=False,
                temp_base_dir=self._workspace_dir,
                preserve_dir=self._workspace_dir,
            )
        return DockerCodeExecToolProvider.from_image(
            _DEFAULT_CODE_IMAGE,
            temp_base_dir=self._workspace_dir,
        )

    def _build_tools(self) -> list:
        if not self._code_enabled:
            provider = self._build_mcp_provider()
            if self._topology == "gateway":
                return [self._wrap_in_gateway(provider)]
            return [provider]

        code_provider = self._build_code_provider()
        mcp_provider = self._build_mcp_provider(exec_env=code_provider)
        if self._topology == "gateway":
            mcp_provider = self._wrap_in_gateway(mcp_provider)
        return [
            code_provider,
            *build_handoff_tools(code_provider),
            mcp_provider,
        ]

    def _build_system_prompt(self) -> str:
        """Append code-execution guidance when the code track is enabled."""
        if not self._code_enabled:
            if self._topology == "gateway":
                return f"{AGENT_SYSTEM_PROMPT}\n{_GATEWAY_SYSTEM_PROMPT}"
            return AGENT_SYSTEM_PROMPT

        backend_prompt = (
            _DOCKER_CODE_EXEC_SYSTEM_PROMPT
            if self._code_backend == "docker"
            else _LOCAL_CODE_EXEC_SYSTEM_PROMPT
        )
        parts = [AGENT_SYSTEM_PROMPT, _CODE_EXEC_SYSTEM_PROMPT, backend_prompt]
        if self._topology == "gateway":
            parts.append(_GATEWAY_SYSTEM_PROMPT)
        return "\n".join(parts)

    # -- run ---------------------------------------------------------------

    async def run(self, question: str) -> AgentResult:
        from stirrup import Agent

        with agent_run_span(
            "stirrup-agent", model=self._model_id, question=question
        ) as span:
            run_started = time.perf_counter()
            started_at = _dt.datetime.now(_dt.UTC).isoformat()

            agent = Agent(
                client=self._build_client(),
                name="assetops",
                system_prompt=self._build_system_prompt(),
                tools=self._build_tools(),
                finish_tool=ASSETOPS_FINISH_TOOL,
                max_turns=self._max_turns,
                context_summarization_cutoff=_CONTEXT_SUMMARIZATION_CUTOFF,
                logger=_build_full_summary_logger(),
            )

            _log.info(
                "StirrupAgentRunner: starting (model=%s, code=%s, backend=%s, workspace=%s, preserve=%s)",
                self._model_id,
                self._code_enabled,
                self._code_backend,
                self._workspace_dir,
                self._preserve_workspace,
            )

            async with agent.session() as session:
                finish_params, history, _metadata = await session.run(question)

            trajectory = build_trajectory(history)
            trajectory.started_at = started_at
            answer = final_answer(history, finish_params)

            self._annotate_span(span, trajectory, answer, run_started)
            persist_trajectory(
                runner_name="stirrup-agent",
                model=self._model_id,
                question=question,
                answer=answer,
                trajectory=trajectory,
            )
            return AgentResult(question=question, answer=answer, trajectory=trajectory)

    def _annotate_span(
        self, span, trajectory: Trajectory, answer: str, started: float
    ) -> None:
        domain_servers = set(self._server_paths)
        counts = {"domain": 0, "code": 0, "other": 0}
        discovery_calls = 0
        for tc in trajectory.all_tool_calls:
            counts[classify_tool(tc.name, domain_servers, tc.input)] += 1
            if tc.name in GATEWAY_DISCOVERY_TOOLS:
                discovery_calls += 1
        total_tools = sum(counts.values())
        bypass = self._code_enabled and counts["code"] > 0 and counts["domain"] == 0

        span.set_attribute("agent.answer.length", len(answer))
        span.set_attribute("gen_ai.usage.input_tokens", trajectory.total_input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", trajectory.total_output_tokens)
        span.set_attribute("agent.turns", len(trajectory.turns))
        span.set_attribute("agent.tool_calls", total_tools)
        span.set_attribute("agent.duration_ms", (time.perf_counter() - started) * 1000)
        span.set_attribute("agent.code_track", self._code_enabled)
        span.set_attribute("agent.domain_tool_calls", counts["domain"])
        span.set_attribute("agent.code_tool_calls", counts["code"])
        span.set_attribute("agent.tool_bypass", bypass)
        span.set_attribute("agent.topology", self._topology)
        # Peak single-request input is the context-pressure number the topology
        # comparison turns on; total input tokens move the other way.
        span.set_attribute(
            "agent.peak_context_tokens",
            max((t.input_tokens for t in trajectory.turns), default=0),
        )
        if self._topology == "gateway":
            # Discovery is the gateway's characteristic cost: turns spent
            # finding and reading schemas rather than doing domain work.
            span.set_attribute("agent.gateway_mode", self._gateway_mode)
            span.set_attribute("agent.gateway_discovery_calls", discovery_calls)
            if self._gateway is not None:
                span.set_attribute("agent.gateway_tool_count", self._gateway.tool_count)
                span.set_attribute(
                    "agent.gateway_schemas_disclosed", len(self._gateway.described)
                )

        _log.info(
            "StirrupAgentRunner: done (topology=%s, turns=%d, domain=%d, "
            "code=%d, discovery=%d, bypass=%s)",
            self._topology,
            len(trajectory.turns),
            counts["domain"],
            counts["code"],
            discovery_calls,
            bypass,
        )
