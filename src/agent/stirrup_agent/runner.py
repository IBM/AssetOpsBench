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
  * ``topology="flat"``     -> every MCP server is attached to the root agent,
    so all domain tool schemas sit in the root context on every turn.  This is
    the default and the shape the other four runners use.
  * ``topology="subagent"`` -> each domain server is attached to its own
    single-server sub-agent (see :mod:`.subagents`) and the root sees one
    delegation tool per domain instead.  ``utilities`` stays on the root.
    Requires the code track: the workspace bridge needs an execution
    environment to spill into, and without it a sub-agent would return
    oversized results straight into the root context, which is strictly worse
    than flat.

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

from observability import agent_run_span, persist_trajectory

from llm.routers import resolve_model, resolve_router_creds
from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult, Trajectory
from ..runner import AgentRunner
from .finish_tool import ASSETOPS_FINISH_TOOL
from .subagents import (
    ROOT_SERVERS,
    SUBAGENT_CONTEXT_WINDOW_TOKENS,
    SUBAGENT_MAX_OUTPUT_TOKENS,
    SUBAGENT_MAX_TURNS,
    SUBAGENT_SERVERS,
    SubAgentHistoryRecorder,
    build_subagent_tools,
)
from .trajectory import build_trajectory, classify_tool, final_answer
from .handoff_tools import build_handoff_tools

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
# A code-track image needs the scientific stack the WO/vibration analyses use.
_DEFAULT_CODE_IMAGE = os.environ.get("STIRRUP_CODE_IMAGE", "assetops-code")

TOPOLOGIES = ("flat", "subagent")

# Stirrup 0.2.0 takes the working-context budget as an explicit client argument
# (``context_window_tokens``) instead of reading ``max_tokens`` for both the
# provider output cap and the summarization trigger, so the adapter this module
# used to carry is gone.  ``max_tokens`` must stay <= ``context_window_tokens``;
# the client constructor validates the pair.
_ROOT_CONTEXT_WINDOW_TOKENS = int(
    os.environ.get("STIRRUP_CONTEXT_BUDGET", 100_000)
)
# The summarizer sends the whole history in one non-streaming request. Left at
# 64k this was the slowest call of every run, and a silent stall there parked
# whole sweeps: nothing crosses the wire while a reasoning model thinks, so an
# upstream idle timer drops the flow and a client with no timeout waits forever.
_ROOT_MAX_OUTPUT_TOKENS = int(os.environ.get("STIRRUP_MAX_TOKENS", 16_000))
# Checked once per turn, after the turn, so the cutoff needs headroom for the
# largest single-turn growth. A run that summarized at 96% of a 75% cutoff had
# overshot on one fat tool result.
_CONTEXT_SUMMARIZATION_CUTOFF = float(os.environ.get("STIRRUP_SUMMARIZE_AT", 0.60))
# No request may wait forever. Retries stay low because the timeout is generous.
_REQUEST_TIMEOUT_S = float(os.environ.get("STIRRUP_REQUEST_TIMEOUT", 900))
_REQUEST_MAX_RETRIES = int(os.environ.get("STIRRUP_MAX_RETRIES", 1))
# Keepalive probes hold the flow open through load balancers and NAT, and
# surface a genuinely dead peer in ~4 minutes instead of never.
_TCP_KEEPALIVE_IDLE_S = int(os.environ.get("STIRRUP_TCP_KEEPALIVE_IDLE", 60))
_TCP_KEEPALIVE_INTERVAL_S = int(os.environ.get("STIRRUP_TCP_KEEPALIVE_INTERVAL", 30))
_TCP_KEEPALIVE_COUNT = int(os.environ.get("STIRRUP_TCP_KEEPALIVE_COUNT", 6))

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
_LOCAL_CODE_EXEC_SYSTEM_PROMPT = """\
The local execution workspace is a temporary directory, but commands run on the
host with the current user's permissions. Keep all reads and writes inside the
workspace and use relative paths.
"""
_SUBAGENT_SYSTEM_PROMPT = """\
Domain delegation:
- You have no domain tools of your own. Each domain (iot, fmsr, tsfm, wo,
  vibration) is reached by delegating one self-contained task to its sub-agent.
- A sub-agent cannot see this conversation. Spell out every identifier it needs
  (asset ids, sensor names, sites, time ranges, model ids) in the task itself.
- A sub-agent returns an answer, the identifiers it found, and handles for any
  workspace files its tools produced. Those files are in your own workspace, so
  read them with code_exec by their workspace_file path.
- Carry identifiers forward yourself. Two sub-agents never talk to each other.
- Delegate one domain at a time and use what comes back; do not ask a sub-agent
  to speculate about another domain's data.
"""


def _keepalive_socket_options() -> list[tuple[int, int, int]]:
    """TCP keepalive options, named differently on macOS and Linux."""
    import socket

    idle_opt = getattr(socket, "TCP_KEEPALIVE", getattr(socket, "TCP_KEEPIDLE", None))
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if idle_opt is not None:
        options.append((socket.IPPROTO_TCP, idle_opt, _TCP_KEEPALIVE_IDLE_S))
    for name, value in (
        ("TCP_KEEPINTVL", _TCP_KEEPALIVE_INTERVAL_S),
        ("TCP_KEEPCNT", _TCP_KEEPALIVE_COUNT),
    ):
        opt = getattr(socket, name, None)
        if opt is not None:
            options.append((socket.IPPROTO_TCP, opt, value))
    return options


def _apply_keepalive_transport(client, *, base_url: str, api_key: str) -> None:
    """Swap in an AsyncOpenAI whose sockets carry TCP keepalive.

    ``ChatCompletionsClient`` builds its own ``AsyncOpenAI`` and exposes no
    ``http_client`` hook, so replace the built one. Upstream should grow that
    parameter; until then this is the seam.
    """
    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError:  # pragma: no cover - openai ships with stirrup
        _log.warning("Could not enable TCP keepalive: openai/httpx unavailable")
        return

    try:
        transport = httpx.AsyncHTTPTransport(
            socket_options=_keepalive_socket_options(), retries=0
        )
    except TypeError:
        _log.warning(
            "httpx %s does not support socket_options; TCP keepalive not enabled",
            httpx.__version__,
        )
        return

    client._client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=_REQUEST_MAX_RETRIES,
        http_client=httpx.AsyncClient(transport=transport, timeout=_REQUEST_TIMEOUT_S),
    )


def _build_full_summary_logger():
    """Return a fresh Stirrup logger that displays summaries without truncation.

    A new instance per agent: ``Agent.to_tool`` mutates ``logger.depth`` on the
    sub-agent's session, so sharing one logger across the root and its
    sub-agents would corrupt the indentation of concurrent-looking output.
    """
    from rich.text import Text
    from stirrup.utils.logging import AgentLogger, console

    class _FullSummaryLogger(AgentLogger):
        def context_summarization_complete(self, summary: str, bridge: str) -> None:
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
        topology: ``"flat"`` (all servers on the root) or ``"subagent"``
            (one sub-agent per domain server). ``"subagent"`` requires
            ``code_enabled=True``.
        workspace_dir: Optional host base directory for Docker/local code execution.
        preserve_workspace: Copy final code-execution files back into ``workspace_dir``.
        max_turns: Stirrup agent loop bound for the root agent.
        subagent_max_turns: Loop bound for each domain sub-agent.
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
        workspace_dir: Path | str | None = None,
        preserve_workspace: bool = False,
        max_turns: int = 30,
        subagent_max_turns: int = SUBAGENT_MAX_TURNS,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(llm, server_paths)
        if code_backend not in {"docker", "local"}:
            raise ValueError("code_backend must be 'docker' or 'local'")
        if topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {TOPOLOGIES}")
        if topology == "subagent" and not code_enabled:
            raise ValueError(
                "topology='subagent' requires code_enabled=True. Domain sub-agents "
                "spill oversized MCP results into the root's code-execution "
                "workspace; with no execution environment there is nowhere to "
                "spill, so every large result would land inline in the root "
                "context and the topology would cost more context than flat."
            )
        self._model_id = model
        self._code_enabled = code_enabled
        self._code_backend = code_backend
        self._topology = topology
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
        self._subagent_max_turns = subagent_max_turns
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._recorder = SubAgentHistoryRecorder()

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
        provider's real window so long runs compact early. Domain sub-agents get
        a smaller budget than the root so a single domain cannot hoard.
        """
        client_kwargs: dict = {}
        if self._temperature is not None:
            client_kwargs["temperature"] = self._temperature

        max_output_tokens = min(max_output_tokens, context_window_tokens)
        _log.info(
            "StirrupAgentRunner: client budget=%d max_tokens=%d cutoff=%.2f "
            "timeout=%.0fs retries=%d",
            context_window_tokens,
            max_output_tokens,
            _CONTEXT_SUMMARIZATION_CUTOFF,
            _REQUEST_TIMEOUT_S,
            _REQUEST_MAX_RETRIES,
        )

        creds = resolve_router_creds(self._model_id)
        if creds is not None:
            from stirrup.clients.chat_completions_client import ChatCompletionsClient

            base_url = creds.base_url.rstrip("/")
            client = ChatCompletionsClient(
                model=resolve_model(self._model_id),
                max_tokens=max_output_tokens,
                context_window_tokens=context_window_tokens,
                base_url=base_url,
                api_key=creds.api_key,
                reasoning_effort=self._reasoning_effort,
                timeout=_REQUEST_TIMEOUT_S,
                max_retries=_REQUEST_MAX_RETRIES,
                kwargs=client_kwargs or None,
            )
            _apply_keepalive_transport(client, base_url=base_url, api_key=creds.api_key)
            return client

        from stirrup.clients.litellm_client import LiteLLMClient

        # LiteLLM forwards unknown kwargs to acompletion, so the timeout rides
        # along here rather than as a constructor argument.
        client_kwargs.setdefault("timeout", _REQUEST_TIMEOUT_S)
        return LiteLLMClient(
            model=self._model_id,
            max_tokens=max_output_tokens,
            context_window_tokens=context_window_tokens,
            reasoning_effort=self._reasoning_effort,
            kwargs=client_kwargs,
        )

    def _build_subagent_client(self):
        return self._build_client(
            context_window_tokens=SUBAGENT_CONTEXT_WINDOW_TOKENS,
            max_output_tokens=SUBAGENT_MAX_OUTPUT_TOKENS,
        )

    def _build_mcp_config(self):
        """Build the Stirrup MCP configuration for AssetOpsBench servers.

        Each server is a stdio process launched exactly as the other runners
        launch it: ``uv run --directory <repo> <entry-point>``.  Every server is
        always present in the config; which of them a given provider connects to
        is decided per provider via ``server_names``.
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

    def _build_mcp_provider(
        self,
        *,
        exec_env=None,
        server_names: list[str] | None = None,
        reader_has_code_exec: bool = True,
    ):
        """Build an MCP provider, bridging large results when code is enabled.

        ``reader_has_code_exec`` is False for a domain sub-agent's provider:
        only the root holds code_exec, so a handle returned to a sub-agent must
        tell it to pass the handle up rather than to analyse the file.
        """
        config = self._build_mcp_config()
        if exec_env is None:
            from stirrup.tools.mcp import MCPToolProvider

            return MCPToolProvider(config=config, server_names=server_names)

        from .workspace_bridge import (
            DEFAULT_PERSIST_THRESHOLD_BYTES,
            SUBAGENT_PERSIST_THRESHOLD_BYTES,
            WorkspaceBridgedMCPToolProvider,
        )

        return WorkspaceBridgedMCPToolProvider(
            config=config,
            exec_env=exec_env,
            server_names=server_names,
            reader_has_code_exec=reader_has_code_exec,
            persist_threshold_bytes=(
                DEFAULT_PERSIST_THRESHOLD_BYTES
                if reader_has_code_exec
                else SUBAGENT_PERSIST_THRESHOLD_BYTES
            ),
        )

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

    def _partition_servers(self) -> tuple[list[str], list[str]]:
        """Split registered servers into (root-attached, delegated-to-sub-agents).

        A registered server that is in neither set stays on the root: a new
        server should show up in the root's tool list rather than disappearing
        from the run because nobody added it to :data:`SUBAGENT_SERVERS`.
        """
        delegated = [n for n in self._server_paths if n in SUBAGENT_SERVERS]
        root = [n for n in self._server_paths if n not in SUBAGENT_SERVERS]
        unclassified = [n for n in root if n not in ROOT_SERVERS]
        if unclassified:
            _log.warning(
                "MCP servers %s are not classified in subagents.py; keeping them "
                "on the root agent.",
                unclassified,
            )
        return root, delegated

    def _build_tools(self) -> list:
        if not self._code_enabled:
            return [self._build_mcp_provider()]

        code_provider = self._build_code_provider()
        base = [code_provider, *build_handoff_tools(code_provider)]

        if self._topology == "flat":
            return [*base, self._build_mcp_provider(exec_env=code_provider)]

        root_servers, delegated = self._partition_servers()
        subagent_tools = build_subagent_tools(
            delegated,
            client_factory=self._build_subagent_client,
            provider_factory=lambda server: self._build_mcp_provider(
                exec_env=code_provider,
                server_names=[server],
                reader_has_code_exec=False,
            ),
            exec_env=code_provider,
            recorder=self._recorder,
            max_turns=self._subagent_max_turns,
            logger=_build_full_summary_logger(),
        )
        tools = list(base)
        if root_servers:
            tools.append(
                self._build_mcp_provider(
                    exec_env=code_provider, server_names=root_servers
                )
            )
        tools.extend(subagent_tools)
        return tools

    def _build_system_prompt(self) -> str:
        """Append code-execution and delegation guidance to the shared prompt."""
        parts = [AGENT_SYSTEM_PROMPT]
        if self._code_enabled:
            parts.append(_CODE_EXEC_SYSTEM_PROMPT)
            parts.append(
                _DOCKER_CODE_EXEC_SYSTEM_PROMPT
                if self._code_backend == "docker"
                else _LOCAL_CODE_EXEC_SYSTEM_PROMPT
            )
        if self._topology == "subagent":
            parts.append(_SUBAGENT_SYSTEM_PROMPT)
        return "\n".join(parts)

    # -- run ---------------------------------------------------------------

    async def run(self, question: str) -> AgentResult:
        from stirrup import Agent

        with agent_run_span(
            "stirrup-agent", model=self._model_id, question=question
        ) as span:
            run_started = time.perf_counter()
            started_at = _dt.datetime.now(_dt.UTC).isoformat()

            self._recorder = SubAgentHistoryRecorder()
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
                "StirrupAgentRunner: starting (model=%s, code=%s, backend=%s, "
                "topology=%s, workspace=%s, preserve=%s)",
                self._model_id,
                self._code_enabled,
                self._code_backend,
                self._topology,
                self._workspace_dir,
                self._preserve_workspace,
            )

            async with agent.session() as session:
                finish_params, history, _metadata = await session.run(question)

            trajectory = build_trajectory(
                history, sub_histories=self._recorder.histories
            )
            trajectory.started_at = started_at
            answer = final_answer(history, finish_params)

            # Computed once, read twice: the span carries it to a tracing
            # backend, and the field rides along into the persisted trajectory
            # so a sweep can be analysed from the run files alone. Two
            # accumulators would eventually disagree.
            trajectory.metrics = self._run_metrics(trajectory, answer, run_started)
            self._annotate_span(span, trajectory.metrics)
            persist_trajectory(
                runner_name="stirrup-agent",
                model=self._model_id,
                question=question,
                answer=answer,
                trajectory=trajectory,
            )
            return AgentResult(question=question, answer=answer, trajectory=trajectory)

    def _run_metrics(
        self, trajectory: Trajectory, answer: str, started: float
    ) -> dict:
        """Compute the per-run accounting once, for both spans and the record.

        Root-only and tree-wide figures diverge under ``--topology subagent``,
        and that divergence is the experiment: the topology buys root context
        headroom by re-paying system prompts and tool schemas inside every
        delegation, so cost and context move in opposite directions. Reporting
        only the total hides the saving; reporting only the peak hides the bill.

        Sub-agent turns are spliced into the trajectory at ``depth >= 1``, so
        every figure below is a filter on that field rather than a separate
        accumulator that could drift from the turns it describes.
        """
        domain_servers = set(self._server_paths)
        counts = {"domain": 0, "code": 0, "other": 0}
        for tc in trajectory.all_tool_calls:
            counts[classify_tool(tc.name, domain_servers)] += 1

        root_turns = [t for t in trajectory.turns if t.depth == 0]
        sub_turns = [t for t in trajectory.turns if t.depth > 0]
        root_input = sum(t.input_tokens for t in root_turns)

        # Per-agent breakdown, so a sweep can see which domain dominated a run
        # without re-walking the turn list.
        by_agent: dict[str, dict[str, int]] = {}
        for turn in trajectory.turns:
            entry = by_agent.setdefault(
                turn.agent,
                {"turns": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0},
            )
            entry["turns"] += 1
            entry["tool_calls"] += len(turn.tool_calls)
            entry["input_tokens"] += turn.input_tokens
            entry["output_tokens"] += turn.output_tokens

        return {
            "topology": self._topology,
            "code_track": self._code_enabled,
            "answer_length": len(answer),
            "duration_ms": (time.perf_counter() - started) * 1000,
            "turns": len(trajectory.turns),
            "tool_calls": sum(counts.values()),
            "domain_tool_calls": counts["domain"],
            "code_tool_calls": counts["code"],
            "other_tool_calls": counts["other"],
            "tool_bypass": (
                self._code_enabled and counts["code"] > 0 and counts["domain"] == 0
            ),
            "input_tokens": trajectory.total_input_tokens,
            "output_tokens": trajectory.total_output_tokens,
            # Cost axis is input_tokens; context axis is root_peak_context_tokens.
            "root_turns": len(root_turns),
            "root_input_tokens": root_input,
            "root_peak_context_tokens": max(
                (t.input_tokens for t in root_turns), default=0
            ),
            "subagent_turns": len(sub_turns),
            "subagent_input_tokens": trajectory.total_input_tokens - root_input,
            "subagent_calls": self._recorder.call_count,
            "by_agent": by_agent,
        }

    def _annotate_span(self, span, metrics: dict) -> None:
        span.set_attribute("agent.answer.length", metrics["answer_length"])
        span.set_attribute("gen_ai.usage.input_tokens", metrics["input_tokens"])
        span.set_attribute("gen_ai.usage.output_tokens", metrics["output_tokens"])
        span.set_attribute("agent.turns", metrics["turns"])
        span.set_attribute("agent.tool_calls", metrics["tool_calls"])
        span.set_attribute("agent.duration_ms", metrics["duration_ms"])
        span.set_attribute("agent.code_track", metrics["code_track"])
        span.set_attribute("agent.topology", metrics["topology"])
        span.set_attribute("agent.domain_tool_calls", metrics["domain_tool_calls"])
        span.set_attribute("agent.code_tool_calls", metrics["code_tool_calls"])
        span.set_attribute("agent.tool_bypass", metrics["tool_bypass"])
        span.set_attribute("agent.root_turns", metrics["root_turns"])
        span.set_attribute("agent.root_input_tokens", metrics["root_input_tokens"])
        span.set_attribute(
            "agent.root_peak_context_tokens", metrics["root_peak_context_tokens"]
        )
        span.set_attribute(
            "agent.subagent_input_tokens", metrics["subagent_input_tokens"]
        )
        span.set_attribute("agent.subagent_calls", metrics["subagent_calls"])

        _log.info(
            "StirrupAgentRunner: done (topology=%s, turns=%d, root_turns=%d, "
            "domain=%d, code=%d, subagent_calls=%d, root_peak_ctx=%d, "
            "total_in=%d, bypass=%s)",
            metrics["topology"],
            metrics["turns"],
            metrics["root_turns"],
            metrics["domain_tool_calls"],
            metrics["code_tool_calls"],
            metrics["subagent_calls"],
            metrics["root_peak_context_tokens"],
            metrics["input_tokens"],
            metrics["tool_bypass"],
        )
