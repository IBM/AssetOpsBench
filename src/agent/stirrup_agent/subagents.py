"""Domain sub-agents: one Stirrup ``Agent`` per AssetOpsBench MCP server.

Topology
--------
``--topology flat`` (the default) attaches every MCP server directly to the root
agent, so all of the domain tool schemas sit in the root context on every turn.
``--topology subagent`` attaches each domain server to its own single-server
sub-agent and gives the root one sub-agent tool per domain instead.

``utilities`` deliberately stays on the root. It is six trivial tools (dates and
catalogue lookups) that every domain needs, and siloing it would force a
delegation hop to ask what today's date is.

Two invariants this design depends on
-------------------------------------
1. **Sub-agents never execute code.** Each receives exactly one tool provider:
   the workspace-bridged MCP provider for its own server. Do *not* reach for
   ``Agent(share_parent_exec_env=True)``. It looks like the right lever, but
   ``Agent.__aenter__`` responds to it by calling ``exec_env.get_code_exec_tool()``
   and appending the result to the sub-agent's active tools, which is precisely
   what we are excluding.

2. **Every oversized MCP result lands in the root's workspace.**
   :class:`WorkspaceBridgedMCPToolProvider` takes the root's
   ``CodeExecToolProvider`` as a *constructor argument*, not as a tool, so a
   sub-agent can spill a large sensor history into ``mcp_results/`` in the root's
   execution environment while remaining unable to run a line of Python. A
   side effect worth knowing: because a sub-agent owns no exec env of its own,
   Stirrup's sub-agent file transfer never runs. ``Agent.__aexit__`` guards the
   ``save_output_files`` path on ``state.exec_env``, so it is skipped entirely
   and no files are copied between environments. The bridge is the only path,
   which is exactly the property we want.

Why the finish tool is load-bearing
-----------------------------------
``Agent.to_tool`` composes the tool result the root actually sees from just
three things: the sub-agent's last assistant message with non-empty text, a
``model_dump()`` of its finish params, and the transferred-file lists (always
empty here, per invariant 2). The full ``message_history`` goes into
``SubAgentMetadata``, which is run metadata and never enters the root's context.

That is what makes the topology save context, and it is also the trap: a
sub-agent that ends with "I analysed the sensor data" has destroyed every
workspace handle and identifier it discovered, and the root cannot re-run its
tools to recover them. :func:`build_domain_finish_tool` forces those through a
typed channel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:  # pragma: no cover - import cost only matters at runtime
    from stirrup.core.models import Tool
    from stirrup.tools.code_backends.base import CodeExecToolProvider

_log = logging.getLogger(__name__)

# Servers that become their own sub-agent under --topology subagent.
SUBAGENT_SERVERS: frozenset[str] = frozenset({"iot", "fmsr", "tsfm", "wo", "vibration"})
# Servers that stay attached to the root agent in every topology.
ROOT_SERVERS: frozenset[str] = frozenset({"utilities"})

# Sub-agents work one domain at a time and should compact early rather than
# hoard. max_tokens must stay <= context_window_tokens: Stirrup 0.2.0 validates
# the pair in the client constructor.
SUBAGENT_CONTEXT_WINDOW_TOKENS = 40_000
SUBAGENT_MAX_OUTPUT_TOKENS = 16_000
SUBAGENT_MAX_TURNS = 12


# --------------------------------------------------------------------------
# Routing manifests
# --------------------------------------------------------------------------
# The root no longer sees individual tool names, so these descriptions are the
# entire basis on which it routes. They describe capabilities rather than
# enumerate tools, so they survive tool-level churn; ``test_subagents.py``
# fails if a server in SUBAGENT_SERVERS has no manifest.

DOMAIN_MANIFESTS: dict[str, str] = {
    "iot": (
        "Sensor and asset inventory plus historical readings. Ask it to: list sites "
        "and the assets at a site; describe one asset; list the sensors measured or "
        "installed on an asset; find assets by the sensors they carry; report a "
        "sensor stream's time extent, coverage, latest reading or summary "
        "statistics; and pull raw sensor history over a time range. Raw history can "
        "be large and comes back as a workspace artifact handle rather than inline "
        "data. It does not forecast, diagnose, or touch work orders."
    ),
    "fmsr": (
        "Failure modes and their sensor mappings, by asset class. Ask it to: list "
        "the known failure modes for an asset class, generate candidate failure "
        "modes, or record new ones. It does not read sensor data and does not know "
        "what any particular asset is currently doing."
    ),
    "tsfm": (
        "Time-series foundation models, features, and evaluation runs. Ask it to: "
        "discover or describe forecasting and anomaly-detection models and features; "
        "profile, characterize or quality-check a series; build and run a recipe or "
        "plan; and read back results, runs and lineage. It also owns registration "
        "and versioning of models and features. It reads data by path or reference, "
        "so give it a workspace artifact handle or a dataset path, never raw numbers "
        "pasted into the task."
    ),
    "vibration": (
        "Vibration analysis for rotating equipment. Ask it to: list vibration "
        "sensors, fetch vibration data, compute FFT or envelope spectra, calculate "
        "bearing fault frequencies, look up known bearings, assess severity, and "
        "produce a diagnosis. It covers vibration only, not general sensor history."
    ),
    "wo": (
        "Maintenance work orders. Ask it to: list, read and filter work orders and "
        "their tasks, costs, actuals-versus-planned, KPIs and schedule calendar; "
        "look up failure codes; and generate, update, approve, assign, close or "
        "cancel a work order. Mutations are real state changes: state exactly one "
        "intended change per task, and never ask it to retry a mutation that may "
        "already have succeeded."
    ),
}

_SUBAGENT_TOOL_PREAMBLE = (
    "Delegate one self-contained {server} task and receive a short result. This "
    "sub-agent cannot see your conversation, so the task must name every "
    "identifier it needs (asset ids, sensor names, sites, time ranges, model ids). "
    "It returns an answer, the identifiers it found, and handles for any workspace "
    "files its tools produced.\n\n"
)

_DOMAIN_SYSTEM_PROMPT = """\
You are the {server} specialist in an industrial asset operations system. You
have the {server} MCP tools and nothing else: you cannot run code, browse the
web, or call another domain's tools.

Do only the task you were given. If it needs data you cannot reach, say so
plainly and finish. Never guess a value, and never describe what another
domain's tools would have returned.

The caller cannot see your tool calls. Only your finish parameters reach it, so
when you finish:
  - answer: the result, short and factual, in the format the task asked for.
  - entities: every identifier the caller needs to continue, as flat key/value
    text (for example asset_id, sensor_names, site, model_id, workorder_id).
    Comma-separate multiple values. An identifier missing here is lost.
  - artifacts: one entry for every workspace artifact handle your tools
    returned, with workspace_file, bytes and sha256 copied verbatim from the
    tool result. A handle you omit is unreachable, because the caller cannot
    re-run your tools to regenerate it.
"""


# --------------------------------------------------------------------------
# Typed finish params
# --------------------------------------------------------------------------


class DomainArtifact(BaseModel):
    """A workspace file handle produced by an MCP tool during a sub-agent run."""

    workspace_file: str = Field(
        min_length=1,
        description=(
            "Path of the artifact inside the shared code-execution workspace, "
            "copied verbatim from the tool result (for example "
            "'mcp_results/iot__history_ab12cd34ef56_0011223344.json')."
        ),
    )
    tool: str = Field(
        default="",
        description="Name of the MCP tool that produced the artifact.",
    )
    bytes: int = Field(
        default=0,
        ge=0,
        description="Size in bytes, copied verbatim from the tool result.",
    )
    sha256: str = Field(
        default="",
        description="Content hash, copied verbatim from the tool result.",
    )


class DomainFinishParams(BaseModel):
    """What a domain sub-agent hands back to the root agent."""

    answer: str = Field(
        min_length=1,
        description=(
            "Result of the delegated task, in the format the task requested. "
            "Content only: no status commentary, no description of your process."
        ),
    )
    entities: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Identifiers the caller needs to continue, as flat key/value text. "
            "Comma-separate multiple values. Omit nothing the caller would have "
            "to re-derive."
        ),
    )
    artifacts: list[DomainArtifact] = Field(
        default_factory=list,
        description=(
            "Workspace artifact handles returned by your tools during this task. "
            "Copy each field verbatim from the tool result."
        ),
    )
    reason: str = Field(
        default="",
        description="Optional internal note on why the run ended. Not shown to the user.",
    )

    @field_validator("answer")
    @classmethod
    def _answer_must_contain_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must include at least one non-whitespace character")
        return value


def build_domain_finish_tool(exec_env: "CodeExecToolProvider | None") -> "Tool":
    """Build the finish tool for domain sub-agents.

    ``exec_env`` is the *root's* code-execution provider, captured in the
    closure. This matters: :data:`~agent.stirrup_agent.finish_tool.ASSETOPS_FINISH_TOOL`
    resolves its exec env from ``stirrup.core.agent._SESSION_STATE``, and inside
    a sub-agent run that ContextVar points at the sub-agent's own session, which
    has no exec env. Reusing the root finish tool here would therefore skip
    artifact validation silently instead of catching a hallucinated handle.
    """
    from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata

    async def executor(params: DomainFinishParams) -> "ToolResult[ToolUseCountMetadata]":
        if exec_env is not None and params.artifacts:
            missing = []
            for artifact in params.artifacts:
                try:
                    exists = await exec_env.file_exists(artifact.workspace_file)
                except Exception:  # noqa: BLE001 - a probe failure must not end the run
                    _log.debug(
                        "Could not verify sub-agent artifact %s",
                        artifact.workspace_file,
                        exc_info=True,
                    )
                    continue
                if not exists:
                    missing.append(artifact.workspace_file)
            if missing:
                return ToolResult(
                    content=(
                        "These artifact paths do not exist in the workspace: "
                        f"{missing}. Copy workspace_file verbatim from the tool "
                        "result that produced it, or drop the entry, then finish "
                        "again."
                    ),
                    success=False,
                    metadata=ToolUseCountMetadata(),
                )
        return ToolResult(content="", metadata=ToolUseCountMetadata())

    return Tool(
        name="finish",
        description=(
            "End the task and return your result to the caller. Everything the "
            "caller receives comes from these parameters."
        ),
        parameters=DomainFinishParams,
        executor=executor,
    )


# --------------------------------------------------------------------------
# History recording
# --------------------------------------------------------------------------


class SubAgentHistoryRecorder:
    """Capture each sub-agent's message history in call order.

    Stirrup already returns sub-agent histories in the root's ``run_metadata``,
    but that dict is pruned when the root summarizes: ``Agent`` drops a turn's
    metadata alongside the turn itself (``run_metadata_by_turn.pop(...)``). On a
    long scenario the earliest delegations would silently vanish from the
    trajectory, and with them their domain tool calls. Recording out of band
    keeps tree-wide tool counts honest regardless of summarization.
    """

    def __init__(self) -> None:
        self.histories: dict[str, list[list[list[Any]]]] = {}

    def record(self, tool_name: str, history: list[list[Any]]) -> None:
        self.histories.setdefault(tool_name, []).append(history)

    @property
    def call_count(self) -> int:
        return sum(len(calls) for calls in self.histories.values())


def _recording_tool(tool: "Tool", recorder: SubAgentHistoryRecorder) -> "Tool":
    """Wrap a sub-agent tool so its message history is recorded on every call.

    One cosmetic side effect: ``Agent._collect_all_tools`` discovers sub-agents
    by inspecting a tool executor's closure for an ``Agent`` instance, and this
    wrapper's closure holds the original executor rather than the agent. Stirrup
    therefore stops counting sub-agent tools in its own configuration warnings.
    Those warnings are about missing *default* tools, which this runner never
    attaches anyway (attaching web search would contaminate the benchmark), so
    nothing actionable is lost.
    """
    inner = tool.executor
    name = tool.name

    async def executor(params: Any) -> Any:
        result = await inner(params)
        history = getattr(getattr(result, "metadata", None), "message_history", None)
        recorder.record(name, history if history else [])
        return result

    return tool.model_copy(update={"executor": executor})


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def build_domain_subagent(
    server: str,
    *,
    client: Any,
    mcp_provider: Any,
    exec_env: "CodeExecToolProvider | None",
    max_turns: int = SUBAGENT_MAX_TURNS,
    context_summarization_cutoff: float = 0.75,
    logger: Any = None,
) -> Any:
    """Build one single-server domain sub-agent.

    ``mcp_provider`` must already be scoped to ``server`` and already bridged to
    ``exec_env``; see :meth:`StirrupAgentRunner._build_mcp_provider`.
    """
    from stirrup import Agent

    # The agent name becomes the delegation tool's name. The ``_agent`` suffix is
    # load-bearing: ``classify_tool`` buckets a call as "domain" when the segment
    # before ``__`` is a registered server name, so an agent named plainly
    # ``tsfm`` would make every delegation look like a domain tool call and
    # double-count against the spliced sub-agent calls underneath it.
    return Agent(
        client=client,
        name=f"{server}_agent",
        system_prompt=_DOMAIN_SYSTEM_PROMPT.format(server=server),
        tools=[mcp_provider],
        finish_tool=build_domain_finish_tool(exec_env),
        max_turns=max_turns,
        context_summarization_cutoff=context_summarization_cutoff,
        # Explicit: sharing would hand this agent a code_exec tool. See module docstring.
        share_parent_exec_env=False,
        logger=logger,
    )


def build_subagent_tools(
    servers: list[str],
    *,
    client_factory: Any,
    provider_factory: Any,
    exec_env: "CodeExecToolProvider | None",
    recorder: SubAgentHistoryRecorder,
    max_turns: int = SUBAGENT_MAX_TURNS,
    logger: Any = None,
) -> list["Tool"]:
    """Build one sub-agent tool per domain server, in ``servers`` order.

    Args:
        servers: Domain server names, each of which must have a manifest.
        client_factory: Zero-argument callable returning a fresh Stirrup client
            configured with the sub-agent context budget.
        provider_factory: Callable ``(server) -> ToolProvider`` returning a
            workspace-bridged MCP provider scoped to that one server.
        exec_env: The root's code-execution provider, or None on the no-code
            track (in which case artifact validation is skipped).
        recorder: Collects sub-agent histories for trajectory flattening.
    """
    tools: list[Tool] = []
    for server in servers:
        manifest = DOMAIN_MANIFESTS.get(server)
        if manifest is None:
            raise ValueError(
                f"No routing manifest for MCP server {server!r}. Add one to "
                "DOMAIN_MANIFESTS: under --topology subagent it is the only thing "
                "the root agent knows about this domain."
            )
        subagent = build_domain_subagent(
            server,
            client=client_factory(),
            mcp_provider=provider_factory(server),
            exec_env=exec_env,
            max_turns=max_turns,
            logger=logger,
        )
        tool = subagent.to_tool(
            description=_SUBAGENT_TOOL_PREAMBLE.format(server=server) + manifest,
        )
        tools.append(_recording_tool(tool, recorder))
    return tools