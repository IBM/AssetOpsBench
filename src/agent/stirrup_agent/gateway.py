"""An MCP gateway: dynamic tool routing with deferred schema disclosure.

Third topology, alongside ``flat`` and ``subagent``. Where delegation partitions
the *agent*, the gateway partitions the *tool manifest*: every MCP server stays
behind one provider, the root keeps a single context and a single trajectory,
and full JSON Schemas are disclosed only for the tools a step actually needs.

The root sees three tools instead of 85:

``search_tools(query, k)``
    Rank the catalogue against a natural-language description of the immediate
    need and return the top ``k`` as ``{name, server, summary}``. This is the
    top-``k`` recommendation protocol MCPBench evaluates.
``describe_tools(names)``
    Return the full parameter schema for named tools. This is the deferred
    half of the cost: schemas are the bulk of a manifest, and a scenario
    touching two domains never pays for the other four.
``call_tool(name, arguments)``
    Validate ``arguments`` against the real tool's model and invoke it.

Two modes, because they separate two different savings
------------------------------------------------------
``index``
    A compact one-line-per-tool manifest of the whole catalogue is pinned into
    ``describe_tools``' description, so it is re-sent every turn. The agent
    always knows what exists and pays only for schema deferral. This is the
    "fixed context, dynamic schema probing" variant.
``search``
    No manifest. The agent is blind to the catalogue until it searches. Minimal
    per-turn cost, but discovery becomes a hard dependency on retrieval quality.

Reporting both isolates how much of any saving comes from deferring schemas
versus from withholding the catalogue, which a single mode cannot distinguish.

Retrieval is lexical (BM25) and implemented here rather than delegated to an
embedding model, deliberately. A benchmark result in which retrieval quality is
an unpinned variable is not replayable, and an offline deterministic ranker
keeps the gateway arm reproducible from the repository alone.

Tool attribution
----------------
Every domain call now arrives as ``call_tool``, which would zero
``agent.domain_tool_calls`` and destroy the ``tool_bypass`` metric exactly as
delegation would have. :func:`~agent.stirrup_agent.trajectory.classify_tool`
therefore reads the ``name`` argument of a gateway call and credits the
underlying server, so counts stay comparable across all three topologies.

The workspace bridge is untouched: the gateway wraps
:class:`WorkspaceBridgedMCPToolProvider`, so an oversized result still spills to
``mcp_results/`` in the root's execution environment and comes back as a handle.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:  # pragma: no cover
    from stirrup.core.models import Tool

try:  # pragma: no cover - exercised by whichever path the environment takes
    from stirrup.core.models import ToolProvider as _ToolProviderBase
except ImportError:  # pragma: no cover
    # Stirrup absent: fall back so the retrieval half of this module, and
    # `trajectory.classify_tool` which imports the tool-name constants below,
    # stay importable and testable without the framework. Real runs always take
    # the branch above, and they must: `Agent.__init__` decides what to enter
    # with `isinstance(t, ToolProvider)`, so a gateway that did not subclass it
    # would silently never connect and the agent would start with no tools.
    _ToolProviderBase = object  # type: ignore[assignment,misc]

_log = logging.getLogger(__name__)

GATEWAY_MODES = ("index", "search")

# Names the root sees. Kept module-level so trajectory classification and the
# span attributes agree with the provider on what counts as discovery overhead.
GATEWAY_CALL_TOOL = "call_tool"
GATEWAY_SEARCH_TOOL = "search_tools"
GATEWAY_DESCRIBE_TOOL = "describe_tools"
GATEWAY_DISCOVERY_TOOLS = frozenset({GATEWAY_SEARCH_TOOL, GATEWAY_DESCRIBE_TOOL})
GATEWAY_TOOLS = frozenset({GATEWAY_CALL_TOOL, *GATEWAY_DISCOVERY_TOOLS})

DEFAULT_TOP_K = 3
MAX_TOP_K = 10
MAX_DESCRIBE = 8
_SUMMARY_CHARS = 110


# --------------------------------------------------------------------------
# Lexical retrieval
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
# BM25 defaults. Fixed rather than tuned: tuning retrieval against the
# evaluation set would leak the scenarios into the routing layer.
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with snake_case and camelCase split apart.

    Tool names carry most of the retrieval signal in this catalogue
    (``get_workorder_costs`` says more than its docstring), so splitting them
    into words matters more than stemming would.
    """
    parts: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        for piece in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|[0-9]+", raw) or [raw]:
            if piece:
                parts.append(piece.lower())
    return parts


class ToolCard(BaseModel):
    """One catalogue entry: what the agent sees before it asks for a schema."""

    name: str
    server: str
    summary: str
    description: str = ""

    def manifest_line(self) -> str:
        return f"  {self.name}: {self.summary}"


def _summarize(description: str, limit: int = _SUMMARY_CHARS) -> str:
    """First sentence of a docstring, clipped, on one line."""
    text = " ".join((description or "").split())
    if not text:
        return "(no description)"
    sentence = re.split(r"(?<=[.!?])\s", text)[0]
    if len(sentence) <= limit:
        return sentence
    return sentence[: limit - 1].rstrip() + "…"


class ToolIndex:
    """A BM25 index over MCP tool names and descriptions."""

    def __init__(self, cards: list[ToolCard]) -> None:
        self.cards = cards
        self._docs = [_tokenize(f"{c.name} {c.server} {c.description}") for c in cards]
        self._lengths = [len(d) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._docs)) if self._docs else 0.0
        self._tf = [Counter(d) for d in self._docs]
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(set(doc))
        n = len(self._docs)
        self._idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    @classmethod
    def build(cls, tools: list["Tool"]) -> "ToolIndex":
        cards = []
        for tool in tools:
            server = tool.name.split("__", 1)[0] if "__" in tool.name else ""
            description = tool.description or ""
            cards.append(
                ToolCard(
                    name=tool.name,
                    server=server,
                    summary=_summarize(description),
                    description=description,
                )
            )
        cards.sort(key=lambda c: c.name)
        return cls(cards)

    def search(self, query: str, k: int) -> list[tuple[ToolCard, float]]:
        terms = _tokenize(query)
        if not terms or not self._docs:
            return []
        scored: list[tuple[ToolCard, float]] = []
        for i, tf in enumerate(self._tf):
            length = self._lengths[i] or 1
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + _BM25_K1 * (
                    1 - _BM25_B + _BM25_B * length / (self._avg_len or 1)
                )
                score += idf * (freq * (_BM25_K1 + 1)) / denom
            if score > 0:
                scored.append((self.cards[i], score))
        # Ties broken by name so a run is byte-reproducible.
        scored.sort(key=lambda pair: (-pair[1], pair[0].name))
        return scored[:k]

    def manifest(self) -> str:
        """Compact one-line-per-tool catalogue, grouped by server."""
        by_server: dict[str, list[ToolCard]] = {}
        for card in self.cards:
            by_server.setdefault(card.server or "(unscoped)", []).append(card)
        blocks = []
        for server in sorted(by_server):
            lines = "\n".join(c.manifest_line() for c in by_server[server])
            blocks.append(f"{server}:\n{lines}")
        return "\n".join(blocks)


# --------------------------------------------------------------------------
# Gateway tool parameters
# --------------------------------------------------------------------------


class SearchToolsParams(BaseModel):
    query: str = Field(
        min_length=1,
        description=(
            "What you need to do right now, in your own words. Include the "
            "domain nouns you are working with (asset, sensor, work order, "
            "forecast, bearing) so the ranking has something to match."
        ),
    )
    k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=MAX_TOP_K,
        description=f"How many candidates to return (1-{MAX_TOP_K}).",
    )


class DescribeToolsParams(BaseModel):
    names: list[str] = Field(
        default_factory=list,
        description=(
            "Exact tool names to retrieve full parameter schemas for, at most "
            f"{MAX_DESCRIBE} at a time."
        ),
    )

    @field_validator("names", mode="before")
    @classmethod
    def _listify(cls, value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value) if isinstance(value, (list, tuple)) else []


class CallToolParams(BaseModel):
    name: str = Field(
        min_length=1,
        description="Exact tool name, as returned by search_tools or describe_tools.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arguments object for that tool. Call describe_tools first if you "
            "have not seen its schema."
        ),
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> dict:
        # Models routinely send the arguments object as a JSON string.
        if value is None:
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return value if isinstance(value, dict) else {}


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------

_SEARCH_DESCRIPTION = (
    "Find the MCP tools relevant to what you are doing right now. Returns tool "
    "names with one-line summaries, ranked. Follow with describe_tools to get "
    "the parameters, then call_tool to run one."
)

_DESCRIBE_DESCRIPTION_HEAD = (
    "Get the full parameter schema for specific MCP tools by exact name. Call "
    "this before call_tool for any tool whose schema you have not already seen."
)

_MANIFEST_PREAMBLE = (
    "\n\nAvailable tools (name: summary). Use describe_tools for parameters:\n\n"
)

_CALL_DESCRIPTION = (
    "Run one MCP tool. `name` must be exact and `arguments` must match that "
    "tool's schema; if a call is rejected the error names the missing or "
    "unexpected fields, so fix and retry rather than guessing."
)


class MCPGatewayToolProvider(_ToolProviderBase):
    """Expose a whole MCP catalogue through three routing tools.

    Wraps an already-constructed MCP provider (normally
    :class:`WorkspaceBridgedMCPToolProvider`, so oversized results keep spilling
    into the root's workspace) and never surfaces its tools directly.
    """

    def __init__(
        self,
        inner: Any,
        *,
        mode: str = "index",
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if mode not in GATEWAY_MODES:
            raise ValueError(f"gateway mode must be one of {GATEWAY_MODES}")
        self._inner = inner
        self._mode = mode
        self._top_k = top_k
        self._tools: dict[str, Any] = {}
        self._index: ToolIndex | None = None
        self.described: set[str] = set()
        self.call_counts: Counter[str] = Counter()

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> list["Tool"]:
        tools = await self._inner.__aenter__()
        if not isinstance(tools, list):
            tools = [tools]
        self._tools = {t.name: t for t in tools}
        self._index = ToolIndex.build(tools)
        _log.info(
            "MCP gateway ready: %d tools behind %d routing tools (mode=%s)",
            len(self._tools),
            len(GATEWAY_TOOLS),
            self._mode,
        )
        return self._build_gateway_tools()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._inner.__aexit__(exc_type, exc_val, exc_tb)

    # -- introspection for metrics ----------------------------------------

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def index(self) -> ToolIndex | None:
        return self._index

    # -- gateway tools -----------------------------------------------------

    def _describe_description(self) -> str:
        if self._mode == "index" and self._index is not None:
            return _DESCRIBE_DESCRIPTION_HEAD + _MANIFEST_PREAMBLE + self._index.manifest()
        return _DESCRIBE_DESCRIPTION_HEAD

    def _schema_for(self, tool: Any) -> dict:
        try:
            return tool.parameters.model_json_schema()
        except Exception:  # noqa: BLE001 - a schema failure must not kill the run
            _log.warning("Could not render schema for %s", tool.name, exc_info=True)
            return {}

    def _unknown_tool_message(self, name: str) -> str:
        suggestions = []
        if self._index is not None:
            suggestions = [c.name for c, _ in self._index.search(name, 3)]
        hint = f" Closest matches: {suggestions}." if suggestions else ""
        return f"No tool named {name!r}.{hint} Use search_tools to find the right name."

    def _build_gateway_tools(self) -> list["Tool"]:
        from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata

        def _result(content: str, *, success: bool = True):
            return ToolResult(
                content=content, success=success, metadata=ToolUseCountMetadata()
            )

        async def search_tools(params: SearchToolsParams):
            assert self._index is not None
            hits = self._index.search(params.query, min(params.k, MAX_TOP_K))
            if not hits:
                return _result(
                    "No tool matched that query. Try the domain noun on its own "
                    "(for example 'work order', 'sensor history', 'forecast "
                    "model', 'bearing frequency').",
                    success=False,
                )
            payload = [
                {"name": c.name, "server": c.server, "summary": c.summary}
                for c, _ in hits
            ]
            return _result(json.dumps(payload, ensure_ascii=False))

        async def describe_tools(params: DescribeToolsParams):
            if not params.names:
                return _result(
                    "`names` was empty. Pass the exact tool names you want "
                    "schemas for, from search_tools.",
                    success=False,
                )
            requested = params.names[:MAX_DESCRIBE]
            payload, unknown = {}, []
            for name in requested:
                tool = self._tools.get(name)
                if tool is None:
                    unknown.append(name)
                    continue
                self.described.add(name)
                payload[name] = {
                    "description": tool.description,
                    "parameters": self._schema_for(tool),
                }
            if not payload:
                return _result(self._unknown_tool_message(requested[0]), success=False)
            body = json.dumps(payload, ensure_ascii=False)
            if unknown:
                body += f"\n\nNot found (check the spelling): {unknown}"
            if len(params.names) > MAX_DESCRIBE:
                body += (
                    f"\n\nOnly the first {MAX_DESCRIBE} names were described; "
                    "ask again for the rest."
                )
            return _result(body)

        async def call_tool(params: CallToolParams):
            tool = self._tools.get(params.name)
            if tool is None:
                return _result(self._unknown_tool_message(params.name), success=False)
            try:
                validated = tool.parameters.model_validate(params.arguments)
            except Exception as exc:  # noqa: BLE001 - surfaced to the agent below
                # The gateway's own parameters always validate, so an argument
                # mistake reaches us here instead of being replaced by Stirrup's
                # opaque "Tool arguments are not valid". Return the schema and
                # the real error so the agent can correct itself in one turn.
                return _result(
                    f"Arguments rejected by {params.name}: {exc}\n\n"
                    f"Schema: {json.dumps(self._schema_for(tool), ensure_ascii=False)}",
                    success=False,
                )
            self.call_counts[params.name] += 1
            return await tool.executor(validated)

        return [
            Tool(
                name=GATEWAY_SEARCH_TOOL,
                description=_SEARCH_DESCRIPTION,
                parameters=SearchToolsParams,
                executor=search_tools,
            ),
            Tool(
                name=GATEWAY_DESCRIBE_TOOL,
                description=self._describe_description(),
                parameters=DescribeToolsParams,
                executor=describe_tools,
            ),
            Tool(
                name=GATEWAY_CALL_TOOL,
                description=_CALL_DESCRIPTION,
                parameters=CallToolParams,
                executor=call_tool,
            ),
        ]
