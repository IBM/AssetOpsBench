"""The adapter specification: what a harness patch is allowed to change.

The MCP servers under ``src/servers/`` are the environment and never change.
The adapter is the client-side layer between the agent and those servers, and
it is the only surface an evolved harness patch may touch.

Everything in this module is pure stdlib and free of any agent framework, so
the adaptation logic is testable without a model, a container, or a live MCP
server. The framework-bound part is a ten-line subclass in ``provider.py``.

The five change kinds map onto AutoSaddler's patch taxonomy (arXiv 2608.23041,
Table 1), restricted to the subtypes that survive a frozen server:

===================  ==============================  ====================
adapter field        AutoSaddler subtype             kind
===================  ==============================  ====================
``rename``           Argument Modification           Capability
``drop``             Argument Modification           Capability
``require``          Argument Modification           Capability
``defaults``         Argument Modification           Capability
``strip_empty``      Argument Modification           Capability
``description``      Tool Description Fix            Steering
``hook``             PreToolUse Hook                 Steering
``composites``       New Tool Addition               Capability
``hidden``           Tool Description Fix            Steering
===================  ==============================  ====================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SPEC_FILENAME = "adapter.json"

#: Every key an adapter spec may carry. A spec with any other key is invalid,
#: which keeps an evolved patch from smuggling behaviour past the validator.
TOOL_KEYS = frozenset(
    {"rename", "drop", "require", "defaults", "strip_empty", "description", "hook"}
)
SPEC_KEYS = frozenset({"tools", "composites", "hidden"})


class AdapterError(ValueError):
    """The spec is malformed. Raised at load time, never at call time."""


@dataclass(frozen=True)
class ToolAdaptation:
    """How one server tool is presented to, and called by, the agent."""

    #: agent-facing argument name -> server-facing argument name
    rename: dict[str, str] = field(default_factory=dict)
    #: arguments hidden from the agent entirely
    drop: tuple[str, ...] = ()
    #: arguments the agent must supply, beyond what the server requires
    require: tuple[str, ...] = ()
    #: values filled in when the agent omits an argument
    defaults: dict[str, Any] = field(default_factory=dict)
    #: drop None, "" and [] before the call reaches the server
    strip_empty: bool = False
    #: replacement description, or None to keep the server's
    description: str | None = None
    #: text injected before the call, the PreToolUse hook
    hook: str | None = None

    @classmethod
    def from_raw(cls, tool: str, raw: dict) -> "ToolAdaptation":
        unknown = set(raw) - TOOL_KEYS
        if unknown:
            raise AdapterError(
                f"tool {tool!r}: unknown adapter keys {sorted(unknown)}; "
                f"allowed are {sorted(TOOL_KEYS)}"
            )
        rename = dict(raw.get("rename", {}))
        for agent_name, server_name in rename.items():
            if not isinstance(agent_name, str) or not isinstance(server_name, str):
                raise AdapterError(f"tool {tool!r}: rename must be str -> str")
        if len(set(rename.values())) != len(rename):
            raise AdapterError(f"tool {tool!r}: two agent names map to one server name")
        drop = tuple(raw.get("drop", ()))
        overlap = set(drop) & set(rename)
        if overlap:
            raise AdapterError(
                f"tool {tool!r}: {sorted(overlap)} is both renamed and dropped"
            )
        dropped_without_default = set(drop) - set(raw.get("defaults", {}))
        require = tuple(raw.get("require", ()))
        bad_require = set(require) & dropped_without_default
        if bad_require:
            raise AdapterError(
                f"tool {tool!r}: {sorted(bad_require)} is required but dropped "
                "and has no default, so the agent could never satisfy it"
            )
        return cls(
            rename=rename,
            drop=drop,
            require=require,
            defaults=dict(raw.get("defaults", {})),
            strip_empty=bool(raw.get("strip_empty", False)),
            description=raw.get("description"),
            hook=raw.get("hook"),
        )

    # -- the two transforms ------------------------------------------------
    def agent_schema(self, server_schema: dict) -> dict:
        """The JSON schema the agent sees, derived from the server's."""
        props = dict(server_schema.get("properties", {}))
        required = list(server_schema.get("required", []))

        for name in self.drop:
            props.pop(name, None)
            if name in required:
                required.remove(name)

        reverse = {server: agent for agent, server in self.rename.items()}
        props = {reverse.get(k, k): v for k, v in props.items()}
        required = [reverse.get(r, r) for r in required]

        for name in self.defaults:
            agent_name = reverse.get(name, name)
            if agent_name in required:
                required.remove(agent_name)

        for name in self.require:
            if name not in required and name in props:
                required.append(name)

        out = dict(server_schema)
        out["properties"] = props
        out["required"] = sorted(set(required))
        return out

    def server_arguments(self, agent_arguments: dict) -> dict:
        """The arguments actually sent to the server."""
        args = {self.rename.get(k, k): v for k, v in agent_arguments.items()}
        for name, value in self.defaults.items():
            args.setdefault(name, value)
        for name in self.drop:
            if name not in self.defaults:
                args.pop(name, None)
        if self.strip_empty:
            args = {k: v for k, v in args.items() if v not in (None, "", [], {})}
        return args


@dataclass(frozen=True)
class Composite:
    """A derived tool built from existing server calls. No server changes.

    ``steps`` is a list of ``{"tool": ..., "arguments": {...}}``. An argument
    value of the form ``"$input.<name>"`` is substituted from the composite's
    own input, and ``"$<step index>.<path>"`` from an earlier step's result.
    Resolution lives in the runtime, not here; this dataclass is the contract.
    """

    name: str
    description: str
    steps: tuple[dict, ...]
    input_schema: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict) -> "Composite":
        for key in ("name", "description", "steps"):
            if key not in raw:
                raise AdapterError(f"composite missing required key {key!r}")
        if not raw["steps"]:
            raise AdapterError(f"composite {raw['name']!r} has no steps")
        for i, step in enumerate(raw["steps"]):
            if "tool" not in step:
                raise AdapterError(
                    f"composite {raw['name']!r} step {i} has no 'tool'"
                )
        return cls(
            name=raw["name"],
            description=raw["description"],
            steps=tuple(raw["steps"]),
            input_schema=dict(raw.get("input_schema", {})),
        )


@dataclass(frozen=True)
class AdapterSpec:
    """The whole client-side adaptation, loaded from one JSON file."""

    tools: dict[str, ToolAdaptation] = field(default_factory=dict)
    composites: tuple[Composite, ...] = ()
    hidden: frozenset[str] = frozenset()

    @property
    def is_empty(self) -> bool:
        """True when this adapter changes nothing.

        An empty adapter must leave behaviour byte-identical to the unadapted
        runner. That is the baseline arm, and it is what makes the comparison
        honest, exactly as ``k0`` does for skills.
        """
        return not self.tools and not self.composites and not self.hidden

    @classmethod
    def empty(cls) -> "AdapterSpec":
        return cls()

    @classmethod
    def from_raw(cls, raw: dict) -> "AdapterSpec":
        unknown = set(raw) - SPEC_KEYS
        if unknown:
            raise AdapterError(
                f"unknown adapter keys {sorted(unknown)}; allowed are "
                f"{sorted(SPEC_KEYS)}"
            )
        composites = tuple(Composite.from_raw(c) for c in raw.get("composites", ()))
        names = [c.name for c in composites]
        if len(set(names)) != len(names):
            raise AdapterError("two composites share a name")
        hidden = frozenset(raw.get("hidden", ()))
        for composite in composites:
            if composite.name in hidden:
                raise AdapterError(
                    f"composite {composite.name!r} is also hidden"
                )
        return cls(
            tools={
                name: ToolAdaptation.from_raw(name, body)
                for name, body in raw.get("tools", {}).items()
            },
            composites=composites,
            hidden=hidden,
        )

    @classmethod
    def load(cls, directory: Path | str | None) -> "AdapterSpec":
        """Load ``adapter.json`` from a directory. Absent directory -> empty."""
        if directory is None:
            return cls.empty()
        path = Path(directory).expanduser().resolve() / SPEC_FILENAME
        if not path.is_file():
            raise AdapterError(f"no {SPEC_FILENAME} in {directory}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AdapterError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise AdapterError(f"{path} must hold a JSON object")
        return cls.from_raw(raw)

    def for_tool(self, name: str) -> ToolAdaptation:
        return self.tools.get(name, ToolAdaptation())
