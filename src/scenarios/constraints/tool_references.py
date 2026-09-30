"""Recognize explicit MCP references without interpreting ordinary prose as tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re


_IDENTIFIER = re.compile(
    r"(?<![a-z0-9_.])(?:(?P<namespace>[a-z_][a-z0-9_]*)\.)?"
    r"(?P<name>[a-z_][a-z0-9_]*)(?![a-z0-9_]|\.[a-z0-9_])",
    re.IGNORECASE,
)
_TOOL_LABEL_BEFORE = re.compile(r"\b(?:tool|function|api)\s+(?:(?:named|called)\s+)?$", re.IGNORECASE)
_TOOL_LABEL_AFTER = re.compile(r"^\s+(?:tool|function|api)\b", re.IGNORECASE)
# Recognized only to flag obsolete references; these do not alias or register tools.
_LEGACY_IOT_NAMES = frozenset({"get_sites", "get_assets", "get_sensors", "get_history"})


@dataclass(frozen=True)
class ToolReference:
    spelling: str
    canonical: str | None
    error: str | None = None


def find_tool_references(
    text: str,
    tool_names_by_focus: Mapping[str, tuple[str, ...]] | None,
) -> list[ToolReference]:
    """Resolve qualified, called, code-quoted, or explicitly labelled references.

    Bare registered snake_case names remain explicit identifiers. Common words
    such as ``sites``, ``assets``, and ``history`` require reference syntax.
    Unqualified names must resolve to exactly one server. Unknown code-quoted
    sensor/asset identifiers are left alone; unknown names qualified by a
    discovered server are reported as invalid references.
    """
    if not tool_names_by_focus:
        return []

    registry = {
        focus.lower(): {name.lower() for name in names}
        for focus, names in tool_names_by_focus.items()
    }
    owners: dict[str, list[str]] = {}
    for focus, names in registry.items():
        for name in names:
            owners.setdefault(name, []).append(focus)

    references: list[ToolReference] = []
    seen: set[str] = set()
    for match in _IDENTIFIER.finditer(text):
        name = match["name"].lower()
        namespace = (match["namespace"] or "").lower()
        spelling = match[0].lower()
        before, after = text[:match.start()], text[match.end():]
        called = after.startswith("(")
        code_quoted = before.endswith("`") and after.startswith("`")
        labelled = bool(_TOOL_LABEL_BEFORE.search(before) or _TOOL_LABEL_AFTER.match(after))
        known = name in owners or name in _LEGACY_IOT_NAMES

        if namespace:
            explicit = namespace in registry or (known and (called or code_quoted or labelled))
        else:
            explicit = known and (called or "_" in name or code_quoted or labelled)
        if not explicit or spelling in seen:
            continue
        seen.add(spelling)

        if namespace:
            candidates = [namespace] if name in registry.get(namespace, set()) else []
        else:
            candidates = owners.get(name, [])

        if len(candidates) == 1:
            references.append(ToolReference(spelling, f"{candidates[0]}.{name}"))
        elif len(candidates) > 1:
            choices = ", ".join(f"{focus}.{name}" for focus in sorted(candidates))
            references.append(ToolReference(
                spelling, None,
                f"ambiguous MCP tool reference '{spelling}'; qualify it as one of: {choices}",
            ))
        else:
            references.append(ToolReference(
                spelling, None,
                f"unknown MCP tool reference '{spelling}'; use a qualified name from the available tools",
            ))
    return references
