"""Skill ablation support.

Provides configuration and transforms to strip building blocks from
SKILL.md files, enabling controlled ablation studies on:

1. **Skill structure** – full skill vs procedure-only vs tool-list-only.
2. **Skill granularity** – filter skills by complexity level
   (single-tool / multi-tool / cross-domain).
"""

from __future__ import annotations

import re
from enum import Enum


class SkillStructure(Enum):
    """Which building blocks to retain in a skill."""

    FULL = "full"
    PROCEDURE_ONLY = "procedure_only"
    TOOL_LIST_ONLY = "tool_list_only"


class SkillGranularity(Enum):
    """Skill complexity level (maps to ``level`` frontmatter field)."""

    LOW = "low"  # single-tool skills
    MID = "mid"  # multi-tool / multi-step skills
    HIGH = "high"  # cross-domain, complex reasoning


# Canonical section headings expected in SKILL.md bodies.
_ALL_SECTIONS = {
    "When to Use",
    "Prerequisites",
    "Procedure",
    "Decision Logic",
    "Expected Outputs",
    "Domain References",
}

_KEEP_SECTIONS: dict[SkillStructure, set[str]] = {
    SkillStructure.FULL: _ALL_SECTIONS,
    SkillStructure.PROCEDURE_ONLY: {"Procedure"},
    SkillStructure.TOOL_LIST_ONLY: {"Prerequisites"},
}


def strip_skill_body(body: str, structure: SkillStructure) -> str:
    """Return *body* with only the sections allowed by *structure*.

    Sections are identified by ``## <Heading>`` lines.  Content before the
    first section heading is always kept (e.g. a one-line description).
    """
    if structure is SkillStructure.FULL:
        return body

    keep = _KEEP_SECTIONS[structure]
    # Split into (heading, content) blocks.
    parts = re.split(r"(?m)^(## .+)$", body)
    # parts[0] is the text before the first heading, then alternating
    # heading / content pairs.
    result_parts: list[str] = [parts[0]]  # preamble always kept
    i = 1
    while i < len(parts) - 1:
        heading_line = parts[i]
        content = parts[i + 1]
        heading_text = heading_line.lstrip("# ").strip()
        if heading_text in keep:
            result_parts.append(heading_line)
            result_parts.append(content)
        i += 2

    return "".join(result_parts).strip()


def filter_skills_by_granularity(
    skills: list[dict],
    granularity: SkillGranularity | None,
) -> list[dict]:
    """Keep only skills whose ``level`` frontmatter matches *granularity*.

    When *granularity* is ``None`` all skills are returned.
    """
    if granularity is None:
        return skills
    return [s for s in skills if s.get("level") == granularity.value]


def apply_ablation(
    skills: list[dict],
    structure: SkillStructure = SkillStructure.FULL,
    granularity: SkillGranularity | None = None,
) -> list[dict]:
    """Apply both structure stripping and granularity filtering.

    Returns a **new** list of skill dicts with modified ``content`` values.
    The original dicts are not mutated.
    """
    filtered = filter_skills_by_granularity(skills, granularity)
    if structure is SkillStructure.FULL:
        return filtered
    return [
        {**s, "content": strip_skill_body(s.get("content", ""), structure)}
        for s in filtered
    ]
