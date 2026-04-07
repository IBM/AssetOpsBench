"""Load and select SKILL.md files for prompt injection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .ablation import SkillGranularity, SkillStructure


def parse_skill_frontmatter(content: str) -> tuple[dict, str]:
    """Split a SKILL.md file into YAML frontmatter dict and body text.

    Returns ``({}, content)`` when no valid frontmatter is found.
    """
    match = re.match(r"^---\s*\n(.+?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        return {}, content
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, content
    return meta if isinstance(meta, dict) else {}, match.group(2)


def load_skills(skills_dir: Path) -> list[dict]:
    """Load all ``*.md`` files under *skills_dir* and return parsed skill dicts.

    Each dict contains the frontmatter keys plus ``"content"`` (the body) and
    ``"path"`` (the source file).
    """
    skills: list[dict] = []
    if not skills_dir.is_dir():
        return skills
    for path in sorted(skills_dir.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_skill_frontmatter(raw)
        meta["content"] = body.strip()
        meta["path"] = str(path)
        skills.append(meta)
    return skills


def _format_skills(skills: list[dict]) -> str:
    """Format a list of skill dicts as a single Markdown text block."""
    if not skills:
        return ""
    parts = []
    for s in skills:
        header = f"### Skill: {s.get('name', 'unnamed')}"
        if s.get("description"):
            header += f"\n{s['description']}"
        parts.append(f"{header}\n\n{s['content']}")
    return "\n\n---\n\n".join(parts)


def load_skills_as_text(
    skills_dir: Path,
    domain_filter: str | None = None,
    structure: "SkillStructure | None" = None,
    granularity: "SkillGranularity | None" = None,
) -> str:
    """Load skills and format them as a single text block for prompt injection.

    Args:
        skills_dir: Root directory containing ``*.md`` skill files.
        domain_filter: If given, only include skills whose ``domain`` matches.
        structure: Ablation mode for skill building blocks (default: FULL).
        granularity: Filter by skill complexity level (default: all).

    Returns:
        A Markdown-formatted string with each skill separated by ``---``.
    """
    from .ablation import SkillStructure, apply_ablation

    skills = load_skills(skills_dir)
    if domain_filter:
        skills = [s for s in skills if s.get("domain") == domain_filter]
    skills = apply_ablation(
        skills,
        structure=structure or SkillStructure.FULL,
        granularity=granularity,
    )
    return _format_skills(skills)


def select_skills_for_scenario(
    scenario_text: str,
    all_skills: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Select the most relevant skills for a scenario via keyword overlap.

    Each skill dict should have a ``"content"`` key and optionally
    ``"trigger_keywords"`` (list of strings).

    Returns the *top_k* skills sorted by descending keyword overlap with
    *scenario_text*.
    """
    scenario_lower = scenario_text.lower()
    scored: list[tuple[int, dict]] = []
    for skill in all_skills:
        keywords = skill.get("trigger_keywords", [])
        if not keywords:
            # Fall back to words in the skill name / description.
            keywords = (
                skill.get("name", "").replace("-", " ").split()
                + skill.get("description", "").lower().split()
            )
        overlap = sum(1 for kw in keywords if kw.lower() in scenario_lower)
        scored.append((overlap, skill))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:top_k]]
