"""Auto-generate executable skills from MCP server tool descriptions."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import yaml

from llm import LLMBackend

_log = logging.getLogger(__name__)

META_SKILL_PROMPT = """\
You are an expert in industrial operations and maintenance. Your task is to \
generate an executable skill specification in SKILL.md format.

## Available MCP Tools

{server_descriptions}

## Target Domain: {domain}
## Domain Context: {domain_context}

## Instructions

Generate a SKILL.md file that:
1. Has a **descriptive kebab-case name** reflecting the concrete task \
(e.g. ``bearing-fft-analysis``, NOT ``skill-1``).
2. Has a **description** field that includes:
   - WHAT the skill does.
   - WHEN to use it (specific trigger phrases a user would say).
   - Key capabilities or file types handled.
   Example: "End-to-end bearing fault diagnosis from raw vibration data. \
Use when user asks to 'diagnose vibration', 'analyse bearing fault', or \
'check asset vibration health'. Covers FFT spectrum, envelope analysis, \
and ISO 10816 severity."
3. Lists PREREQUISITE MCP tools (must be from the available tools above).
4. Provides a STEP-BY-STEP PROCEDURE with explicit tool calls: \
`call tool_name(param=value)`.
5. Includes DECISION LOGIC with if-then branching based on intermediate results.
6. Specifies EXPECTED OUTPUTS as a JSON schema.
7. References relevant DOMAIN STANDARDS (ISO, FMEA, etc.).

## Output Format

Return ONLY the SKILL.md content in this exact format:
---
name: {{descriptive-kebab-case-name}}
description: >-
  {{What it does. Use when user asks to "trigger phrase 1",
  "trigger phrase 2", or "trigger phrase 3". Covers capability1,
  capability2.}}
domain: {domain}
required_mcp_tools:
  - {{tool_name_1}}
  - {{tool_name_2}}
level: {{low|mid|high}}
trigger_keywords:
  - {{keyword1}}
  - {{keyword2}}
---
# {{Skill Name}}
## When to Use
[Trigger conditions — be specific about task patterns that match]
## Prerequisites
[Required MCP tools and data availability]
## Procedure
[Numbered steps with EXPLICIT tool calls]
## Decision Logic
[If-then branching based on intermediate results]
## Expected Outputs
[JSON schema]
## Domain References
[Standards, methodologies]
"""

DOMAIN_CONTEXTS: dict[str, str] = {
    "vibration": (
        "Vibration analysis for rotating machinery: bearing fault detection, "
        "FFT spectrum analysis, envelope spectrum, ISO 10816 severity assessment."
    ),
    "iot": (
        "IoT sensor monitoring: multi-sensor data retrieval, anomaly detection, "
        "asset health scoring across sites."
    ),
    "fmsr": (
        "Failure Mode and Symptom Reasoning: FMEA-based root cause analysis, "
        "failure mode mapping, sensor-to-failure correlations."
    ),
    "tsfm": (
        "Time Series Foundation Models: anomaly detection, trend forecasting, "
        "predictive maintenance scheduling from sensor data."
    ),
    "cross-domain": (
        "Cross-domain asset assessment combining vibration, IoT, FMSR, "
        "and TSFM tools for comprehensive asset health evaluation."
    ),
}

# Number of skills to generate per domain, with their target level.
_SKILLS_PER_DOMAIN: dict[str, list[str]] = {
    "vibration": ["low", "mid", "high"],
    "iot": ["low", "mid"],
    "fmsr": ["low", "mid"],
    "tsfm": ["low", "mid"],
    "cross-domain": ["high"],
}


def _format_server_descriptions(descs: dict[str, str]) -> str:
    return "\n\n".join(f"### {name}\n{body}" for name, body in descs.items())


def _extract_skill_name(skill_md: str) -> str:
    """Extract the ``name`` field from SKILL.md frontmatter.

    Falls back to ``"unnamed-skill"`` when parsing fails.
    """
    match = re.match(r"^---\s*\n(.+?)\n---", skill_md, re.DOTALL)
    if match:
        try:
            meta = yaml.safe_load(match.group(1))
            if isinstance(meta, dict) and meta.get("name"):
                return str(meta["name"])
        except Exception:  # noqa: BLE001
            pass
    return "unnamed-skill"


def generate_skills_for_domain(
    domain: str,
    llm: LLMBackend,
    server_descriptions: dict[str, str],
    levels: list[str] | None = None,
) -> list[str]:
    """Generate SKILL.md texts for *domain* using the LLM.

    *levels* controls how many skills are produced and at what complexity:
    ``["low", "mid", "high"]`` generates three skills, one per level.
    """
    if levels is None:
        levels = _SKILLS_PER_DOMAIN.get(domain, ["low", "mid"])
    prompt = META_SKILL_PROMPT.format(
        server_descriptions=_format_server_descriptions(server_descriptions),
        domain=domain,
        domain_context=DOMAIN_CONTEXTS.get(domain, ""),
    )
    skills: list[str] = []
    for i, level in enumerate(levels, 1):
        suffix = (
            f"\n\nGenerate skill {i} of {len(levels)} for the {domain} domain. "
            f"This skill MUST have `level: {level}` in its frontmatter. "
            f"Level meaning — low: single-tool invocation, "
            f"mid: multi-tool workflow, high: cross-domain orchestration. "
            f"The `name` field must be a descriptive kebab-case identifier "
            f"reflecting the concrete task (e.g. bearing-fft-analysis), "
            f"NOT a generic label like skill-1."
        )
        skill_md = llm.generate(prompt + suffix)
        skills.append(skill_md)
    return skills


def generate_all_skills(
    llm: LLMBackend,
    server_descriptions: dict[str, str],
    output_dir: Path = Path("src/skills/generated"),
) -> dict[str, list[Path]]:
    """Generate skills for every domain and persist them to *output_dir*.

    Also updates ``src/skills/manifest.json`` with the generated skill paths.

    Returns a mapping of domain -> list of written file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_paths: dict[str, list[Path]] = {}

    for domain in DOMAIN_CONTEXTS:
        levels = _SKILLS_PER_DOMAIN.get(domain, ["low", "mid"])
        skills = generate_skills_for_domain(domain, llm, server_descriptions, levels)
        domain_dir = output_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for skill_md in skills:
            # Each skill goes into its own kebab-case folder as SKILL.md.
            skill_name = _extract_skill_name(skill_md)
            skill_dir = domain_dir / skill_name
            skill_dir.mkdir(parents=True, exist_ok=True)
            path = skill_dir / "SKILL.md"
            path.write_text(skill_md, encoding="utf-8")
            _log.info("Wrote skill %s", path)
            paths.append(path)
        all_paths[domain] = paths

    # Update manifest
    manifest_path = Path("src/skills/manifest.json")
    entries = []
    for domain, paths in all_paths.items():
        for p in paths:
            entries.append({"domain": domain, "path": str(p)})
    manifest = {
        "version": "1.0.0",
        "description": "Auto-generated skills registry for AssetOpsBench evaluation.",
        "skills": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _log.info("Updated manifest at %s with %d skills.", manifest_path, len(entries))

    return all_paths
