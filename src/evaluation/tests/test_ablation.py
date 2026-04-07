"""Tests for evaluation.ablation."""

from __future__ import annotations

from evaluation.ablation import (
    SkillGranularity,
    SkillStructure,
    apply_ablation,
    filter_skills_by_granularity,
    strip_skill_body,
)

# ── sample skill body ───────────────────────────────────────────────

_BODY = """\
Diagnose chiller vibration anomalies using sensor data and failure mode lookup.

## When to Use
- Vibration alerts or anomaly flags from IoT sensors.

## Prerequisites
- `vibration_analysis` server with `analyze_vibration` tool
- `fmsr` server with `failure_mode_lookup` tool

## Procedure
1. Call `analyze_vibration` with asset_id and time range.
2. If anomaly detected, call `failure_mode_lookup` with symptom.
3. Return recommended action.

## Decision Logic
- If RMS > threshold → escalate to maintenance.
- If RMS normal → close ticket.

## Expected Outputs
- Anomaly flag (bool)
- Recommended maintenance action (str)

## Domain References
- ISO 10816 vibration severity chart
"""


# ── strip_skill_body ────────────────────────────────────────────────


def test_strip_full_returns_unchanged():
    assert strip_skill_body(_BODY, SkillStructure.FULL) == _BODY


def test_strip_procedure_only():
    result = strip_skill_body(_BODY, SkillStructure.PROCEDURE_ONLY)
    assert "## Procedure" in result
    assert "Call `analyze_vibration`" in result
    # Other sections removed
    assert "## When to Use" not in result
    assert "## Prerequisites" not in result
    assert "## Decision Logic" not in result
    assert "## Expected Outputs" not in result
    assert "## Domain References" not in result


def test_strip_tool_list_only():
    result = strip_skill_body(_BODY, SkillStructure.TOOL_LIST_ONLY)
    assert "## Prerequisites" in result
    assert "`vibration_analysis`" in result
    # Other sections removed
    assert "## Procedure" not in result
    assert "## Decision Logic" not in result
    assert "## When to Use" not in result


def test_strip_preserves_preamble():
    result = strip_skill_body(_BODY, SkillStructure.PROCEDURE_ONLY)
    assert result.startswith("Diagnose chiller vibration")


def test_strip_empty_body():
    assert strip_skill_body("", SkillStructure.PROCEDURE_ONLY) == ""


def test_strip_no_sections():
    plain = "Just a plain description with no sections."
    assert strip_skill_body(plain, SkillStructure.PROCEDURE_ONLY) == plain


# ── filter_skills_by_granularity ────────────────────────────────────

_SKILLS = [
    {"name": "s1", "level": "low", "content": "a"},
    {"name": "s2", "level": "mid", "content": "b"},
    {"name": "s3", "level": "high", "content": "c"},
    {"name": "s4", "content": "d"},  # no level
]


def test_filter_none_returns_all():
    assert filter_skills_by_granularity(_SKILLS, None) == _SKILLS


def test_filter_low():
    result = filter_skills_by_granularity(_SKILLS, SkillGranularity.LOW)
    assert len(result) == 1
    assert result[0]["name"] == "s1"


def test_filter_mid():
    result = filter_skills_by_granularity(_SKILLS, SkillGranularity.MID)
    assert len(result) == 1
    assert result[0]["name"] == "s2"


def test_filter_high():
    result = filter_skills_by_granularity(_SKILLS, SkillGranularity.HIGH)
    assert len(result) == 1
    assert result[0]["name"] == "s3"


# ── apply_ablation ──────────────────────────────────────────────────


def test_apply_ablation_full_no_filter():
    out = apply_ablation(_SKILLS, SkillStructure.FULL, None)
    assert out == _SKILLS


def test_apply_ablation_procedure_only():
    skills = [{"name": "x", "level": "low", "content": _BODY}]
    out = apply_ablation(skills, SkillStructure.PROCEDURE_ONLY, None)
    assert len(out) == 1
    assert "## Procedure" in out[0]["content"]
    assert "## Decision Logic" not in out[0]["content"]
    # Original not mutated
    assert "## Decision Logic" in skills[0]["content"]


def test_apply_ablation_combined():
    skills = [
        {"name": "x", "level": "low", "content": _BODY},
        {"name": "y", "level": "high", "content": _BODY},
    ]
    out = apply_ablation(skills, SkillStructure.TOOL_LIST_ONLY, SkillGranularity.LOW)
    assert len(out) == 1
    assert "## Prerequisites" in out[0]["content"]
    assert "## Procedure" not in out[0]["content"]
