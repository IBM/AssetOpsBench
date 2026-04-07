"""Tests for evaluation.skill_loader."""

from __future__ import annotations

import textwrap


from evaluation.skill_loader import (
    load_skills,
    load_skills_as_text,
    parse_skill_frontmatter,
    select_skills_for_scenario,
)


# ── parse_skill_frontmatter ──────────────────────────────────────────────────

_VALID_SKILL = textwrap.dedent("""\
    ---
    name: vibration-diagnosis
    description: End-to-end bearing fault diagnosis
    domain: vibration
    required_mcp_tools:
      - compute_fft_spectrum
      - diagnose_vibration
    level: mid
    trigger_keywords:
      - vibration
      - bearing
      - fault
    ---
    # Vibration Diagnosis
    ## When to Use
    When a vibration alert is raised.
    ## Procedure
    1. call list_vibration_sensors()
    2. call get_vibration_data(sensor_id=...)
""")


def test_parse_valid_frontmatter():
    meta, body = parse_skill_frontmatter(_VALID_SKILL)
    assert meta["name"] == "vibration-diagnosis"
    assert meta["domain"] == "vibration"
    assert "compute_fft_spectrum" in meta["required_mcp_tools"]
    assert "# Vibration Diagnosis" in body


def test_parse_no_frontmatter():
    meta, body = parse_skill_frontmatter("# Just a heading\nSome text.")
    assert meta == {}
    assert body == "# Just a heading\nSome text."


def test_parse_invalid_yaml():
    bad = "---\n: :\n---\nbody"
    meta, body = parse_skill_frontmatter(bad)
    # Falls back gracefully
    assert isinstance(meta, dict)


# ── load_skills ──────────────────────────────────────────────────────────────


def test_load_skills_from_dir(tmp_path):
    d = tmp_path / "skills" / "vibration"
    d.mkdir(parents=True)
    (d / "skill-1.md").write_text(_VALID_SKILL, encoding="utf-8")
    skills = load_skills(tmp_path / "skills")
    assert len(skills) == 1
    assert skills[0]["name"] == "vibration-diagnosis"
    assert "# Vibration Diagnosis" in skills[0]["content"]


def test_load_skills_empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert load_skills(d) == []


def test_load_skills_nonexistent(tmp_path):
    assert load_skills(tmp_path / "nope") == []


# ── load_skills_as_text ──────────────────────────────────────────────────────


def test_load_skills_as_text(tmp_path):
    d = tmp_path / "skills" / "vibration"
    d.mkdir(parents=True)
    (d / "skill-1.md").write_text(_VALID_SKILL, encoding="utf-8")
    text = load_skills_as_text(tmp_path / "skills")
    assert "vibration-diagnosis" in text
    assert "# Vibration Diagnosis" in text


def test_load_skills_as_text_domain_filter(tmp_path):
    d = tmp_path / "skills" / "iot"
    d.mkdir(parents=True)
    iot_skill = _VALID_SKILL.replace("vibration", "iot").replace("Vibration", "IoT")
    (d / "skill-1.md").write_text(iot_skill, encoding="utf-8")
    # Create a vibration skill too
    dv = tmp_path / "skills" / "vibration"
    dv.mkdir(parents=True)
    (dv / "skill-1.md").write_text(_VALID_SKILL, encoding="utf-8")

    text = load_skills_as_text(tmp_path / "skills", domain_filter="iot")
    assert "iot" in text.lower()
    # Vibration should be filtered out
    assert "vibration-diagnosis" not in text


# ── select_skills_for_scenario ───────────────────────────────────────────────


def test_select_skills_keyword_ranking():
    skills = [
        {"name": "a", "content": "", "trigger_keywords": ["vibration", "bearing"]},
        {"name": "b", "content": "", "trigger_keywords": ["forecast", "anomaly"]},
        {"name": "c", "content": "", "trigger_keywords": ["work order", "maintenance"]},
    ]
    selected = select_skills_for_scenario(
        "diagnose the vibration bearing fault", skills, top_k=2
    )
    assert len(selected) == 2
    assert selected[0]["name"] == "a"


def test_select_skills_fallback_to_name():
    skills = [
        {"name": "vibration-diagnosis", "content": "", "description": "bearing fault"},
    ]
    selected = select_skills_for_scenario("vibration problem", skills, top_k=1)
    assert selected[0]["name"] == "vibration-diagnosis"
