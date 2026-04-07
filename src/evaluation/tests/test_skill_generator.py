"""Tests for evaluation.skill_generator (non-LLM helpers)."""

from __future__ import annotations

from evaluation.skill_generator import _extract_skill_name


def test_extract_skill_name_valid():
    md = "---\nname: bearing-fft-analysis\ndescription: test\n---\n# Body"
    assert _extract_skill_name(md) == "bearing-fft-analysis"


def test_extract_skill_name_missing():
    assert _extract_skill_name("# No frontmatter at all") == "unnamed-skill"


def test_extract_skill_name_bad_yaml():
    md = "---\n: :\n---\nbody"
    assert _extract_skill_name(md) == "unnamed-skill"
