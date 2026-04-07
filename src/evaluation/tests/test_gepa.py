"""Tests for evaluation.gepa_refiner — unit tests that don't require dspy."""

from __future__ import annotations

import pytest

from evaluation.gepa_refiner import _score_skill_quality, _score_skill_pass_rate


# ── _score_skill_quality (structural heuristic) ────────────────────

_GOOD_SKILL = """\
## When to Use
Vibration alerts from rotating machinery.

## Prerequisites
- `vibration_analysis` server with `compute_fft_spectrum` tool

## Procedure
1. call compute_fft_spectrum(asset_id=X)
2. call assess_vibration_severity(spectrum=result)

## Decision Logic
- If severity > 6.3 → escalate

## Expected Outputs
{"severity": "float", "action": "string"}
"""

_BARE_SKILL = "Just a plain text skill with no sections."


def test_score_quality_full_skill():
    score = _score_skill_quality(_GOOD_SKILL, "")
    # Has all 6 markers → 1.0
    assert score == 1.0


def test_score_quality_bare_skill():
    score = _score_skill_quality(_BARE_SKILL, "")
    assert score == 0.0


def test_score_quality_partial():
    partial = "## Procedure\n1. call foo()\n## Decision Logic\nif x > 1"
    score = _score_skill_quality(partial, "")
    # Has "## Procedure", "## Decision Logic", "call " → 3/6
    assert abs(score - 0.5) < 1e-9


# ── _score_skill_pass_rate ──────────────────────────────────────────


def test_pass_rate_metric_all_pass():
    results = [
        {"scenario_id": "s1", "passed": True},
        {"scenario_id": "s2", "passed": True},
    ]
    score = _score_skill_pass_rate("any skill text", results)
    assert score == 1.0


def test_pass_rate_metric_all_fail():
    results = [
        {"scenario_id": "s1", "passed": False},
        {"scenario_id": "s2", "passed": False},
    ]
    score = _score_skill_pass_rate("any skill text", results)
    assert score == 0.0


def test_pass_rate_metric_mixed():
    results = [
        {"scenario_id": "s1", "passed": True},
        {"scenario_id": "s2", "passed": False},
        {"scenario_id": "s3", "passed": True},
    ]
    score = _score_skill_pass_rate("any skill text", results)
    assert abs(score - 2 / 3) < 1e-9


def test_pass_rate_metric_empty():
    score = _score_skill_pass_rate("any skill text", [])
    assert score == 0.0


# ── SkillRefiner / run_gepa_refinement require dspy ─────────────────


def test_skill_refiner_requires_dspy():
    """SkillRefiner raises ImportError when dspy is not available."""
    # We can't easily mock the import, but we can verify the guard exists.
    from evaluation.gepa_refiner import _HAS_DSPY, _ensure_dspy

    if not _HAS_DSPY:
        with pytest.raises(ImportError, match="dspy is required"):
            _ensure_dspy()
