"""Tests for evaluation.scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


from evaluation.scorer import score_result, _deterministic_score


@dataclass
class _FakeScenario:
    id: str = "test-001"
    text: str = "What sensors are on Chiller 6?"
    expected_result: Any = None
    characteristic_form: Optional[str] = None
    deterministic: bool = False


def test_deterministic_exact_match():
    s = _FakeScenario(deterministic=True, expected_result="MAIN")
    result = score_result(s, answer="MAIN")
    assert result["passed"] is True
    assert result["method"] == "exact"


def test_deterministic_case_insensitive():
    s = _FakeScenario(deterministic=True, expected_result="MAIN")
    result = score_result(s, answer="main")
    assert result["passed"] is True


def test_deterministic_numeric_match():
    s = _FakeScenario(deterministic=True, expected_result=42.0)
    result = score_result(s, answer="42.0")
    assert result["passed"] is True
    assert result["method"] == "numeric"


def test_deterministic_fail():
    s = _FakeScenario(deterministic=True, expected_result="MAIN")
    result = score_result(s, answer="SECONDARY")
    assert result["passed"] is False
    assert result["method"] == "deterministic_fail"


def test_non_deterministic_without_judge():
    # Without the reactxen judge, non-deterministic scenarios are unscored
    s = _FakeScenario(deterministic=False, expected_result=None)
    result = score_result(s, answer="Some answer")
    assert result["method"] in ("llm_judge", "unscored")


def test_deterministic_score_helper_numeric():
    result = _deterministic_score("3.14", 3.14)
    assert result is not None
    assert result["passed"] is True


def test_deterministic_score_helper_no_match():
    result = _deterministic_score("apples", "oranges")
    assert result is None
