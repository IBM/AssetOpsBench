"""Pure deterministic graders — no LLM, no network."""

from __future__ import annotations

import math

from ..models import GradeResult, Scenario
from . import register


def exact_string_match(
    scenario: Scenario, answer: str, trajectory_text: str
) -> GradeResult:
    expected = scenario.expected_answer
    if expected is None:
        return GradeResult(
            grading_method="exact_string_match",
            passed=False,
            score=0.0,
            rationale="scenario has no expected_answer",
        )

    a = str(answer).strip().lower()
    e = str(expected).strip().lower()
    passed = a == e
    return GradeResult(
        grading_method="exact_string_match",
        passed=passed,
        score=1.0 if passed else 0.0,
        rationale="" if passed else f"expected {expected!r}, got {answer!r}",
        details={"expected": expected, "actual": answer},
    )


def numeric_match(
    scenario: Scenario, answer: str, trajectory_text: str
) -> GradeResult:
    expected_raw = scenario.expected_answer
    extra = scenario.model_extra or {}
    tolerance = float(extra.get("tolerance", 1e-6))

    if expected_raw is None:
        return GradeResult(
            grading_method="numeric_match",
            passed=False,
            rationale="scenario has no expected_answer",
        )

    try:
        a = float(answer)
        e = float(expected_raw)
    except (TypeError, ValueError) as err:
        return GradeResult(
            grading_method="numeric_match",
            passed=False,
            rationale=f"could not parse numbers: {err}",
            details={"expected": expected_raw, "actual": answer},
        )

    passed = math.isclose(a, e, rel_tol=tolerance, abs_tol=tolerance)
    return GradeResult(
        grading_method="numeric_match",
        passed=passed,
        score=1.0 if passed else 0.0,
        rationale="" if passed else f"|{a} - {e}| > tol={tolerance}",
        details={"expected": e, "actual": a, "tolerance": tolerance},
    )


register("exact_string_match", exact_string_match)
register("numeric_match", numeric_match)
