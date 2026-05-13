"""Semantic-Score scorer — similarity without an LLM call.

Uses ``difflib.SequenceMatcher`` over normalised text so the scorer has
no external dependencies and is stable in CI.  A scenario can override
the pass threshold via ``scenario.similarity_threshold`` (default 0.6).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..models import Scenario, ScorerResult
from . import register

_DEFAULT_THRESHOLD = 0.6
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", str(text).strip().lower())


def semantic_similarity(
    scenario: Scenario, answer: str, trajectory_text: str
) -> ScorerResult:
    reference = scenario.characteristic_form or scenario.expected_answer
    if not reference:
        return ScorerResult(
            scorer="semantic_similarity",
            passed=False,
            rationale="scenario has neither characteristic_form nor expected_answer",
        )

    extra = scenario.model_extra or {}
    threshold = float(extra.get("similarity_threshold", _DEFAULT_THRESHOLD))

    score = SequenceMatcher(None, _normalize(reference), _normalize(answer)).ratio()
    passed = score >= threshold
    return ScorerResult(
        scorer="semantic_similarity",
        passed=passed,
        score=round(score, 4),
        rationale=(
            "" if passed else f"similarity {score:.3f} below threshold {threshold}"
        ),
        details={"threshold": threshold, "reference": reference},
    )


register("semantic_similarity", semantic_similarity)
