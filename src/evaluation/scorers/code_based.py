"""Code-Based scorers — deterministic, no LLM, no network.

Skeleton only — fill in the implementations and re-register with the
scorer registry before use.
"""

from __future__ import annotations

from ..models import Scenario, ScorerResult


def exact_string_match(
    scenario: Scenario, answer: str, trajectory_text: str
) -> ScorerResult:
    raise NotImplementedError


def numeric_match(
    scenario: Scenario, answer: str, trajectory_text: str
) -> ScorerResult:
    raise NotImplementedError
