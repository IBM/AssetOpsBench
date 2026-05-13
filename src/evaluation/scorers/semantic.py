"""Semantic-Score scorer — similarity-based scoring without an LLM call.

Skeleton only — fill in the implementation (e.g. embedding cosine, BLEU,
sentence-transformers, or difflib ratio) and re-register with the
scorer registry before use.
"""

from __future__ import annotations

from ..models import Scenario, ScorerResult


def semantic_similarity(
    scenario: Scenario, answer: str, trajectory_text: str
) -> ScorerResult:
    raise NotImplementedError
