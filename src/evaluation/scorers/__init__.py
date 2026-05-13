"""Pluggable scorer registry.

Each scorer is a callable taking ``(scenario, answer, trajectory_text)``
and returning a :class:`~evaluation.models.ScorerResult`.  The vocabulary
follows MLflow's evaluation concept: an ``Evaluator`` orchestrates one
or more ``Scorer`` s; scorers fall into three families:

* **Code-Based** — deterministic, no model required (string/numeric
  matchers in :mod:`evaluation.scorers.code_based`).
* **LLM-As-Judge** — model-scored against a rubric
  (:mod:`evaluation.scorers.llm_judge`).
* **Semantic-Score** — similarity-based, no model call
  (:mod:`evaluation.scorers.semantic`).
"""

from __future__ import annotations

from typing import Callable

from ..models import Scenario, ScorerResult

Scorer = Callable[[Scenario, str, str], ScorerResult]

_REGISTRY: dict[str, Scorer] = {}


def register(name: str, scorer: Scorer) -> None:
    _REGISTRY[name] = scorer


def get(name: str) -> Scorer:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown scorer {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


# Code-Based and Semantic-Score families ship as skeletons — their
# modules are importable but register no scorers until an
# implementation is filled in.  LLM-As-Judge is registered explicitly
# via :func:`evaluation.scorers.llm_judge.install`.
from . import code_based  # noqa: E402,F401
from . import semantic  # noqa: E402,F401
