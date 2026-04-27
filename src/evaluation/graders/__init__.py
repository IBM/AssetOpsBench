"""Pluggable grader registry.

Each grader is a callable taking ``(scenario, answer, trajectory_text)``
and returning a :class:`~evaluation.models.GradeResult`.  Registration
happens via :func:`register`; the CLI looks up graders by name from
``scenario.grading_method`` (falling back to a CLI-supplied default).
"""

from __future__ import annotations

from typing import Callable

from ..models import GradeResult, Scenario

Grader = Callable[[Scenario, str, str], GradeResult]

_REGISTRY: dict[str, Grader] = {}


def register(name: str, grader: Grader) -> None:
    _REGISTRY[name] = grader


def get(name: str) -> Grader:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown grader {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


from . import deterministic  # noqa: E402,F401  — register-on-import
