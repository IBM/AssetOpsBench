"""Offline evaluation harness for AssetOpsBench agent runs.

Consumes saved trajectory files (written by
:func:`observability.persistence.persist_trajectory`) and scenario files
(under ``src/scenarios/``) and emits a structured JSON report combining
graded outcomes with operational metrics.

The shape mirrors conventions from SWE-bench, HELM, and τ-bench:
``run`` (executes the agent — already exists) → ``evaluate`` (this
module) → ``report.json``.  Re-grading from saved trajectories is
first-class.
"""

from .models import (
    AggregateOps,
    EvalReport,
    GradeResult,
    OpsMetrics,
    PersistedTrajectory,
    Scenario,
    ScenarioResult,
    TypeBreakdown,
)

__all__ = [
    "AggregateOps",
    "EvalReport",
    "GradeResult",
    "OpsMetrics",
    "PersistedTrajectory",
    "Scenario",
    "ScenarioResult",
    "TypeBreakdown",
]
