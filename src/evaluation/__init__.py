"""Offline evaluation harness for AssetOpsBench agent runs.

Consumes saved trajectory files (written by
:func:`observability.persistence.persist_trajectory`) and scenario files
(under ``src/scenarios/``) and emits a structured JSON report combining
scored outcomes with operational metrics.

The shape mirrors conventions from SWE-bench, HELM, and τ-bench:
``run`` (executes the agent — already exists) → ``evaluate`` (this
module) → ``report.json``.  Re-scoring from saved trajectories is
first-class.

The evaluation concept follows MLflow's vocabulary: an
:class:`Evaluator` orchestrates one or more :data:`Scorer` callables
(:class:`ScorerResult` records the outcome).  Scorers fall into three
families — Code-Based, LLM-As-Judge, and Semantic-Score — registered
under :mod:`evaluation.scorers`.
"""

from .evaluator import Evaluator
from .models import (
    AggregateOps,
    EvalReport,
    OpsMetrics,
    PersistedTrajectory,
    Scenario,
    ScenarioResult,
    ScorerResult,
    TypeBreakdown,
)
from .scorers import Scorer

__all__ = [
    "AggregateOps",
    "EvalReport",
    "Evaluator",
    "OpsMetrics",
    "PersistedTrajectory",
    "Scenario",
    "ScenarioResult",
    "Scorer",
    "ScorerResult",
    "TypeBreakdown",
]
