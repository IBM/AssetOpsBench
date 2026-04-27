"""Pydantic models for the offline evaluation pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Scenario(BaseModel):
    """One evaluation scenario.

    Mirrors the on-disk shape under ``src/scenarios/`` and is permissive
    via ``extra='allow'`` so domain-specific fields (e.g. category,
    characteristic_form) survive the round-trip.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    text: str
    type: str = ""
    category: str = ""
    characteristic_form: str | None = None
    expected_answer: str | None = None
    grading_method: str | None = None

    @classmethod
    def from_raw(cls, raw: dict) -> "Scenario":
        d = dict(raw)
        if "id" in d:
            d["id"] = str(d["id"])
        return cls.model_validate(d)


class PersistedTrajectory(BaseModel):
    """Record written by ``observability.persistence.persist_trajectory``."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    scenario_id: str | None = None
    runner: str
    model: str
    question: str
    answer: str
    trajectory: Any = None

    @classmethod
    def from_raw(cls, raw: dict) -> "PersistedTrajectory":
        d = dict(raw)
        if d.get("scenario_id") is not None:
            d["scenario_id"] = str(d["scenario_id"])
        return cls.model_validate(d)


class OpsMetrics(BaseModel):
    """Per-task operational metrics derived from a trajectory."""

    turn_count: int = 0
    tool_call_count: int = 0
    unique_tools: list[str] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float | None = None
    est_cost_usd: float | None = None


class GradeResult(BaseModel):
    grading_method: str
    passed: bool
    score: float = 0.0
    rationale: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    scenario_id: str
    scenario_type: str = ""
    runner: str
    model: str
    question: str
    answer: str
    grade: GradeResult
    ops: OpsMetrics


class AggregateOps(BaseModel):
    tokens_in_total: int = 0
    tokens_out_total: int = 0
    duration_ms_p50: float | None = None
    duration_ms_p95: float | None = None
    tool_calls_total: int = 0
    est_cost_usd_total: float | None = None


class TypeBreakdown(BaseModel):
    total: int = 0
    passed: int = 0
    pass_rate: float = 0.0


class EvalReport(BaseModel):
    generated_at: str
    runners: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    totals: dict[str, Any] = Field(default_factory=dict)
    by_scenario_type: dict[str, TypeBreakdown] = Field(default_factory=dict)
    ops: AggregateOps = Field(default_factory=AggregateOps)
    results: list[ScenarioResult] = Field(default_factory=list)
