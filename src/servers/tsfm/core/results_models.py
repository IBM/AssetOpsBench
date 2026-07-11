from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class ErrorResult(BaseModel):
    error: str


# ---- discovery / catalog (no bulk data) ----
class TasksResult(BaseModel):
    tasks: List[dict]


class ComponentsResult(BaseModel):
    task: str
    components: dict


class CandidatesResult(BaseModel):
    task_id: str
    candidates: List[dict]


class ModelsResult(BaseModel):
    models: List[dict]


class FeaturesResult(BaseModel):
    features: List[dict]


class ComponentResult(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    component_id: str


# ---- evidence / learn ----
class ProfileResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: str
    n_observations: int
    n_channels: int
    dominant_period: Optional[int] = None


class FeatureSelectionResult(BaseModel):
    selected: List[str]
    lookback: int
    reference: str
    scorers: List[str]
    detail_file: str  # file pointer to the full per-scorer scores


class CharacterizeResult(BaseModel):
    """Pattern EVIDENCE for a series: per-group state+rate phases + bivariate relations + a
    shape-only NL summary. Domain-agnostic; full structured evidence at evidence_file.
    """

    model_config = ConfigDict(extra="allow")
    status: str
    summary: str  # shape-only NL description (never names a fault)
    n_observations: int
    evidence_file: str  # file pointer to the full pattern object
    message: str


# ---- compose + run (file pointers) ----
class RecipeResult(BaseModel):
    """A recipe run. Forecasting carries metric+backtest_score; anomaly carries n_anomalies+
    n_observations (extra-allowed) — run_recipe dispatches by recipe.task."""

    model_config = ConfigDict(extra="allow")
    status: str
    run_id: str
    results_file: (
        str  # file pointer to the run record (forecast/intervals OR anomaly labels)
    )
    training_regime: str
    message: str
    metric: Optional[str] = None
    backtest_score: Optional[float] = None


class TabularResult(BaseModel):
    status: str
    run_id: str
    results_file: str
    task: str
    metric: str
    cv_score: Any
    n_features: int
    message: str


class DataQualityResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    cleaned_file: str  # file pointer to the cleaned series
    rows_in: int
    rows_out: int
    message: str


class PlanResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    results_file: str
    message: str


class EvaluateResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    results_file: str
    message: str


# ---- write-back / results ----
class RegisterResult(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    status: str
    id: str


class CardResult(BaseModel):
    """A single catalog card returned by update / deprecate / new_version / register_finetuned."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class LineageResult(BaseModel):
    """A card's evolution chain (ancestors + descendants / supersedes links)."""

    model_config = ConfigDict(extra="allow")


class ResultsListResult(BaseModel):
    results: List[dict]


class ResultRecord(BaseModel):
    """A single stored result-table record (task-polymorphic; extra fields allowed)."""

    model_config = ConfigDict(extra="allow")


class RunRecord(BaseModel):
    """A single stored run record (recipe run / plan; extra fields allowed)."""

    model_config = ConfigDict(extra="allow")
    run_id: str


class RunsResult(BaseModel):
    runs: List[dict]
    plans: List[dict]


# ---- evolve (AlphaEvolve loop) ----
class EvolveAskResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    task: str
    kind: str


class EvolveTellResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    accepted: bool


class EvolveBestResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    task: str

class ExtractResult(BaseModel):
    n_windows: int
    window: Optional[int]
    columns: List[str]                 # feature columns: '<channel>.<extractor>' (or '<extractor>')
    features: List[List[float]]        # n_windows rows x columns (whole-series => 1 row)
    message: str

class ExtractorNamesResult(BaseModel):
    count: int
    extractors: List[str]


class ExtractorDescription(BaseModel):
    extractor_name: str
    description: Optional[str]


class DescribeExtractorsResult(BaseModel):
    extractors: List[ExtractorDescription]
    unknown: List[str]
    message: str