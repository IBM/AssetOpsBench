from __future__ import annotations

from typing import Any, List, Optional

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
    message: Optional[str] = None


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
    n_observations (extra-allowed); run_recipe dispatches by recipe.task."""

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

class FeatureNamesResult(BaseModel):
    count: int
    kind: Optional[str]                 # filter applied: "extractor" | "transform" | None (all)
    features: List[str]


class FeatureDescription(BaseModel):
    feature_id: str
    kind: str                           # "extractor" | "transform"
    name: Optional[str]
    description: Optional[str]


class DescribeFeaturesResult(BaseModel):
    features: List[FeatureDescription]
    unknown: List[str]
    message: str

class FeatureTemplateResult(BaseModel):
    required_fields: List[str]
    optional_fields: List[str]
    interface_choices: List[str]
    code_skeleton: str
    validity_rules: List[str]
    example: dict

class FeatureCountResult(BaseModel):
    extractors: int
    transforms: int
    total: int


class ModelCountResult(BaseModel):
    total: int
    by_task: dict

class ResolveResult(BaseModel):
    model_id: str
    resolvable: bool
    reason: str
    sktime_class: Optional[str] = None
    training_regime: Optional[str] = None
    weights_from: Optional[str] = None

class DomainsResult(BaseModel):
    domains: dict


class HfStatsResult(BaseModel):
    """HuggingFace popularity lookup for one model (read-only)."""
    model_id: Optional[str] = None
    hf_repo: str
    downloads: Optional[int] = None
    likes: Optional[int] = None


class GiftStatusResult(BaseModel):
    """GIFT-Eval leaderboard standing lookup for one model (read-only)."""
    model_id: Optional[str] = None
    key: str
    found: bool
    rank: Optional[int] = None
    score: Optional[float] = None

class ModelDescription(BaseModel):
    model_id: str
    description: Optional[str]
    family: Optional[str]
    sktime_class: Optional[str]
    context_length: Optional[int]
    domain: Optional[str]
    tags: List[str] = []


class DescribeModelsResult(BaseModel):
    models: List[ModelDescription]
    unknown: List[str]
    message: str


class RecipeTemplateResult(BaseModel):
    task_choices: List[str]
    estimator_spec: List[str]
    optional_blocks: List[str]
    rules: List[str]
    examples: dict


class ModelTemplateResult(BaseModel):
    required_fields: List[str]
    pointer_choices: List[str]
    optional_fields: List[str]
    resolution_rules: List[str]
    example: dict
