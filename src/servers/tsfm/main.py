from __future__ import annotations

import logging
import os
from typing import List, Optional, Union

import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .bootstrap import fresh_store
from .config import RUNS_COLLECTION, PLANS_COLLECTION
from .core import tasks as task_spec
from .core import glossary as _glossary
from .core.results_models import (
    ErrorResult,
    TasksResult,
    ComponentsResult,
    CandidatesResult,
    ModelsResult,
    FeaturesResult,
    ComponentResult,
    ProfileResult,
    FeatureSelectionResult,
    RecipeResult,
    TabularResult,
    PlanResult,
    EvaluateResult,
    RegisterResult,
    ResultsListResult,
    ResultRecord,
    RunRecord,
    RunsResult,
    ExtractResult,
    FeatureNamesResult,
    FeatureDescription,
    DescribeFeaturesResult,
    FeatureTemplateResult,
    ModelDescription,
    DescribeModelsResult,
    ModelTemplateResult,
    FeatureCountResult,
    ModelCountResult,
    ResolveResult,
    DomainsResult,
    HfStatsResult,
    GiftStatusResult,
)

from .core.results_models import EvolveAskResult, EvolveTellResult, EvolveBestResult
from .core.results_models import CardResult, LineageResult, DataQualityResult
from .core.results_models import CharacterizeResult
from .reasoning import dataquality as _dq
from .io import refs
from .stores import model_store, feature_store, results
from .engine import composition, plan, evolve
from .eval import forecast_eval
from .reasoning import param_space, profile, patterns

load_dotenv()
logging.basicConfig(
    level=getattr(
        logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
    )
)
logger = logging.getLogger("tsfm-mcp-server")

mcp = FastMCP(
    "tsfm",
    instructions=(
        "Time-series AI on an sktime substrate. Discover components (models/features are catalog "
        "data), read evidence (profile_series), compose a recipe (transforms + single/ensemble + "
        "conformal), run it (run_recipe / run_tabular_recipe / run_plan), score it (evaluate, "
        "GIFT-Eval). Data is passed as FILE POINTERS (dataset_path); results come back as a "
        "results_file pointer. You reason every parameter; the server gives evidence + grades. "
        "Zero-shot is the default; fine-tune is optional. Forecasting and anomaly both run through "
        "run_recipe (anomaly via recipe.task). Results hand off downstream; no alerts here.\n\n"
        "VOCABULARY: " + _glossary.short_glossary() + "\n\n"
        "WORKFLOW: " + " ".join(_glossary.WORKFLOW) + "\n"
        "Call discover_components for the full glossary, the menu, and the recipe blocks."
    ),
)

_STORE = fresh_store()


def _load_target(
    dataset_path: str, timestamp_column: Optional[str], target_columns: List[str]
):
    """Resolve a file pointer to the (univariate) target series for forecasting."""
    obj = refs.load_series(
        dataset_path, time_col=timestamp_column, channels=target_columns
    )
    return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj


def _check_recipe(recipe) -> Optional[str]:
    """Validate the shape of a recipe before the engine touches it."""
    if not isinstance(recipe, dict) or not recipe:
        return "recipe must be a non-empty object"
    if "estimator" not in recipe and "ensemble" not in recipe:
        return "recipe must include an 'estimator' or an 'ensemble'"
    return None


def _check_task(task_id: str) -> Optional[str]:
    """Validate a task_id against the 8 standardized tasks; error lists the valid ids."""
    if not (task_id or "").strip():
        return "task_id is required"
    if task_id not in task_spec.TASKS:
        return f"unknown task '{task_id}'. Valid tasks: {list(task_spec.TASKS)}"
    return None


# ════════════════════════════ redesigned surface ════════════════════════════
# ---- discover ----
@mcp.tool(title="List Tasks")
def list_tasks() -> Union[TasksResult, ErrorResult]:
    """List the 8 standardized TS-AI TASKS (forecasting, regression, classification, anomaly,
    imputation, evaluation, similarity_search, clustering). Each entry has a plain `description`
    plus its contract (required inputs, output, eval protocol). Start here, then profile_series.
    """
    try:
        return TasksResult(tasks=task_spec.list_tasks())
    except Exception as exc:
        logger.error("list_tasks failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Discover Components")
def discover_components(
    task: str = "tsfm_forecasting",
) -> Union[ComponentsResult, ErrorResult]:
    """The mix-and-match MENU for a task: installed + foundation MODELS, FEATURE transforms,
    ensemble combiners, training REGIMES, and the recipe blocks you can fill. Also returns the
    full GLOSSARY + WORKFLOW so you know every term. A COMPONENT is any model or feature you
    place in a RECIPE (they are catalog data, not tools)."""
    bad = _check_task(task)
    if bad:
        return ErrorResult(error=bad)
    try:
        return ComponentsResult(
            task=task, components=composition.discover_components(_STORE, task=task)
        )
    except Exception as exc:
        logger.error("discover_components failed: %s", exc)
        return ErrorResult(error=str(exc))

@mcp.tool(title="List Domains")
def list_domains(task_id: Optional[str] = None) -> Union[DomainsResult, ErrorResult]:
    """The distinct domains present in the model catalog (the valid values for the `domain` filter
    of list_models / find_models / describe_candidates), each with its model count. Optionally
    scoped to a task."""
    if task_id:
        bad = _check_task(task_id)
        if bad:
            return ErrorResult(error=bad)
    try:
        counts: dict = {}
        for mdl in model_store.list_models(_STORE, task_id=task_id):
            d = mdl.get("domain") or "unspecified"
            counts[d] = counts.get(d, 0) + 1
        return DomainsResult(domains=dict(sorted(counts.items())))
    except Exception as exc:
        logger.error("list_domains failed: %s", exc)
        return ErrorResult(error=str(exc))

@mcp.tool(title="Describe Candidates")
def describe_candidates(
    task_id: str, top_k: int = 5, domain: Optional[str] = None
) -> Union[CandidatesResult, ErrorResult]:
    """CANDIDATE models for a task (HuggingGPT-style shortlist, CATALOG ORDER — no ranking). A
    candidate is a shortlisted MODEL; you decide which to use; top_k caps the list. Use hf_stats /
    gift_status to judge popularity / quality yourself."""
    bad = _check_task(task_id)
    if bad:
        return ErrorResult(error=bad)
    top_k = max(1, min(int(top_k), 50))
    try:
        return CandidatesResult(
            task_id=task_id,
            candidates=model_store.describe_candidates(
                _STORE, task_id, top_k=top_k, domain=domain
            ),
        )
    except Exception as exc:
        logger.error("describe_candidates failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Describe Models")
def describe_models(model_ids: List[str]) -> Union[DescribeModelsResult, ErrorResult]:
    """Return a compact record (description + family + sktime_class + context_length + domain + tags)
    for ONLY the given model_ids. The by-ids detail lookup that pairs with list_models / find_models
    (mirrors describe_features on the feature side)."""
    if not model_ids:
        return ErrorResult(error="provide at least one model_id (see list_models / find_models)")
    try:
        found, unknown = [], []
        for mid in model_ids:
            card = model_store.get_model(_STORE, mid)
            if card:
                found.append(
                    ModelDescription(
                        model_id=mid,
                        description=card.get("description"),
                        family=card.get("model_family") or card.get("family"),
                        sktime_class=card.get("sktime_class"),
                        context_length=card.get("context_length"),
                        domain=card.get("domain"),
                        tags=card.get("tags", []),
                    )
                )
            else:
                unknown.append(mid)
        return DescribeModelsResult(
            models=found,
            unknown=unknown,
            message=(
                f"described {len(found)} model(s)"
                + (f"; unknown: {unknown}" if unknown else "")
                + "."
            ),
        )
    except Exception as exc:
        logger.error("describe_models failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Find Models")
def find_models(
    task_id: str,
    min_context_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    domain: Optional[str] = None,
    top_k: int = 5,
) -> Union[ModelsResult, ErrorResult]:
    """Filter the MODEL catalog for a task, ranked shortlist. Filters: `domain` (exact),
    `min_context_length` / `prediction_length` (models lacking that field are excluded, e.g.
    classical models have no context_length). A model is an estimator card."""
    bad = _check_task(task_id)
    if bad:
        return ErrorResult(error=bad)
    top_k = max(1, min(int(top_k), 50))
    try:
        return ModelsResult(
            models=model_store.find_models(
                _STORE,
                task_id,
                min_context_length=min_context_length,
                prediction_length=prediction_length,
                domain=domain,
                top_k=top_k,
            )
        )
    except Exception as exc:
        logger.error("find_models failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Get Component")
def get_component(component_id: str) -> Union[ComponentResult, ErrorResult]:
    """Fetch one COMPONENT by id: a MODEL card or a FEATURE card (it resolves either). For a
    model it also returns the `param_schema` (the parameters + hints + ranges you must reason).
    """
    if not component_id.strip():
        return ErrorResult(error="component_id is required")
    card = model_store.get_model(_STORE, component_id)
    if card:
        try:
            card = {**card, "param_schema": param_space.param_schema(card)}
        except Exception as e:
            card = {**card, "param_schema_error": str(e)[:120]}
        return ComponentResult(component_id=component_id, **card)
    feat = feature_store.get_feature(_STORE, component_id)
    if feat:
        return ComponentResult(component_id=component_id, **feat)
    return ErrorResult(error=f"component '{component_id}' not found")


# ---- evidence / learn (file pointers in) ----
@mcp.tool(title="Profile Series")
def profile_series(
    dataset_path: str,
    timestamp_column: Optional[str] = None,
    channels: Optional[List[str]] = None,
) -> Union[ProfileResult, ErrorResult]:
    """EVIDENCE about the data behind a file pointer (dataset_path): seasonality, stationarity,
    channels, length. Facts only, no recommendations: you reason the recipe from these. This is
    the data the param_schema hints depend on (e.g. context_length ≥ 2× dominant_period).
    """
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    try:
        ev = profile.profile_ref(
            dataset_path, timestamp_column=timestamp_column, channels=channels
        )
        return ProfileResult(**ev)
    except Exception as exc:
        logger.error("profile_series failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Select Features")
def select_features(
    dataset_path: str,
    channel: str,
    extractors: List[str],
    timestamp_column: Optional[str] = None,
    reference_feature: str = "mean",
    cd_margin: float = 0.05,
) -> Union[FeatureSelectionResult, ErrorResult]:
    """Rank a CANDIDATE set of extractors on one series and return the shortlist worth keeping.
    Method: self-supervised one-step-ahead forecasting - slide a window over the series and score
    each candidate by how well the window's features predict the NEXT value (no labels needed),
    combining several criteria (correlation, F-test, mutual information, model importance) by mean
    rank, then keep those that beat `reference_feature` by at least `cd_margin`.

    `channel` (required) names the column to analyze - no default column is assumed. `extractors`
    (required) is the list of extractor names to score. Returns names only."""
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    if not channel:
        return ErrorResult(error="channel is required (name the column to analyze)")
    if not extractors:
        return ErrorResult(
            error="extractors is required: a list of extractor names to score (see list_features)"
        )
    from .reasoning import feature_selection as FS

    unknown = [c for c in extractors if c not in FS.EXTRACTORS]
    if unknown:
        return ErrorResult(
            error=f"unknown extractor(s): {unknown}. See list_features(kind='extractor')."
        )
    try:
        obj = refs.load_series(
            dataset_path, time_col=timestamp_column, channels=[channel]
        )
        series = (obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj).to_numpy()
        # score the candidates + the reference feature (so the cd_margin cutoff is meaningful)
        names = list(
            dict.fromkeys(
                list(extractors)
                + ([reference_feature] if reference_feature in FS.EXTRACTORS else [])
            )
        )
        subset = {n: FS.EXTRACTORS[n] for n in names}
        res = feature_store.select_features(
            series,
            reference_feature=reference_feature,
            cd_margin=cd_margin,
            extractors=subset,
        )
        detail = refs.write_json(res, name="feature_select")
        return FeatureSelectionResult(
            selected=res["selected"],
            lookback=res["lookback"],
            reference=res["reference"],
            scorers=res["scorers"],
            detail_file=detail,
        )
    except Exception as exc:
        logger.error("select_features failed: %s", exc)
        return ErrorResult(error=str(exc))

@mcp.tool(title="Characterize Series (pattern evidence)")
def characterize_series(
    dataset_path: str,
    timestamp_column: Optional[str] = None,
    channels: Optional[List[str]] = None,
    groups: Optional[dict] = None,
    group_rules: Optional[str] = None,
) -> Union[CharacterizeResult, ErrorResult]:
    """Describe the SHAPE of a series as structured EVIDENCE for an LLM to reason over (fault,
    cause, RUL, work-order, …): it never names a fault. Generic: any signals, any count, any
    names. Per channel-group it labels a state (stable / rise / decline / spike / level_shift /
    cessation / oscillation) + rate over changepoint phases, plus the bivariate relation
    (decoupled / co_move / lead_lag) between groups. Grouping is optional and yours to choose:
    pass groups={group:[channels]}, or group_rules (a preset name like 'vibration_temperature');
    default is one group per channel. Reference-free (reads the series' own median/MAD scale).
    """
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    try:
        obj = refs.load_series(
            dataset_path, time_col=timestamp_column, channels=channels
        )
        frame = (
            obj
            if isinstance(obj, pd.DataFrame)
            else obj.to_frame(name=(channels[0] if channels else "value"))
        )
        ev = patterns.describe_series(frame, groups=groups, group_rules=group_rules)
        evidence_file = refs.write_json(ev, name="pattern_evidence")
        return CharacterizeResult(
            status="success",
            summary=ev["summary"],
            n_observations=ev["n_observations"],
            evidence_file=evidence_file,
            groups=ev["groups"],
            phases=ev["phases"],
            message=f"Pattern evidence ({len(ev['phases'])} phase(s)). Full object at {evidence_file}.",
        )
    except Exception as exc:
        logger.error("characterize_series failed: %s", exc)
        return ErrorResult(error=str(exc))


# ---- compose + run (file pointers in/out) ----
@mcp.tool(title="Run Recipe")
def run_recipe(
    dataset_path: str,
    timestamp_column: str,
    target_columns: List[str],
    recipe: dict,
    asset_id: str = "asset",
    parent_run_id: Optional[str] = None,
) -> Union[RecipeResult, ErrorResult]:
    """Run a recipe on a series from a file pointer. Dispatches by recipe['task']:
    'tsfm_anomaly_detection' → detector path (e.g. tspulse_ad zero-shot, sublof) producing dense
    anomaly labels; otherwise FORECASTING (transforms + single/ensemble + conformal). Writes the
    run record to a results_file pointer."""
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    if not target_columns:
        return ErrorResult(error="target_columns must not be empty")
    bad = _check_recipe(recipe)
    if bad:
        return ErrorResult(error=bad)
    try:
        series = _load_target(dataset_path, timestamp_column, target_columns)
        res = composition.run_recipe(
            _STORE, series, recipe, asset_id=asset_id, parent_run_id=parent_run_id
        )
        if res.get("task") == "tsfm_anomaly_detection":  # detector path
            results_file = refs.write_json(
                {
                    "anomaly_label": res.pop("labels"),
                    "n_anomalies": res["n_anomalies"],
                    "anomaly_indices": res["anomaly_indices_head"],
                },
                name="anomaly",
            )
            return RecipeResult(
                status="success",
                run_id=res["run_id"],
                results_file=results_file,
                training_regime=res["training_regime"],
                n_anomalies=res["n_anomalies"],
                n_observations=res["n_observations"],
                message=f"Anomaly run complete ({res['training_regime']}): "
                f"{res['n_anomalies']}/{res['n_observations']} flagged. Labels at {results_file}.",
            )
        results_file = refs.write_json(res, name="recipe_run")  # forecasting path
        return RecipeResult(
            status="success",
            run_id=res["run_id"],
            results_file=results_file,
            metric=res["metric"],
            backtest_score=res["backtest_score"],
            training_regime=res["training_regime"],
            message=f"Recipe run complete ({res['training_regime']}). Record at {results_file}.",
        )
    except Exception as exc:
        logger.error("run_recipe failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Data Quality")
def data_quality(
    dataset_path: str,
    timestamp_column: str = "timestamp",
) -> Union[DataQualityResult, ErrorResult]:
    """Clean a series from a file pointer (NaN removal) + report a data-quality summary; returns a
    cleaned file pointer to feed forecasting / anomaly. (The continuous-segment IoT filter lives
    in reasoning/dataquality for the forecasting-context path.)"""
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    try:
        df = pd.read_csv(refs._path(dataset_path))
        nan = _dq._df_nan_stats(df)
        out = _dq._efficient_nan_removal(df)
        cleaned = out["df_filter"]
        cleaned_file = refs.write_series(cleaned, name="cleaned")
        nan_per_col = {
            str(k): float(v) for k, v in (nan.get("%NaN_per_column") or {}).items()
        }
        return DataQualityResult(
            status="success",
            cleaned_file=cleaned_file,
            rows_in=int(len(df)),
            rows_out=int(len(cleaned)),
            nan_per_column=nan_per_col,
            removed_cost=int(out.get("cost_total", 0)),
            message=f"Cleaned {len(df)}→{len(cleaned)} rows. Cleaned series at {cleaned_file}.",
        )
    except Exception as exc:
        logger.error("data_quality failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Run Tabular Recipe")
def run_tabular_recipe(
    dataset_path: str,
    recipe: dict,
    label_column: Optional[str] = None,
    asset_id: str = "asset",
) -> Union[TabularResult, ErrorResult]:
    """Series→tabular run (regression/classification/clustering): each row of the CSV file
    pointer is an instance; ``label_column`` (if supervised) holds y. FeatureUnion → estimator.
    """
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    bad = _check_recipe(recipe)
    if bad:
        return ErrorResult(error=bad)
    try:
        df = pd.read_csv(refs._path(dataset_path))
        y = None
        if label_column:
            if label_column not in df.columns:
                return ErrorResult(
                    error=f"label_column '{label_column}' not in dataset"
                )
            y = df[label_column].to_numpy()
            df = df.drop(columns=[label_column])
        X = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        res = composition.run_tabular_recipe(_STORE, X, recipe, y=y, asset_id=asset_id)
        results_file = refs.write_json(res, name="tabular_run")
        return TabularResult(
            status="success",
            run_id=res["run_id"],
            results_file=results_file,
            task=res["task"],
            metric=res["metric"],
            cv_score=res["cv_score"],
            n_features=res["n_features"],
            message=f"Tabular run complete ({res['task']}). Record at {results_file}.",
        )
    except Exception as exc:
        logger.error("run_tabular_recipe failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Run Plan")
def run_plan(
    plan_spec: dict, asset_id: str = "asset", scenario_id: Optional[str] = None
) -> Union[PlanResult, ErrorResult]:
    """Execute a recipe DAG (file-pointer chaining; HuggingGPT task-list)."""
    if not plan_spec:
        return ErrorResult(error="plan_spec is required")
    try:
        res = plan.run_plan(
            _STORE, plan_spec, asset_id=asset_id, scenario_id=scenario_id
        )
        results_file = refs.write_json(res, name="plan_run")
        return PlanResult(
            status="success",
            results_file=results_file,
            message=f"Plan complete. Record at {results_file}.",
            **res,
        )
    except Exception as exc:
        logger.error("run_plan failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Evaluate (GIFT-Eval)")
def evaluate(recipe: dict, configs: List[dict]) -> Union[EvaluateResult, ErrorResult]:
    """GIFT-Eval: seasonal-naive-normalized MASE+CRPS, geo-mean over configs."""
    bad = _check_recipe(recipe)
    if bad:
        return ErrorResult(error=bad)
    if not configs:
        return ErrorResult(error="configs must not be empty")
    try:
        res = forecast_eval.evaluate_recipe(_STORE, recipe, configs)
        results_file = refs.write_json(res, name="gifteval")
        return EvaluateResult(
            status="success",
            results_file=results_file,
            message=f"Evaluation complete. Scores at {results_file}.",
            **res,
        )
    except Exception as exc:
        logger.error("evaluate failed: %s", exc)
        return ErrorResult(error=str(exc))


# ---- write-back ----
@mcp.tool(title="Model Template")
def model_template() -> ModelTemplateResult:
    """The template for authoring a NEW model card for register_model. Returns the required and
    optional fields, the weight-pointer choices (a card must resolve via at least one), the
    resolution rules, and a filled example. Fill it in, then submit via register_model. (Use
    register_finetuned for fine-tune checkpoints.) Mirrors feature_template on the feature side."""
    return ModelTemplateResult(
        required_fields=["model_id", "description", "task_ids"],
        pointer_choices=[
            "sktime_class (+ params)  - resolve & construct via sktime, e.g. Est(**params)",
            "hf_repo                  - load weights lazily from HuggingFace",
            "artifact_path            - local checkpoint directory",
            "remote_endpoint          - hosted inference service",
            "model_checkpoint         - toolkit checkpoint (e.g. anomalykits://...)",
        ],
        optional_fields=[
            "model_family", "domain", "context_length", "prediction_length", "provenance",
            "base_model_id", "usage_modes", "param_hints", "training_regime", "frequency", "tags",
        ],
        resolution_rules=[
            "a card must be resolvable via at least one pointer_choice (else it is a catalog-only stub)",
            "provenance='finetuned' requires base_model_id (lineage)",
            "context_length / prediction_length must be >= 0",
        ],
        example={
            "model_id": "chronos_t5_small",
            "description": "Chronos T5 small zero-shot forecaster",
            "task_ids": ["tsfm_forecasting"],
            "sktime_class": "sktime.forecasting.chronos.ChronosForecaster",
            "params": {"model_path": "amazon/chronos-t5-small"},
            "hf_repo": "amazon/chronos-t5-small",
            "model_family": "chronos",
            "domain": "general",
            "context_length": 512,
            "prediction_length": 64,
            "provenance": "pretrained",
            "tags": ["foundation", "zero-shot"],
        },
    )


@mcp.tool(title="Register Model")
def register_model(model: dict) -> Union[RegisterResult, ErrorResult]:
    """Register a model card in the catalog (schema-validated). Requires `model_id`, `description`,
    and `task_ids`; point it at weights via one of `sktime_class` / `artifact_path` / `hf_repo` /
    `remote_endpoint` / `model_checkpoint`. Use register_finetuned for fine-tune checkpoints."""
    if not model:
        return ErrorResult(error="model card is required")
    try:
        rec = model_store.register_model(_STORE, model, overwrite=True)
        return RegisterResult(status="registered", id=rec.get("model_id", ""), card=rec)
    except Exception as exc:
        logger.error("register_model failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Feature Template")
def feature_template() -> FeatureTemplateResult:
    """The template for authoring a NEW Evolutionary Feature Engineering transform feature card. Returns the card fields
    (required + optional) and a fillable `Transformation` code skeleton matching the exact interface
    the EFE runner enforces. Fill it in, then submit via register_feature. (Extractors are a fixed
    library and are NOT authored here.)"""
    skeleton = (
        "class Transformation:\n"
        "    def fit(self, X, metadata):\n"
        "        # learn state from X (training data); return the state\n"
        "        # (or store it on self as self.state_ and return None)\n"
        "        return state\n"
        "    def transform(self, X, state):\n"
        "        # return a NEW array - never mutate X in place\n"
        "        return X_new\n"
        "    def inverse_transform(self, X_new, state):\n"
        "        # ONLY when interface == 'fit_transform_inverse'; must round-trip\n"
        "        return X\n"
    )
    example_code = (
        "class Transformation:\n"
        "    def fit(self, X, metadata):\n"
        "        import numpy as np\n"
        "        self.med_ = np.median(X, axis=0)\n"
        "        self.mad_ = np.median(np.abs(X - self.med_), axis=0) + 1e-9\n"
        "        return None\n"
        "    def transform(self, X, state):\n"
        "        return (X - self.med_) / self.mad_\n"
        "    def inverse_transform(self, X_new, state):\n"
        "        return X_new * self.mad_ + self.med_\n"
    )
    return FeatureTemplateResult(
        required_fields=["feature_id", "interface", "code"],
        optional_fields=["name", "description", "invertible", "target_task", "target_model", "tags"],
        interface_choices=["fit_transform", "fit_transform_inverse"],
        code_skeleton=skeleton,
        validity_rules=[
            "entry points: fit + transform must exist (+ inverse_transform if invertible)",
            "transform must return a NEW object - no in-place mutation of X",
            "invertible transforms must round-trip: inverse_transform(transform(X)) ~= X",
        ],
        example={
            "feature_id": "robust_norm_v1",
            "interface": "fit_transform_inverse",
            "name": "robust per-channel normalization",
            "description": "center by median, scale by MAD; invertible",
            "invertible": True,
            "code": example_code,
        },
    )

@mcp.tool(title="Register Feature")
def register_feature(feature: dict) -> Union[RegisterResult, ErrorResult]:
    """Register a NEW transform (Evolutionary Feature Engineering) feature card: an executable fit/transform program.
    Requires `feature_id`, `interface` ('fit_transform' | 'fit_transform_inverse'), and `code`
    (validated + gated by the Evolutionary Feature Engineering runner). Extractors are NOT registered here: they are the fixed
     scalar library (see list_features(kind='extractor'))."""
    if not feature:
        return ErrorResult(error="feature card is required")
    try:
        rec = feature_store.register_feature(_STORE, feature, overwrite=True)
        return RegisterResult(
            status="registered", id=rec.get("feature_id", ""), card=rec
        )
    except Exception as exc:
        logger.error("register_feature failed: %s", exc)
        return ErrorResult(error=str(exc))

@mcp.tool(title="Count Features")
def count_features() -> Union[FeatureCountResult, ErrorResult]:
    """How many features are in the catalog. Returns extractor / transform / total counts."""
    try:
        ex = len(feature_store.find_features(_STORE, kind="extractor"))
        tr = len(feature_store.find_features(_STORE, kind="transform"))
        return FeatureCountResult(extractors=ex, transforms=tr, total=ex + tr)
    except Exception as exc:
        logger.error("count_features failed: %s", exc)
        return ErrorResult(error=str(exc))

# ════════════════════════ catalog management (lifecycle) ═════════════════════
# Pull + update + version + retire, for both stores. The agent curates the catalog: search it,
# trace lineage, edit/retire a card, cut a new version, or register a fine-tuned checkpoint.

@mcp.tool(title="Count Models")
def count_models() -> Union[ModelCountResult, ErrorResult]:
    """How many models are in the catalog. Returns the total active models and a per-task breakdown."""
    try:
        models = model_store.list_models(_STORE)
        by_task: dict = {}
        for mdl in models:
            for tid in mdl.get("task_ids", []):
                by_task[tid] = by_task.get(tid, 0) + 1
        return ModelCountResult(total=len(models), by_task=dict(sorted(by_task.items())))
    except Exception as exc:
        logger.error("count_models failed: %s", exc)
        return ErrorResult(error=str(exc))

# ---- model store ----
@mcp.tool(title="List Models")
def list_models(
    task_id: Optional[str] = None, domain: Optional[str] = None, status: str = "active"
) -> Union[ModelsResult, ErrorResult]:
    """List model cards in the catalog (optionally filtered by task / domain)."""
    if task_id:
        bad = _check_task(task_id)
        if bad:
            return ErrorResult(error=bad)
    try:
        return ModelsResult(
            models=model_store.list_models(
                _STORE, task_id=task_id, domain=domain, status=status
            )
        )
    except Exception as exc:
        logger.error("list_models failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Search Models")
def search_models(
    text: str, tags: Optional[List[str]] = None, status: str = "active"
) -> Union[ModelsResult, ErrorResult]:
    """Substring (case-insensitive) search over the model catalog (id / description / model_family
    / tags). `text` (required) is the substring to match; use list_models to browse all."""
    if not text.strip():
        return ErrorResult(
            error="text is required: a substring to search for (use list_models to browse all)"
        )
    try:
        return ModelsResult(
            models=model_store.search(_STORE, text, tags=tags, status=status)
        )
    except Exception as exc:
        logger.error("search_models failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Get Model Lineage")
def get_model_lineage(model_id: str) -> Union[LineageResult, ErrorResult]:
    """A model's lineage: the fine-tune chain (base-model ancestors + fine-tune descendants) plus
    its version links (supersedes / superseded_by)."""
    if not model_id.strip():
        return ErrorResult(error="model_id is required")
    try:
        return LineageResult(**model_store.get_lineage(_STORE, model_id))
    except Exception as exc:
        logger.error("get_model_lineage failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Update Model")
def update_model(model_id: str, fields: dict) -> Union[CardResult, ErrorResult]:
    """Patch fields on a model card (status, tags, domain, …)."""
    if not model_id.strip() or not fields:
        return ErrorResult(error="model_id and fields are required")
    try:
        return CardResult(**model_store.update_model(_STORE, model_id, fields))
    except Exception as exc:
        logger.error("update_model failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Deprecate Model")
def deprecate_model(
    model_id: str, reason: Optional[str] = None
) -> Union[CardResult, ErrorResult]:
    """Retire a model card (status=deprecated); it stops appearing in active listings."""
    if not model_id.strip():
        return ErrorResult(error="model_id is required")
    try:
        return CardResult(
            **model_store.deprecate_model(_STORE, model_id, reason=reason)
        )
    except Exception as exc:
        logger.error("deprecate_model failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="New Model Version")
def new_model_version(
    model_id: str, fields: dict, new_model_id: Optional[str] = None
) -> Union[CardResult, ErrorResult]:
    """Create a successor version of a model; the predecessor is marked superseded + linked."""
    if not model_id.strip():
        return ErrorResult(error="model_id is required")
    try:
        return CardResult(
            **model_store.new_version(
                _STORE, model_id, fields or {}, new_model_id=new_model_id
            )
        )
    except Exception as exc:
        logger.error("new_model_version failed: %s", exc)
        return ErrorResult(error=str(exc))

@mcp.tool(title="Resolve Model")
def resolve_model(model_id: str) -> Union[ResolveResult, ErrorResult]:
    """Read-only PREFLIGHT: check whether a model card can be resolved (loaded) before you compose a
    recipe. Confirms the card exists, has an importable sktime_class, and reports where its weights
    come from (params.model_path / hf_repo / checkpoint, or none for classical models). Does NOT
    download weights or fit."""
    if not model_id.strip():
        return ErrorResult(error="model_id is required")
    card = model_store.get_model(_STORE, model_id)
    if not card:
        return ErrorResult(error=f"model '{model_id}' not found")
    from .substrate import resolver as R

    regime = R.training_regime(card)
    params = card.get("params") or {}
    weights_from = (
        params.get("model_path")
        or card.get("hf_repo")
        or card.get("model_checkpoint")
        or card.get("artifact_path")
    )
    sk = card.get("sktime_class")
    if not sk:
        return ResolveResult(
            model_id=model_id, resolvable=False, sktime_class=None,
            training_regime=regime, weights_from=weights_from,
            reason="no sktime_class (toolkit/checkpoint model; not sktime-resolvable)",
        )
    try:
        R._import_target(sk)  # import only; no instantiation / no weight download
    except Exception as e:
        return ResolveResult(
            model_id=model_id, resolvable=False, sktime_class=sk,
            training_regime=regime, weights_from=weights_from,
            reason=f"sktime_class not importable: {type(e).__name__}: {e}"[:140],
        )
    return ResolveResult(
        model_id=model_id, resolvable=True, sktime_class=sk,
        training_regime=regime, weights_from=weights_from or "none (classical; fit from series)",
        reason="importable sktime_class + params; weights load lazily at fit",
    )

@mcp.tool(title="HF Model Stats")
def hf_stats(
    model_id: Optional[str] = None, hf_repo: Optional[str] = None
) -> Union[HfStatsResult, ErrorResult]:
    """Look up a model's HuggingFace popularity: downloads + likes (READ-ONLY, does not change the
    catalog). Give a catalog `model_id` (its hf_repo is resolved) OR an `hf_repo` directly. Use it
    to weigh how widely adopted a model is before choosing it. Needs network to huggingface.co."""
    repo = hf_repo
    if model_id and not repo:
        card = model_store.get_model(_STORE, model_id)
        if not card:
            return ErrorResult(error=f"model '{model_id}' not found")
        repo = card.get("hf_repo")
    if not repo:
        return ErrorResult(error="give a model_id that has an hf_repo, or an hf_repo directly")
    try:
        st = model_store._hf_model_stats(repo)
        return HfStatsResult(
            model_id=model_id, hf_repo=repo,
            downloads=st.get("downloads"), likes=st.get("likes"),
        )
    except Exception as exc:
        logger.error("hf_stats failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="GIFT-Eval Status")
def gift_status(
    model_id: Optional[str] = None, name: Optional[str] = None, url: Optional[str] = None
) -> Union[GiftStatusResult, ErrorResult]:
    """Look up a model's GIFT-Eval leaderboard standing: rank + score (READ-ONLY). Give a catalog
    `model_id` (matched by leaderboard_id / hf_repo / model_id) OR a leaderboard `name`. `url` (or
    env GIFTEVAL_URL) points at the GIFT-Eval results JSON. Pair with hf_stats to decide."""
    import os

    key = name
    if model_id and not key:
        card = model_store.get_model(_STORE, model_id)
        if not card:
            return ErrorResult(error=f"model '{model_id}' not found")
        key = card.get("leaderboard_id") or card.get("hf_repo") or model_id
    if not key:
        return ErrorResult(error="give a model_id or a leaderboard name")
    src = url or os.environ.get("GIFTEVAL_URL") or os.environ.get("TSFM_LEADERBOARD_URL")
    if not src:
        return ErrorResult(error="no GIFT-Eval source (pass url= or set GIFTEVAL_URL)")
    try:
        entry = model_store._leaderboard_stats(src).get(key) or {}
        return GiftStatusResult(
            model_id=model_id, key=key, found=bool(entry),
            rank=entry.get("rank"), score=entry.get("score"),
        )
    except Exception as exc:
        logger.error("gift_status failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Register Finetuned Model")
def register_finetuned(
    model_id: str,
    checkpoint_path: str,
    base_model_id: str,
    context_length: int,
    prediction_length: int,
    description: str,
    domain: str = "general",
) -> Union[CardResult, ErrorResult]:
    """Add a fine-tuned model: point the catalog at a checkpoint, with lineage to its base model."""
    for k, v in (
        ("model_id", model_id),
        ("checkpoint_path", checkpoint_path),
        ("base_model_id", base_model_id),
        ("description", description),
    ):
        if not (v and str(v).strip()):
            return ErrorResult(error=f"{k} is required")
    try:
        return CardResult(
            **model_store.register_finetuned(
                _STORE,
                model_id=model_id,
                checkpoint_path=checkpoint_path,
                base_model_id=base_model_id,
                context_length=context_length,
                prediction_length=prediction_length,
                description=description,
                domain=domain,
            )
        )
    except Exception as exc:
        logger.error("register_finetuned failed: %s", exc)
        return ErrorResult(error=str(exc))


# ---- feature store ----
@mcp.tool(title="Search Features")
def search_features(
    text: str, tags: Optional[List[str]] = None, status: str = "active"
) -> Union[FeaturesResult, ErrorResult]:
    """Substring (case-insensitive) search over the FEATURE catalog only, both extractors and
    transforms, matching id / name / description / tags. Literal substring, NOT semantic: 'spectral'
    or 'entropy' hit; a concept only implied by wording may not."""
    if not text.strip():
        return ErrorResult(
            error="text is required: a substring to search for (use list_features to browse all)"
        )
    try:
        return FeaturesResult(
            features=feature_store.search(_STORE, text, tags=tags, status=status)
        )
    except Exception as exc:
        logger.error("search_features failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="List Features")
def list_features(kind: Optional[str] = None) -> Union[FeatureNamesResult, ErrorResult]:
    """List feature NAMES from the catalog (compact, no descriptions). `kind` filters:
    'extractor' (the  scalar library), 'transform' (Evolutionary Feature Engineering preprocessing programs), or omit for
    all."""
    if kind is not None and kind not in ("extractor", "transform"):
        return ErrorResult(error="kind must be 'extractor', 'transform', or omitted")
    try:
        cards = feature_store.find_features(
            _STORE, kind=kind
        )  # kind=None -> all feature cards
        names = sorted(c["feature_id"] for c in cards if c.get("feature_id"))
        return FeatureNamesResult(count=len(names), kind=kind, features=names)
    except Exception as exc:
        logger.error("list_features failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Describe Features")
def describe_features(names: List[str]) -> Union[DescribeFeaturesResult, ErrorResult]:
    """Return kind + name + description for ONLY the given feature names (extractors OR transforms)."""
    if not names:
        return ErrorResult(
            error="provide at least one feature name (see list_features)"
        )
    try:
        found, unknown = [], []
        for n in names:
            card = feature_store.get_feature(_STORE, n)
            if card and card.get("kind") in ("extractor", "transform"):
                found.append(
                    FeatureDescription(
                        feature_id=n,
                        kind=card.get("kind"),
                        name=card.get("name"),
                        description=card.get("description"),
                    )
                )
            else:
                unknown.append(n)
        return DescribeFeaturesResult(
            features=found,
            unknown=unknown,
            message=(
                f"described {len(found)} feature(s)"
                + (f"; unknown: {unknown}" if unknown else "")
                + "."
            ),
        )
    except Exception as exc:
        logger.error("describe_features failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Get Feature Lineage")
def get_feature_lineage(feature_id: str) -> Union[LineageResult, ErrorResult]:
    """A feature's evolution chain (parent / generation / descendants). Extractors are a fixed
    library and return an empty chain; lineage is meaningful only for transforms."""
    if not feature_id.strip():
        return ErrorResult(error="feature_id is required")
    try:
        return LineageResult(**feature_store.get_lineage(_STORE, feature_id))
    except Exception as exc:
        logger.error("get_feature_lineage failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Update Feature")
def update_feature(feature_id: str, fields: dict) -> Union[CardResult, ErrorResult]:
    """Patch fields on a feature card (status, description, …)."""
    if not feature_id.strip() or not fields:
        return ErrorResult(error="feature_id and fields are required")
    try:
        return CardResult(**feature_store.update_feature(_STORE, feature_id, fields))
    except Exception as exc:
        logger.error("update_feature failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Deprecate Feature")
def deprecate_feature(
    feature_id: str, reason: Optional[str] = None
) -> Union[CardResult, ErrorResult]:
    """Retire a feature card (status=deprecated)."""
    if not feature_id.strip():
        return ErrorResult(error="feature_id is required")
    try:
        return CardResult(
            **feature_store.deprecate_feature(_STORE, feature_id, reason=reason)
        )
    except Exception as exc:
        logger.error("deprecate_feature failed: %s", exc)
        return ErrorResult(error=str(exc))

@mcp.tool(title="New Feature Version")
def new_feature_version(
    feature_id: str, fields: Optional[dict] = None, new_feature_id: Optional[str] = None
) -> Union[CardResult, ErrorResult]:
    """Create a successor version of a TRANSFORM feature; predecessor superseded + linked.
    Extractors are a fixed library and cannot be versioned."""
    if not feature_id.strip():
        return ErrorResult(error="feature_id is required")
    try:
        return CardResult(
            **feature_store.new_version(
                _STORE, feature_id, fields or {}, new_feature_id=new_feature_id
            )
        )
    except Exception as exc:
        logger.error("new_feature_version failed: %s", exc)
        return ErrorResult(error=str(exc))


# ---- results / runs ----
@mcp.tool(title="Get Result")
def get_result(task_type: str, result_id: str) -> Union[ResultRecord, ErrorResult]:
    rec = results.get_result(_STORE, task_type, result_id)
    return ResultRecord(**rec) if rec else ErrorResult(error="not found")


@mcp.tool(title="List Results")
def list_results(
    task_type: str, asset_id: Optional[str] = None, scenario_id: Optional[str] = None
) -> ResultsListResult:
    return ResultsListResult(
        results=results.list_results(
            _STORE, task_type, asset_id=asset_id, scenario_id=scenario_id
        )
    )


@mcp.tool(title="Get Run")
def get_run(run_id: str) -> Union[RunRecord, ErrorResult]:
    rec = _STORE.get(RUNS_COLLECTION, run_id)
    return RunRecord(**rec) if rec else ErrorResult(error="not found")


@mcp.tool(title="List Runs")
def list_runs(asset_id: Optional[str] = None) -> RunsResult:
    sel = {"asset_id": asset_id} if asset_id else {}
    return RunsResult(
        runs=_STORE.find(RUNS_COLLECTION, sel), plans=_STORE.find(PLANS_COLLECTION, sel)
    )


# ════════════════════════════ evolve (AlphaEvolve loop) ═════════════════════
# Agent-generates / server-grades: the server samples + validates + evaluates + archives;
# the agent (LLM) is the proposer. A "program" is a recipe OR an EFE feature program.
@mcp.tool(title="Evolve: Ask")
def evolve_ask(
    task: str,
    kind: str = "recipe",
    dataset_path: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    channels: Optional[List[str]] = None,
    n_parents: int = 2,
    n_inspirations: int = 3,
) -> Union[EvolveAskResult, ErrorResult]:
    """Sample parent(s) + diverse inspirations from the evolutionary archive + data evidence +
    the task contract, so YOU (the agent) can mutate/recombine them into one new candidate to
    submit via evolve_tell. kind = 'recipe' | 'feature'."""
    bad = _check_task(task)
    if bad:
        return ErrorResult(error=bad)
    try:
        return EvolveAskResult(
            **evolve.evolve_ask(
                _STORE,
                task,
                kind=kind,
                data_ref=dataset_path,
                timestamp_column=timestamp_column,
                channels=channels,
                n_parents=n_parents,
                n_inspirations=n_inspirations,
            )
        )
    except Exception as exc:
        logger.error("evolve_ask failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Evolve: Tell")
def evolve_tell(
    task: str,
    kind: str,
    program: dict,
    parent_id: Optional[str] = None,
    dataset_path: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    target_columns: Optional[List[str]] = None,
    label_column: Optional[str] = None,
) -> Union[EvolveTellResult, ErrorResult]:
    """Submit a candidate PROGRAM (a recipe or a feature program). The server validates it,
    evaluates it to a scalar fitness (run_recipe / run_tabular_recipe / EFE gate), and places it
    in the MAP-Elites archive with lineage. Returns the fitness + whether it's a new elite.
    """
    bad = _check_task(task)
    if bad:
        return ErrorResult(error=bad)
    if kind not in ("recipe", "feature"):
        return ErrorResult(error="kind must be 'recipe' or 'feature'")
    if not program:
        return ErrorResult(error="program is required")
    try:
        return EvolveTellResult(
            **evolve.evolve_tell(
                _STORE,
                task,
                kind,
                program,
                parent_id=parent_id,
                data_ref=dataset_path,
                timestamp_column=timestamp_column,
                target_columns=target_columns,
                label_column=label_column,
            )
        )
    except Exception as exc:
        logger.error("evolve_tell failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Evolve: Best")
def evolve_best(
    task: str, kind: Optional[str] = None, top_k: int = 5
) -> Union[EvolveBestResult, ErrorResult]:
    """The current elites (best program per behaviour cell) for a task: the evolved frontier."""
    bad = _check_task(task)
    if bad:
        return ErrorResult(error=bad)
    try:
        return EvolveBestResult(
            **evolve.evolve_best(
                _STORE, task, kind=kind, top_k=max(1, min(int(top_k), 50))
            )
        )
    except Exception as exc:
        logger.error("evolve_best failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Extract Features")
def extract_features(
    dataset_path: str,
    extractors: List[str],
    target_columns: List[str],
    timestamp_column: Optional[str] = None,
    window: Optional[int] = None,
) -> Union[ExtractResult, ErrorResult]:
    """Apply the chosen  extractors to a series and RETURN the extracted feature values,
    raw feature extraction, no model. Pick `extractors` by name from list_features(kind="extractor").
    window=None -> one feature vector for the whole series; window=W -> non-overlapping W-length
    tiles -> a (windows x features) matrix. Multivariate: each target column yields its own
    '<column>.<extractor>' feature columns. `target_columns` (required) names the column(s) to extract from - no default is assumed."""
    from .reasoning import feature_selection as FS
    import numpy as np
    import pandas as pd

    if not target_columns:
        return ErrorResult(
            error="target_columns is required: name the column(s) to extract from"
        )

    if not extractors:
        return ErrorResult(
            error="provide at least one extractor name (see list_features(kind='extractor'))"
        )
    unknown = [e for e in extractors if e not in FS.EXTRACTORS]
    if unknown:
        return ErrorResult(
            error=f"unknown extractor(s): {unknown}. See list_features(kind='extractor')."
        )

    try:
        obj = refs.load_series(
            dataset_path, time_col=timestamp_column, channels=target_columns
        )
    except Exception as exc:
        logger.error("extract_features load failed: %s", exc)
        return ErrorResult(error=str(exc))

    if isinstance(obj, pd.Series):
        name = obj.name or (target_columns[0] if target_columns else "value")
        channels = {str(name): np.asarray(obj, dtype=float)}
    else:
        channels = {str(c): np.asarray(obj[c], dtype=float) for c in obj.columns}

    cols, F = composition.extract_features(channels, extractors, window=window)
    return ExtractResult(
        n_windows=int(F.shape[0]),
        window=window,
        columns=cols,
        features=[[round(float(v), 6) for v in row] for row in F.tolist()],
        message=(
            f"extracted {len(cols)} feature column(s) over {F.shape[0]} window(s) "
            f"from {len(channels)} channel(s)."
        ),
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()