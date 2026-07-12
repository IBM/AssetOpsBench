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


@mcp.tool(title="Describe Candidates")
def describe_candidates(
    task_id: str, top_k: int = 5, domain: Optional[str] = None
) -> Union[CandidatesResult, ErrorResult]:
    """Ranked CANDIDATE models for a task (HuggingGPT-style, by description + popularity). A
    candidate is a shortlisted MODEL; you still decide which to use. top_k caps the list.
    """
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


@mcp.tool(title="Find Models")
def find_models(
    task_id: str,
    min_context_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    domain: Optional[str] = None,
    top_k: int = 5,
) -> Union[ModelsResult, ErrorResult]:
    """Filter the MODEL catalog for a task → ranked shortlist. A model is an estimator card; use
    get_component(model_id) to read its full card + param_schema before composing a recipe.
    """
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


@mcp.tool(title="Select Features (FLOps)")
def select_features(
    dataset_path: str,
    timestamp_column: Optional[str] = None,
    target_column: Optional[str] = None,
    reference_feature: str = "mean",
    cd_margin: float = 0.05,
) -> Union[FeatureSelectionResult, ErrorResult]:
    """FLOps multi-config feature SELECTION: scores the extractor library against the target and
    returns the SHORTLIST of most-informative extractor names (+ the auto-discovered lookback,
    reference, and per-scorer detail file). This RANKS/PICKS names; it does not compute values;
    feed the selected names to extract_features to get the actual feature matrix."""
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    try:
        chans = [target_column] if target_column else None
        obj = refs.load_series(dataset_path, time_col=timestamp_column, channels=chans)
        series = (obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj).to_numpy()
        res = feature_store.select_features(
            series, reference_feature=reference_feature, cd_margin=cd_margin
        )
        detail = refs.write_json(res, name="flops_select")
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
@mcp.tool(title="Register Model")
def register_model(model: dict) -> Union[RegisterResult, ErrorResult]:
    if not model:
        return ErrorResult(error="model card is required")
    try:
        rec = model_store.register_model(_STORE, model, overwrite=True)
        return RegisterResult(status="registered", id=rec.get("model_id", ""), card=rec)
    except Exception as exc:
        logger.error("register_model failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Register Feature")
def register_feature(feature: dict) -> Union[RegisterResult, ErrorResult]:
    """Register a NEW transform (EFE) feature card: an executable fit/transform program.
    Requires `feature_id`, `interface` ('fit_transform' | 'fit_transform_inverse'), and `code`
    (validated + gated by the EFE runner). Extractors are NOT registered here: they are the fixed
    FLOps scalar library (see list_features(kind='extractor'))."""
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


# ════════════════════════ catalog management (lifecycle) ═════════════════════
# Pull + update + version + retire, for both stores. The agent curates the catalog: search it,
# trace lineage, edit/retire a card, cut a new version, or register a fine-tuned checkpoint.


# ---- model store ----
@mcp.tool(title="List Models")
def list_models(
    task_id: Optional[str] = None, domain: Optional[str] = None, status: str = "active"
) -> Union[ModelsResult, ErrorResult]:
    """List model cards in the catalog (optionally filtered by task / domain). Unranked; the
    mirror of list_features; use find_models / describe_candidates to rank for a task.
    """
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
    text: str = "", tags: Optional[List[str]] = None, status: str = "active"
) -> Union[ModelsResult, ErrorResult]:
    """Free-text/tag search over the model catalog (id, description, family, tags)."""
    try:
        return ModelsResult(
            models=model_store.search(_STORE, text, tags=tags, status=status)
        )
    except Exception as exc:
        logger.error("search_models failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Get Model Lineage")
def get_model_lineage(model_id: str) -> Union[LineageResult, ErrorResult]:
    """A model's version chain: what it supersedes / is superseded by (the evolution trail)."""
    if not model_id.strip():
        return ErrorResult(error="model_id is required")
    try:
        return LineageResult(**model_store.get_lineage(_STORE, model_id))
    except Exception as exc:
        logger.error("get_model_lineage failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Update Model")
def update_model(model_id: str, fields: dict) -> Union[CardResult, ErrorResult]:
    """Patch fields on a model card (status, metrics, tags, domain, …)."""
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


@mcp.tool(title="Register Finetuned Model")
def register_finetuned(
    model_id: str,
    checkpoint_path: str,
    base_model_id: str,
    context_length: int,
    prediction_length: int,
    description: str,
    domain: str = "general",
    metrics: Optional[list] = None,
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
                metrics=metrics,
            )
        )
    except Exception as exc:
        logger.error("register_finetuned failed: %s", exc)
        return ErrorResult(error=str(exc))


# ---- feature store ----
@mcp.tool(title="Search Features")
def search_features(
    text: str = "", tags: Optional[List[str]] = None, status: str = "active"
) -> Union[FeaturesResult, ErrorResult]:
    """Substring (case-insensitive) search over the FEATURE catalog only, both extractors and
    transforms, matching id / name / description / tags. Literal substring, NOT semantic: 'spectral'
    or 'entropy' hit; a concept only implied by wording may not. Use list_features for the full
    name list; use search_models for the model catalog."""
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
    'extractor' (the FLOps scalar library), 'transform' (EFE preprocessing programs), or omit for
    all. Shortlist by name, call describe_features([...]) for descriptions, then extract_features(...)
    (extractors) or use transforms in a recipe."""
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
    """Return kind + name + description for ONLY the given feature names (extractors OR transforms)
    Use after list_features to read descriptions for the handful you're weighing."""
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
    """Patch fields on a feature card (status, description, metrics, …)."""
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
    feature_id: str, fields: dict, new_feature_id: Optional[str] = None
) -> Union[CardResult, ErrorResult]:
    """Create a successor version of a TRANSFORM (EFE) feature; predecessor superseded + linked.
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
    timestamp_column: Optional[str] = None,
    target_columns: Optional[List[str]] = None,
    window: Optional[int] = None,
) -> Union[ExtractResult, ErrorResult]:
    """Apply the chosen FLOps extractors to a series and RETURN the extracted feature values,
    raw feature extraction, no model. Pick `extractors` by name from list_features(kind="extractor").
    window=None -> one feature vector for the whole series; window=W -> non-overlapping W-length
    tiles -> a (windows x features) matrix. Multivariate: each target column yields its own
    '<column>.<extractor>' feature columns."""
    from ..reasoning import feature_selection as FS
    import numpy as np
    import pandas as pd

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
