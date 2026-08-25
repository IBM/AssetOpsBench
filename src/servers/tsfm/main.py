"""TSFM MCP server for time-series evidence, catalogs, and recipe execution.

The MCP-facing tools help agents profile asset time series, choose catalog cards, and run
forecasting or anomaly-detection recipes. In particular, `run_recipe` dispatches
`recipe["task"] == "tsfm_anomaly_detection"` to the detector path and returns anomaly labels,
counts, run records, and results-file pointers.

Catalogs are CouchDB data loaded by src/couchdb/init_data.py; `MODEL_CATALOG_DBNAME` and
`FEATURE_CATALOG_DBNAME` select the collections. Model cards are pointers to loadable models,
never weights.
"""

from __future__ import annotations

import logging
import sys
import os
from typing import List, Optional, Union

import pandas as pd
from mcp.server.fastmcp import FastMCP

from .core import tasks as task_spec
from .core.results_models import (
    CandidatesResult,
    CharacterizeResult,
    CardResult,
    EvaluateResult,
    DataQualityResult,
    DescribeModelsResult,
    DomainsResult,
    ErrorResult,
    HfStatsResult,
    LineageResult,
    ModelCountResult,
    ModelDescription,
    ModelsResult,
    PlanResult,
    ModelTemplateResult,
    ProfileResult,
    RecipeResult,
    RecipeTemplateResult,
    RegisterResult,
    ResolveResult,
    ResultRecord,
    ResultsListResult,
    RunRecord,
    RunsResult,
    TabularResult,
    TasksResult,
    FeaturesResult,
    FeatureCountResult,
    DescribeFeaturesResult,
    FeatureDescription,
    ExtractResult,
    FeatureSelectionResult,
)
from .core.store import make_store
from .io import refs
from .reasoning import dataquality as _dq
from .reasoning import patterns, profile
from .stores import model_store
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from .config import PLANS_COLLECTION, RUNS_COLLECTION
from .engine import composition, plan
from .eval import forecast_eval
from .stores import feature_store, results

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("tsfm-mcp-server")

mcp = FastMCP(
    "tsfm",
    instructions=(
        "The TSFM server provides task discovery, file-pointer evidence tools, recipe execution, "
        "and CouchDB-backed model and feature catalog tools for forecasting and anomaly detection. "
        "For anomaly detection, follow this workflow: 1) call `recipe_template` to confirm the "
        "recipe shape; 2) choose a detector with `find_models(task_id=\"tsfm_anomaly_detection\")` "
        "or `search_models`; 3) call `run_recipe` with `recipe={\"task\": "
        "\"tsfm_anomaly_detection\", \"estimator\": {\"model_id\": \"<model_id>\"}}`; "
        "4) report the anomalous segment from returned labels, indices, or `results_file`."
    ),
)

# The catalog is an ordinary AssetOpsBench CouchDB collection (MODEL_CATALOG_DBNAME, default
# model_catalog), loaded by src/couchdb/init_data.py. The server reads it; it does not seed.
_STORE = make_store()


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


# =============================================================================
# Tasks & discovery
# =============================================================================


@mcp.tool(title="List Tasks")
def list_tasks() -> Union[TasksResult, ErrorResult]:
    """List the standardized TSFM tasks available in the benchmark.

This is a discovery tool: it returns the canonical task definitions, including each task's
required inputs, output type, evaluation protocol, and supporting notes. Use it first when you
need to understand what task families the TSFM server supports.

Args:
    None

Returns:
    TasksResult: The task catalog, including task IDs, descriptions, required inputs, metrics,
    and protocol metadata. ErrorResult if the task catalog cannot be loaded.
"""
    try:
        return TasksResult(tasks=task_spec.list_tasks())
    except Exception as exc:
        logger.error("list_tasks failed: %s", exc)
        return ErrorResult(error=str(exc))


# =============================================================================
# Data & evidence (file pointers in)
# =============================================================================


@mcp.tool(title="Profile Series")
def profile_series(
    dataset_path: str,
    timestamp_column: Optional[str] = None,
    channels: Optional[List[str]] = None,
) -> Union[ProfileResult, ErrorResult]:
    """Profile a time-series dataset behind a file pointer.

Returns factual evidence only; it does not predict, diagnose, or choose a model.

Args:
    dataset_path: Dataset path or `file://` URI.
    timestamp_column: Optional time column name.
    channels: Optional numeric signal columns; omitted means infer usable numeric columns.

Returns:
    ProfileResult: Counts, channels, temporal/statistical evidence, or ErrorResult.
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


@mcp.tool(title="Characterize Series (pattern evidence)")
def characterize_series(
    dataset_path: str,
    timestamp_column: Optional[str] = None,
    channels: Optional[List[str]] = None,
    groups: Optional[dict] = None,
    group_rules: Optional[str] = None,
) -> Union[CharacterizeResult, ErrorResult]:
    """Characterize the shape of a time-series dataset as structured evidence.

Reports grouped-channel states and relationships; it does not assign fault labels.

Args:
    dataset_path: Dataset path or `file://` URI.
    timestamp_column: Optional name of the time column.
    channels: Optional numeric signal columns.
    groups: Optional `{group_name: [channel_names]}` mapping.
    group_rules: Optional grouping preset, such as `"vibration_temperature"`.

Returns:
    CharacterizeResult: Summary, groups, phases, evidence file, or ErrorResult.
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


@mcp.tool(title="Data Quality")
def data_quality(
    dataset_path: str,
    timestamp_column: str = "timestamp",
) -> Union[DataQualityResult, ErrorResult]:
    """Assess data quality for a time-series dataset and produce a cleaned file pointer.

Removes rows that fail TSFM cleaning rules and reports missing-value stats.

Args:
    dataset_path: Dataset path or `file://` URI.
    timestamp_column: Timestamp column name.

Returns:
    DataQualityResult: Cleaned file, row counts, NaN stats, or ErrorResult.
"""
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


_FEATURE_STORE = make_store()


def _validate_feature_kind(kind: Optional[str]) -> Optional[ErrorResult]:
    if kind is not None and kind not in {"extractor", "transform"}:
        return ErrorResult(error="kind must be 'extractor', 'transform', or omitted")
    return None


def _feature_count_message(count: int, kind: Optional[str] = None) -> str:
    label = f"{kind} feature" if kind else "feature"
    if count != 1:
        label = f"{label}s"
    return f"{count} {label}"


def _status_message(status: Optional[str]) -> str:
    return f"with status {status}" if status else "across all statuses"


def _card_with_message(card: dict, message: str) -> CardResult:
    return CardResult(**{**card, "message": message})


@mcp.tool(title="List Feature Catalog")
def list_features(
    kind: Optional[str] = None,
    status: Optional[str] = "active",
) -> Union[FeaturesResult, ErrorResult]:
    """List feature catalog cards from the configured database.

    Args:
        kind: Optional `transform` or `extractor` filter; omit for both.
        status: Optional exact status filter; defaults to `active`.

    Returns:
        FeaturesResult: Matching feature cards and a summary message, or ErrorResult.
    """
    err = _validate_feature_kind(kind)
    if err:
        return err
    try:
        features = feature_store.find_features(_FEATURE_STORE, kind=kind, status=status)
        return FeaturesResult(
            features=features,
            message=(
                f"listed {_feature_count_message(len(features), kind)} "
                f"{_status_message(status)}."
            ),
        )
    except Exception as exc:
        logger.error("list_features failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="List Models")
def list_models(
    task_id: Optional[str] = None, domain: Optional[str] = None, status: str = "active"
) -> Union[ModelsResult, ErrorResult]:
    """List model cards in the catalog, optionally filtered by task / domain.

    Args:
        task_id: Optional known task id, e.g. `tsfm_forecasting`.
        domain: Optional exact domain filter, e.g. `energy`.
        status: Optional lifecycle status; defaults to `active`.

    Returns:
        ModelsResult: Full matching card dicts, or ErrorResult for unknown `task_id`.
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
    text: str, tags: Optional[List[str]] = None, status: str = "active"
) -> Union[ModelsResult, ErrorResult]:
    """Substring (case-insensitive) search over the model catalog.

    Use this for text/tag discovery. For anomaly detection, select a card whose `task_ids`
    include `tsfm_anomaly_detection`, then pass its `model_id` as the `run_recipe` estimator.

    Args:
        text: Required substring to match against id, description, family, or tags.
        tags: Optional required tag set.
        status: Lifecycle status to include; defaults to `active`.

    Returns:
        ModelsResult: Full matching card dicts, or ErrorResult if `text` is empty.
    """
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


@mcp.tool(title="Find Models")
def find_models(
    task_id: str,
    min_context_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    domain: Optional[str] = None,
    top_k: int = 5,
) -> Union[ModelsResult, ErrorResult]:
    """Filter the model catalog for a task and return a ranked shortlist.

    Returns at most `top_k` cards (default 5) - this is a SHORTLIST tool. To enumerate every
    card for a task (e.g. to build a leaderboard or map model ids to their params), use
    `list_models`, which applies no `top_k` limit.

    For anomaly detection, use `task_id="tsfm_anomaly_detection"` and pass the selected
    `model_id` as the `run_recipe` estimator.

    Cards lacking filtered fields are excluded from the shortlist.

    Args:
        task_id: Required known task id.
        min_context_length: Optional minimum `context_length`.
        prediction_length: Optional required horizon.
        domain: Optional exact domain filter.
        top_k: Maximum cards to return; clamped to 1..50.

    Returns:
        ModelsResult: Shortlisted full card dicts, or ErrorResult for unknown `task_id`.
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


@mcp.tool(title="Describe Candidates")
def describe_candidates(
    task_id: str, top_k: int = 5, domain: Optional[str] = None
) -> Union[CandidatesResult, ErrorResult]:
    """Return a shortlist of candidate models for a task, in catalog order.

    No scoring is applied; use `hf_stats` separately for popularity.

    Args:
        task_id: Required known task id.
        top_k: Maximum candidates; clamped to 1..50.
        domain: Optional exact domain filter.

    Returns:
        CandidatesResult: Echoed `task_id` and candidate cards, or ErrorResult.
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


@mcp.tool(title="Describe Models")
def describe_models(model_ids: List[str]) -> Union[DescribeModelsResult, ErrorResult]:
    """Return a compact record for each of the given model ids.

    The by-id detail lookup that pairs with `list_models` / `find_models` (the model-side mirror
    of `describe_features`). For each id it returns a trimmed view — description, family,
    `sktime_class`, `context_length`, domain and tags — rather than the full card. Ids not in the
    catalog are reported separately rather than raising an error.

    Args:
        model_ids: The model ids to describe. Discover valid ids with `list_models` /
            `find_models`. At least one id is required.

    Returns:
        DescribeModelsResult: `models` (a compact record per found id), `unknown` (ids not in the
        catalog), and a summary `message`. ErrorResult if `model_ids` is empty.
    """
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


@mcp.tool(title="Count Models")
def count_models() -> Union[ModelCountResult, ErrorResult]:
    """Count the models in the catalog.

    A quick catalog-size summary over active cards: the total, plus how many support each task (a
    card that lists several `task_ids` is counted under each). Read-only; takes no arguments.

    Returns:
        ModelCountResult: `total`, the number of active models, and `by_task`, a task_id → count
        breakdown (sorted by task id).
    """
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


@mcp.tool(title="List Domains")
def list_domains(task_id: Optional[str] = None) -> Union[DomainsResult, ErrorResult]:
    """List the distinct domains present in the model catalog, with counts.

    Enumerates the values you can pass to the `domain` filter of `list_models` / `find_models` /
    `describe_candidates`, each with the number of models in it (cards without a domain are
    grouped under `unspecified`). Optionally scope the tally to a single task.

    Args:
        task_id: Optional. Restrict the tally to cards whose `task_ids` include this task.
            Validated against the known tasks. None counts across every task.

    Returns:
        DomainsResult: `domains`, a domain → model-count mapping (sorted by domain name).
        ErrorResult if `task_id` is not a known task.
    """
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


@mcp.tool(title="Get Model Lineage")
def get_model_lineage(model_id: str) -> Union[LineageResult, ErrorResult]:
    """Return a model card's lineage.

    Traces two relationships for the card: its fine-tune chain (base-model ancestors and
    fine-tuned descendants, as recorded by `register_finetuned`) and its version links
    (`supersedes` / `superseded_by`, as set by `new_model_version`). Use it to see where a card
    came from and what replaced it.

    Args:
        model_id: Required. The card whose lineage to trace. Discover valid ids with `list_models`.

    Returns:
        LineageResult: the lineage graph for the card — its ancestors, descendants and supersede
        links. ErrorResult if `model_id` is empty.
    """
    if not model_id.strip():
        return ErrorResult(error="model_id is required")
    try:
        return LineageResult(**model_store.get_lineage(_STORE, model_id))
    except Exception as exc:
        logger.error("get_model_lineage failed: %s", exc)
        return ErrorResult(error=str(exc))


# =============================================================================
# Authoring / lifecycle
# =============================================================================


@mcp.tool(title="Register Model")
def register_model(model: dict) -> Union[RegisterResult, ErrorResult]:
    """Register a model card in the catalog.

    The card is schema-validated and stores pointers, never weights. Registering an existing
    `model_id` is REJECTED; use `new_model_version` or `update_model` instead.

    Args:
        model: Card with `model_id`, `description`, `task_ids`, and at least one pointer.

    Returns:
        RegisterResult: `status`, registered `id`, stored `card`, or ErrorResult.
    """
    if not model:
        return ErrorResult(error="model card is required")
    try:
        rec = model_store.register_model(_STORE, model, overwrite=False)
        return RegisterResult(status="registered", id=rec.get("model_id", ""), card=rec)
    except Exception as exc:
        logger.error("register_model failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Model Template")
def model_template() -> ModelTemplateResult:
    """Return the template for authoring a new model card.

    Read this before `register_model` to learn the card shape: which fields are required, which
    are optional, the ways a card can point at a model, and a filled example that registers
    as-is. Static - it reads nothing from the catalog. Use `register_finetuned` for fine-tune
    checkpoints.

    Returns:
        ModelTemplateResult: `required_fields`, `optional_fields`, `pointer_choices` (the ways a
        card can reference a model), `resolution_rules`, and a worked `example`.
    """
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
    """Register a fine-tuned model as a card pointing at its checkpoint.
    `base_model_id` must already be in the catalog with `sktime_class`; the new card inherits
    class/params, sets `params.model_path`, and records lineage.

    Args:
        model_id: Id for the new fine-tuned card.
        checkpoint_path: Fine-tuned weights path; becomes `params.model_path`.
        base_model_id: Existing base card id whose wrapper/params are inherited.
        context_length: Tuned input window.
        prediction_length: Tuned forecast horizon.
        description: Fine-tune description.
        domain: Optional domain tag; defaults to `general`.

    Returns: CardResult: Stored card, FLAT at top level, or ErrorResult.
    """
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


@mcp.tool(title="Update Model")
def update_model(model_id: str, fields: dict) -> Union[CardResult, ErrorResult]:
    """Patch fields on an existing model card.

    `fields` are merged into the card, `updated_at` is stamped, and the result is re-validated
    against the schema, so an invalid patch is rejected. Unknown keys are accepted and stored.

    Args:
        model_id: Id of the card to patch. Use `list_models` to discover valid ids.
        fields: Keys to merge, such as `status`, `tags`, `domain` or `description`.

    Returns:
        CardResult: The updated card, FLAT (fields at the top level). ErrorResult if the model is
        unknown or the patch fails validation.
    """
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
    """Retire a model card by setting `status=deprecated`.

    A soft delete: the card stays in the catalog but drops out of the default active listings
    (`list_models`, `find_models`). Reversible with `update_model(model_id, {"status": "active"})`.

    Args:
        model_id: Id of the card to retire. Use `list_models` to discover valid ids.
        reason: Optional note, stored on the card as `deprecation_reason`.

    Returns:
        CardResult: The deprecated card, FLAT (fields at the top level). ErrorResult if the model
        is unknown.
    """
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
    """Create a successor version of a model card.

    Copies the card, applies `fields`, bumps `version` (its leading integer + 1) and links the
    pair: the successor records `supersedes`, the predecessor is marked `status=superseded` with
    `superseded_by`. Prefer this over re-registering, which would overwrite the original.

    Args:
        model_id: Id of the card to supersede. Use `list_models` to discover valid ids.
        fields: Keys to change on the successor, such as `context_length` or `params`.
        new_model_id: Id for the successor. Defaults to `<model_id>_v<version>`.

    Returns:
        CardResult: The successor's card, FLAT (fields at the top level). ErrorResult if the model
        is unknown.
    """
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


def _unsatisfied_requirements(target) -> List[str]:
    """Requirements an sktime estimator declares but this environment does not satisfy.

    Two separate class tags matter, and BOTH are load-bearing:

      * `python_dependencies` - third-party packages, str or list, possibly with version specs
        ("numpy<2", "transformers[torch]>=4.52.0").
      * `python_version` - an interpreter floor, e.g. TSPulseAnomalyDetector declares ">=3.11".
        This one cannot be fixed by pip installing anything, so reporting only the packages sends
        the caller off to install things that will not help.

    Returns human-readable unsatisfied requirements, or [] when everything is satisfied or the
    class declares nothing. Never raises: a preflight must not be the thing that fails.
    """
    out: List[str] = []

    try:
        from sktime.utils.dependencies import _check_python_version, _check_soft_dependencies
    except Exception:
        return []

    try:
        pyver = target.get_class_tag("python_version", None)
    except Exception:
        pyver = None
    if pyver:
        try:
            if not _check_python_version(target, severity="none"):
                out.append(f"python{pyver} (running {sys.version.split()[0]})")
        except Exception:
            pass

    try:
        deps = target.get_class_tag("python_dependencies", None)
    except Exception:
        deps = None
    if deps:
        if isinstance(deps, str):
            deps = [deps]
        for d in deps:
            try:
                if not _check_soft_dependencies(d, severity="none"):
                    out.append(str(d))
            except Exception:
                pass
    return out


@mcp.tool(title="Resolve Model")
def resolve_model(model_id: str) -> Union[ResolveResult, ErrorResult]:
    """Preflight a model card: check that it can actually be loaded.

    Read-only. Confirms the card exists, that its `sktime_class` is importable, and reports where
    the weights come from (`params.model_path`, `hf_repo` or a checkpoint; none for classical
    models, which fit from scratch). Does NOT download weights or fit. Run it before composing a
    recipe to fail fast on a broken card.

    Args:
        model_id: Id of the card to check. Use `list_models` to discover valid ids.

    Returns:
        ResolveResult: `resolvable` plus a human `reason`, and when resolvable the
        `sktime_class`, `training_regime` and `weights_from`. ErrorResult if `model_id` is empty
        or unknown.
    """
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
        target = R._import_target(sk)  # import only; no instantiation / no weight download
    except Exception as e:
        return ResolveResult(
            model_id=model_id, resolvable=False, sktime_class=sk,
            training_regime=regime, weights_from=weights_from,
            reason=f"sktime_class not importable: {type(e).__name__}: {e}"[:140],
        )

    # Importing the class is NOT enough. sktime estimators wrap third-party libraries and defer the
    # check to fit(), so e.g. ARIMA imports and even CONSTRUCTS without pmdarima installed and only
    # blows up inside the backtest. Every sktime estimator declares its soft deps in the
    # `python_dependencies` class tag, so we can preflight them here for free - which is the whole
    # point of this tool.
    missing = _unsatisfied_requirements(target)
    if missing:
        pipable = [m for m in missing if not m.startswith("python>")]
        hint = ""
        if pipable:
            hint = (" Install with: pip install "
                    + " ".join(f"'{m}'" for m in pipable))
        if len(pipable) < len(missing):
            hint += " (the python version floor cannot be pip-installed)"
        return ResolveResult(
            model_id=model_id, resolvable=False, sktime_class=sk,
            training_regime=regime, weights_from=weights_from,
            reason=(f"sktime_class imports, but requirement(s) not satisfied: "
                    f"{', '.join(missing)}.{hint}")[:300],
        )

    return ResolveResult(
        model_id=model_id, resolvable=True, sktime_class=sk,
        training_regime=regime, weights_from=weights_from or "none (classical; fit from series)",
        reason="importable sktime_class + params, soft deps satisfied; weights load lazily at fit",
    )


# =============================================================================
# Model stats (read-only external lookups)
# =============================================================================


@mcp.tool(title="HF Model Stats")
def hf_stats(
    model_id: Optional[str] = None, hf_repo: Optional[str] = None
) -> Union[HfStatsResult, ErrorResult]:
    """Look up HuggingFace popularity for a model card or repo.

    Pass either a catalog `model_id` with `hf_repo` or an `hf_repo` directly.
    Requires network access to huggingface.co.

    Args:
        model_id: Optional catalog model id to resolve to an HuggingFace repo.
        hf_repo: Optional HuggingFace repository name, e.g. `ibm-granite/...`.

    Returns:
        HfStatsResult: `model_id`, `hf_repo`, `downloads`, `likes`, or ErrorResult.
    """
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


@mcp.tool(title="Count Features")
def count_features() -> Union[FeatureCountResult, ErrorResult]:
    """Count the feature catalog cards by kind.

    Use this for a quick sense of catalog size before browsing with `list_features()`
    or `search_features()`.

    Returns:
        FeatureCountResult: The number of `extractor` cards, `transform` cards, and
        their `total`. Returns ErrorResult if the backing database query fails.
    """
    try:
        ex = len(feature_store.find_features(_STORE, kind="extractor"))
        tr = len(feature_store.find_features(_STORE, kind="transform"))
        return FeatureCountResult(extractors=ex, transforms=tr, total=ex + tr)
    except Exception as exc:
        logger.error("count_features failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Describe Features")
def describe_features(names: List[str]) -> Union[DescribeFeaturesResult, ErrorResult]:
    """Describe specific feature cards by name.

    Use this after `list_features()` or `search_features()` to get a compact record
    for a chosen subset, without pulling each full card. Names that are not extractor
    or transform cards are reported separately rather than raising.

    Args:
        names: Feature ids to describe (extractors or transforms). Discover valid
            names with `list_features()`. At least one name is required.

    Returns:
        DescribeFeaturesResult: `features` (a compact record per found id: `feature_id`,
        `kind`, `name`, `description`), `unknown` (ids not found as a feature card), and
        a summary `message`. Returns ErrorResult if `names` is empty.
    """
    if not names:
        return ErrorResult(error="provide at least one feature name (see list_features)")
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


@mcp.tool(title="Extract Features")
def extract_features(
    dataset_path: str,
    extractors: List[str],
    target_columns: List[str],
    timestamp_column: Optional[str] = None,
    window: Optional[int] = None,
) -> Union[ExtractResult, ErrorResult]:
    """Compute scalar feature values from a series with the named extractors.

    Use this for raw feature extraction with no model attached, e.g. to inspect what a
    set of extractors produces before feeding the values into `run_tabular_recipe`.

    Args:
        dataset_path: File pointer to the input series (as returned by the evidence
            tools or `materialize_iot`).
        extractors: Extractor names to apply. Discover valid names with
            `list_features(kind="extractor")`; an unknown name returns ErrorResult.
        target_columns: The column(s) to extract from. Required; no default column is
            assumed. Each column yields its own `<column>.<extractor>` feature columns.
        timestamp_column: Optional name of the time column, used to order the series.
        window: Windowing. `None` yields one feature vector for the whole series;
            an integer `W` yields a (windows x features) matrix over non-overlapping
            `W`-length tiles.

    Returns:
        ExtractResult: `columns` (the feature-column names), `features` (the value
        matrix, one row per window), `n_windows`, `window`, and a `message`. Returns
        ErrorResult for empty `target_columns`/`extractors`, an unknown extractor, or a
        load failure.
    """
    import numpy as np

    from .reasoning import feature_selection as FS

    if not target_columns:
        return ErrorResult(error="target_columns is required: name the column(s) to extract from")
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
        obj = refs.load_series(dataset_path, time_col=timestamp_column, channels=target_columns)
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


@mcp.tool(title="Select Features")
def select_features(
    dataset_path: str,
    channel: str,
    extractors: List[str],
    timestamp_column: Optional[str] = None,
    reference_feature: str = "mean",
    cd_margin: float = 0.05,
) -> Union[FeatureSelectionResult, ErrorResult]:
    """Rank candidate extractors on one series and return the shortlist worth keeping.

    Use this to narrow a large candidate set to the few extractors that carry signal for
    a given series, before computing them with `extract_features()`. The method is
    self-supervised one-step-ahead forecasting: slide a window over the series and score
    each candidate by how well the window's features predict the next value (no labels
    needed), combining correlation, F-test, mutual information, and model importance by
    mean rank, then keep those that beat `reference_feature` by at least `cd_margin`.

    Args:
        dataset_path: File pointer to the input series.
        channel: The column to analyze. Required; no default column is assumed.
        extractors: Candidate extractor names to score. Discover valid names with
            `list_features(kind="extractor")`; an unknown name returns ErrorResult.
        timestamp_column: Optional name of the time column, used to order the series.
        reference_feature: The baseline extractor a candidate must beat to be kept.
            Defaults to `mean`.
        cd_margin: The minimum margin over `reference_feature` required to keep a
            candidate. Defaults to 0.05.

    Returns:
        FeatureSelectionResult: `selected` (the shortlist, names only), `lookback`,
        `reference`, `scorers`, and a `detail_file` pointer to the full scoring record.
        Returns ErrorResult for a blank `dataset_path`/`channel`, empty `extractors`, or
        an unknown extractor.
    """
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
        obj = refs.load_series(dataset_path, time_col=timestamp_column, channels=[channel])
        series = (obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj).to_numpy()
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


@mcp.tool(title="Search Feature Catalog")
def search_features(
    text: str = "",
    tags: Optional[List[str]] = None,
    status: Optional[str] = "active",
) -> Union[FeaturesResult, ErrorResult]:
    """Search feature catalog cards by id, name, description, or tags.

    The match is literal and case-insensitive, not semantic retrieval.

    Args:
        text: Optional substring; empty means all cards allowed by filters.
        tags: Optional required tag set.
        status: Optional exact status; defaults to `active`.

    Returns:
        FeaturesResult: Matching feature cards and message, or ErrorResult.
    """
    try:
        features = feature_store.search(
            _FEATURE_STORE, text=text, tags=tags, status=status
        )
        criteria = []
        if text:
            criteria.append(f"text '{text}'")
        if tags:
            criteria.append(f"tags {', '.join(tags)}")
        criteria_message = (
            f" matching {' and '.join(criteria)}" if criteria else ""
        )
        return FeaturesResult(
            features=features,
            message=(
                f"found {_feature_count_message(len(features))}{criteria_message} "
                f"{_status_message(status)}."
            ),
        )
    except Exception as exc:
        logger.error("search_features failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Get Feature")
def get_feature(feature_id: str) -> Union[CardResult, ErrorResult]:
    """Return one feature catalog card by feature id.

    Use this after `list_features()` or `search_features()` when the full card
    is needed, including executable transform code and validity metadata.

    Args:
        feature_id: Exact feature id without the database `feature:` prefix, such
            as `efe_time_robust_norm_v1`. Empty input returns ErrorResult.

    Returns:
        CardResult: The stored feature card with database revision metadata
        stripped. Returns ErrorResult when the id is blank, absent, or the
        backing database query fails.
    """
    if not feature_id.strip():
        return ErrorResult(error="feature_id is required")
    try:
        card = feature_store.get_feature(_FEATURE_STORE, feature_id)
        if not card:
            return ErrorResult(error=f"feature '{feature_id}' not found")
        kind = card.get("kind", "feature")
        status = card.get("status", "unknown")
        return _card_with_message(
            card,
            f"found {kind} feature {feature_id} with status {status}.",
        )
    except Exception as exc:
        logger.error("get_feature failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Register Feature")
def register_feature(
    feature: dict,
    overwrite: bool = False,
) -> Union[RegisterResult, ErrorResult]:
    """Register an executable transform feature card.

    Validates schema, required entry points, no in-place mutation, and optional invertibility.

    Args:
        feature: Card with `feature_id`, `interface`, and executable `code`.
        overwrite: Whether to replace an existing `feature_id`.

    Returns:
        RegisterResult: Status, id, stored card, message, or ErrorResult.
    """
    if not feature:
        return ErrorResult(error="feature card is required")
    try:
        rec = feature_store.register_feature(
            _FEATURE_STORE, feature, overwrite=overwrite
        )
        return RegisterResult(
            status="registered",
            id=rec.get("feature_id", ""),
            card=rec,
            message=f"registered feature {rec.get('feature_id', '')}.",
        )
    except Exception as exc:
        logger.error("register_feature failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Update Feature")
def update_feature(feature_id: str, fields: dict) -> Union[CardResult, ErrorResult]:
    """Patch fields on an existing feature catalog card.

    Does not rerun transform-code validation; use versioning for executable code changes.

    Args:
        feature_id: Exact feature id without the `feature:` prefix.
        fields: Non-empty fields to merge into the stored card.

    Returns:
        CardResult: Updated card with `updated_at`, or ErrorResult.
    """
    if not feature_id.strip() or not fields:
        return ErrorResult(error="feature_id and fields are required")
    try:
        card = feature_store.update_feature(_FEATURE_STORE, feature_id, fields)
        return _card_with_message(
            card,
            (
                f"updated feature {feature_id} with {len(fields)} "
                f"field{'s' if len(fields) != 1 else ''}."
            ),
        )
    except Exception as exc:
        logger.error("update_feature failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Deprecate Feature")
def deprecate_feature(
    feature_id: str, reason: Optional[str] = None
) -> Union[CardResult, ErrorResult]:
    """Mark a feature catalog card as deprecated.

    Deprecation keeps the document available for lineage and audit purposes but
    removes it from default `active` list/search results.

    Args:
        feature_id: Exact feature id without the database `feature:` prefix.
            Empty input returns ErrorResult.
        reason: Optional human-readable reason stored as `deprecation_reason`.

    Returns:
        CardResult: The updated feature card with `status=deprecated`. Returns
        ErrorResult when the id is blank, unknown, or the backing database write
        fails.
    """
    if not feature_id.strip():
        return ErrorResult(error="feature_id is required")
    try:
        card = feature_store.deprecate_feature(
            _FEATURE_STORE, feature_id, reason=reason
        )
        reason_message = f" Reason: {reason}." if reason else ""
        return _card_with_message(
            card,
            f"deprecated feature {feature_id}.{reason_message}",
        )
    except Exception as exc:
        logger.error("deprecate_feature failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="New Feature Version")
def new_feature_version(
    feature_id: str,
    fields: Optional[dict] = None,
    new_feature_id: Optional[str] = None,
) -> Union[CardResult, ErrorResult]:
    """Create a successor version for a transform feature.

    Only `kind=transform` cards can be versioned; the predecessor is marked `superseded`.

    Args:
        feature_id: Exact transform feature id to version.
        fields: Optional changes for the successor card.
        new_feature_id: Optional explicit successor id; default is `<feature_id>_v<version>`.

    Returns:
        CardResult: New successor card, or ErrorResult.
    """
    if not feature_id.strip():
        return ErrorResult(error="feature_id is required")
    try:
        card = feature_store.new_version(
            _FEATURE_STORE,
            feature_id,
            fields or {},
            new_feature_id=new_feature_id,
        )
        return _card_with_message(
            card,
            f"created feature version {card.get('feature_id')} from {feature_id}.",
        )
    except Exception as exc:
        logger.error("new_feature_version failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Get Feature Lineage")
def get_feature_lineage(feature_id: str) -> Union[LineageResult, ErrorResult]:
    """Return the parent and descendant chain for a feature catalog card.

    Lineage is most useful for transform cards created through
    `new_feature_version()`. Extractor cards can be queried but typically return
    an empty ancestor and descendant list.

    Args:
        feature_id: Exact feature id without the database `feature:` prefix.
            Empty input returns ErrorResult.

    Returns:
        LineageResult: `feature_id`, ordered ancestor ids, root id, and direct
        descendant ids. Returns ErrorResult when the id is blank or the backing
        database query fails.
    """
    if not feature_id.strip():
        return ErrorResult(error="feature_id is required")
    try:
        lineage = feature_store.get_lineage(_FEATURE_STORE, feature_id)
        ancestors = lineage.get("ancestors", [])
        descendants = lineage.get("descendants", [])
        return LineageResult(
            **{
                **lineage,
                "message": (
                    f"lineage for feature {feature_id} has {len(ancestors)} "
                    f"ancestor{'s' if len(ancestors) != 1 else ''} and "
                    f"{len(descendants)} descendant"
                    f"{'s' if len(descendants) != 1 else ''}; "
                    f"root is {lineage.get('root', feature_id)}."
                ),
            }
        )
    except Exception as exc:
        logger.error("get_feature_lineage failed: %s", exc)
        return ErrorResult(error=str(exc))


# =============================================================================
# Compose and run (file pointers in/out)
# =============================================================================


@mcp.tool(title="Recipe Template")
def recipe_template() -> RecipeTemplateResult:
    """Return the template for authoring a recipe for run_recipe / run_tabular_recipe.

    Read this before run_recipe. A recipe is the agent's decision surface: it names the model and
    every choice around it, and the server executes exactly what it says. Static - it reads nothing
    from the catalog. Pair it with find_models / describe_candidates to choose a `model_id`, and
    resolve_model to preflight that the card loads. For anomaly detection, set
    `recipe["task"] == "tsfm_anomaly_detection"` and provide a detector estimator; `run_recipe`
    will route that recipe to the anomaly detector path.

    Returns:
        RecipeTemplateResult: `task_choices` (what recipe["task"] dispatches on), `estimator_spec`
        (the two ways to name a model), `optional_blocks` (what else a recipe may carry), `rules`,
        and worked `examples` keyed by scenario - each one runs as-is.
    """
    return RecipeTemplateResult(
        task_choices=[
            "omitted / anything else  - FORECASTING: transforms -> single|ensemble -> conformal",
            "tsfm_anomaly_detection   - detector path, producing dense anomaly labels",
            "tsfm_classification | tsfm_regression | tsfm_clustering  - run_tabular_recipe only",
        ],
        estimator_spec=[
            'model_id     - a catalog card, e.g. {"model_id": "ttm_r1_512_96"}; its sktime_class '
            "and params are read from the card (see find_models / resolve_model)",
            'sktime_class - an inline class path + params, e.g. '
            '{"sktime_class": "sktime.forecasting.naive.NaiveForecaster", '
            '"params": {"strategy": "drift"}}',
        ],
        optional_blocks=[
            'fh         - forecast horizon, e.g. [1, 2, 3]. Default [1, 2, 3, 4, 5]',
            'transforms - list of transform specs applied to the target before the forecaster',
            'ensemble   - {"members": [<estimator spec>, ...], "combine": '
            '"mean|median|min|max|weighted|stack", "weights": [...]} - use INSTEAD of estimator',
            'conformal  - {"coverage": 0.9} for calibrated prediction intervals',
            'finetune   - training block; see param_hints (lr, epochs, batch_size, ...)',
            'save_to    - directory path; after a fine-tune, persist the fitted weights there and '
            'return checkpoint_path in the result (feed it to register_finetuned)',
            'anomaly    - detector block (false_alarm, ad_model_type, window_size, ...)',
            'impute     - fill gaps before fitting',
            'eval       - {"metrics": ["smape", ...]} - the first metric scores the backtest',
        ],
        rules=[
            "a recipe MUST carry an `estimator` or an `ensemble` (not both)",
            "`model_id` must exist in the catalog; run_recipe errors if it does not",
            "run_tabular_recipe extracts FLOps features first, then fits `estimator`; omit "
            "label_column for clustering",
            "the series itself is never in the recipe - it arrives via dataset_path",
        ],
        examples={
            "forecast_with_a_catalog_model": {
                "estimator": {"model_id": "ttm_96_28"},
                "fh": [1, 2, 3],
            },
            "forecast_inline_estimator": {
                "estimator": {
                    "sktime_class": "sktime.forecasting.naive.NaiveForecaster",
                    "params": {"strategy": "drift"},
                },
                "fh": [1, 2, 3, 4, 5],
                "eval": {"metrics": ["smape"]},
            },
            "forecast_ensemble_with_intervals": {
                "ensemble": {
                    "members": [
                        {"sktime_class": "sktime.forecasting.naive.NaiveForecaster",
                         "params": {"strategy": "drift"}},
                        {"sktime_class": "sktime.forecasting.naive.NaiveForecaster",
                         "params": {"strategy": "mean"}},
                    ],
                    "combine": "mean",
                },
                "fh": [1, 2, 3],
                "conformal": {"coverage": 0.9},
            },
            "anomaly_detection": {
                "task": "tsfm_anomaly_detection",
                "estimator": {"sktime_class": "sktime.detection.lof.SubLOF",
                              "params": {"window_size": 24, "n_neighbors": 5, "novelty": True}},
            },
            "tabular_classification": {
                "task": "tsfm_classification",
                "estimator": {"sktime_class": "sklearn.ensemble.RandomForestClassifier",
                              "params": {"n_estimators": 100}},
            },
        },
    )


def _index_result(task_type, *, asset_id, results_file, model_id=None,
                  summary=None, metrics=None, scenario_id=None):
    """Record a produced result in its typed collection so list_results / get_result can find it.

    run_recipe already returns the results_file pointer AND persists the run record to tsfm_runs.
    This is the third leg: the per-task result index (forecast_result, anomaly_result, ...) that
    get_result / list_results read. Without it those two tools return nothing. Best-effort: a run
    that succeeded must not fail because its result could not be indexed, so failures are logged,
    not raised.
    """
    try:
        results.write_result(
            _STORE, task_type, asset_id=asset_id, results_file=results_file,
            model_id=model_id, summary=summary or {}, metrics=metrics or [],
            scenario_id=scenario_id,
        )
    except Exception as exc:  # never let indexing sink a good run
        logger.warning("result indexing skipped for %s: %s", task_type, exc)


@mcp.tool(title="Run Recipe")
def run_recipe(
    dataset_path: str,
    timestamp_column: str,
    target_columns: List[str],
    recipe: dict,
    asset_id: str = "asset",
    parent_run_id: Optional[str] = None,
) -> Union[RecipeResult, ErrorResult]:
    """Run a forecasting or anomaly-detection recipe on a target series from a file pointer.

    For anomaly detection, first select a detector with
    `find_models(task_id="tsfm_anomaly_detection")` or `search_models`; call `run_recipe` with
    `recipe={"task": "tsfm_anomaly_detection", "estimator": {"model_id": "<model_id>"}}`.
    The anomaly path returns dense labels, counts, indexed records, and a `results_file` pointer;
    ground final segment/JSON answers in those outputs. Recipes without that task are forecasting
    (transforms + single/ensemble + optional conformal intervals). Use `recipe_template()` for the
    recipe contract. The result is also findable later via `list_runs()` / `get_run()` and
    `list_results()` / `get_result()`.

    Args:
        dataset_path: File pointer to the input series (from the evidence tools or
            `materialize_iot`).
        timestamp_column: Name of the time column used to order the series.
        target_columns: The column(s) to forecast or screen for anomalies. Must not be empty.
        recipe: The recipe dict: an `estimator` (a catalog `model_id`, or an inline
            `sktime_class` + `params`) or an `ensemble`, plus optional blocks (`fh`,
            `transforms`, `conformal`, `finetune`, `save_to`, `eval`, ...). For anomaly detection,
            include `task: "tsfm_anomaly_detection"` and a detector estimator. See
            `recipe_template()`.
        asset_id: Asset this run belongs to; used to group runs and results.
        parent_run_id: Optional id of a parent run, for chaining within a plan.

    Returns:
        RecipeResult: `status`, `run_id`, the engine's full `results` payload, a `results_file`
        pointer, `training_regime`, `folds`, and either `backtest_score`/`metric` (forecasting)
        or `n_anomalies`/`n_observations` (anomaly). Carries `checkpoint_path` when the recipe
        used `save_to`. Returns ErrorResult on empty inputs or a run failure.
    """
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
                    "anomaly_label": res["labels"],
                    "n_anomalies": res["n_anomalies"],
                    "anomaly_indices": res["anomaly_indices_head"],
                },
                name="anomaly",
            )
            _index_result(
                "tsfm_anomaly_detection", asset_id=asset_id, results_file=results_file,
                model_id=(recipe.get("estimator") or {}).get("model_id"),
                summary={"total_records": res["n_observations"],
                         "anomaly_count": res["n_anomalies"]},
            )
            return RecipeResult(
                status="success",
                run_id=res["run_id"],
                results=res,
                results_file=results_file,
                training_regime=res["training_regime"],
                n_anomalies=res["n_anomalies"],
                n_observations=res["n_observations"],
                message=f"Anomaly run complete ({res['training_regime']}): "
                f"{res['n_anomalies']}/{res['n_observations']} flagged. Labels at {results_file}.",
            )
        results_file = refs.write_json(res, name="recipe_run")  # forecasting path
        _index_result(
            "tsfm_forecasting", asset_id=asset_id, results_file=results_file,
            model_id=(recipe.get("estimator") or {}).get("model_id"),
            summary={"horizon": len(recipe.get("fh") or [1, 2, 3, 4, 5])},
            metrics=[{"metric": res.get("metric"), "score": res.get("backtest_score"),
                      "folds": res.get("folds")}],
        )
        extra = {}
        if res.get("checkpoint_path"):
            extra["checkpoint_path"] = res["checkpoint_path"]
        saved = (
            f" Checkpoint saved to {res['checkpoint_path']} "
            f"(register it with register_finetuned)."
            if res.get("checkpoint_path")
            else ""
        )
        return RecipeResult(
            status="success",
            run_id=res["run_id"],
            results=res,
            results_file=results_file,
            metric=res["metric"],
            backtest_score=res["backtest_score"],
            folds=res.get("folds"),
            training_regime=res["training_regime"],
            message=f"Recipe run complete ({res['training_regime']}, "
            f"{res.get('folds')} fold(s)). Record at {results_file}.{saved}",
            **extra,
        )
    except Exception as exc:
        logger.error("run_recipe failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Run Tabular Recipe")
def run_tabular_recipe(
    dataset_path: str,
    recipe: dict,
    label_column: Optional[str] = None,
    asset_id: str = "asset",
) -> Union[TabularResult, ErrorResult]:
    """Run a series-to-tabular recipe: regression, classification, or clustering.

    Each row of the CSV file pointer is one instance; features are extracted (FeatureUnion)
    and passed to the estimator. Omit `label_column` for unsupervised clustering.

    Args:
        dataset_path: File pointer to the tabular CSV (one instance per row).
        recipe: The recipe dict naming the `estimator` and any feature blocks. See
            `recipe_template()`.
        label_column: Name of the target column for supervised tasks (regression /
            classification). Omit for clustering. Returns ErrorResult if named but absent.
        asset_id: Asset this run belongs to; used to group runs and results.

    Returns:
        TabularResult: `status`, `run_id`, the engine's full `results` payload, a `results_file`
        pointer, the `task`, `metric`, `cv_score`, and `n_features`. Returns ErrorResult on a bad
        recipe, a missing `label_column`, or a run failure.
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
        _index_result(
            res["task"], asset_id=asset_id, results_file=results_file,
            summary={"num_classes": int(len(set(y)))} if (res["task"] == "tsfm_classification"
                                                          and y is not None) else {},
            metrics=[{"metric": res.get("metric"), "score": res.get("cv_score")}],
        )
        return TabularResult(
            status="success",
            run_id=res["run_id"],
            results=res,
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
    """Execute a plan: a DAG of recipes chained by file pointers.

    A plan is a HuggingGPT-style task list where each step's output file pointer can feed
    the next step's input, so multi-stage workflows (e.g. clean -> extract -> forecast)
    run as one call. Individual steps are recorded like any other run.

    Args:
        plan_spec: The plan definition: the ordered steps and how their file pointers
            chain. Must not be empty.
        asset_id: Asset this plan belongs to; used to group runs and results.
        scenario_id: Optional scenario tag carried onto the produced results.

    Returns:
        PlanResult: `status`, a `results_file` pointer to the plan record, and the plan's
        per-step outcome fields. Returns ErrorResult on an empty `plan_spec` or a step
        failure.
    """
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
    """Evaluate a recipe GIFT-Eval style across several dataset configs.

    Scores the recipe with seasonal-naive-normalized MASE and CRPS on each config, then
    reports the geometric mean across configs - a leaderboard-comparable summary of how a
    recipe generalizes beyond a single series.

    Args:
        recipe: The recipe to evaluate (same shape as `run_recipe`; see `recipe_template()`).
        configs: The dataset configs to score against. Must not be empty; each names the
            series/horizon/settings for one evaluation.

    Returns:
        EvaluateResult: `status`, a `results_file` pointer to the full scores, a `message`,
        and the per-config plus geo-mean MASE/CRPS (extra fields). Returns ErrorResult on a
        bad recipe, empty `configs`, or an evaluation failure.
    """
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


# =============================================================================
# Results and runs (the ledger)
# =============================================================================


@mcp.tool(title="Get Result")
def get_result(task_type: str, result_id: str) -> Union[ResultRecord, ErrorResult]:
    """Fetch one persisted result by task type and result id.

    A result record is what a run produced and stored: its `results_file` pointer, model
    id, and task summary. Discover ids with `list_results()`.

    Args:
        task_type: The task the result belongs to, e.g. `tsfm_forecasting`.
        result_id: The result id from `list_results()`.

    Returns:
        ResultRecord: The stored result. Returns ErrorResult if no such id exists for that
        task type.
    """
    rec = results.get_result(_STORE, task_type, result_id)
    return ResultRecord(**rec) if rec else ErrorResult(error="not found")


@mcp.tool(title="List Results")
def list_results(
    task_type: str, asset_id: Optional[str] = None, scenario_id: Optional[str] = None
) -> ResultsListResult:
    """List persisted results for a task type, optionally narrowed by asset or scenario.

    Returns compact records; use `get_result()` for the full stored payload of one id.

    Args:
        task_type: The task whose result collection to list, e.g. `tsfm_forecasting` or
            `tsfm_anomaly_detection`. Unknown task types return ErrorResult.
        asset_id: Optional asset filter.
        scenario_id: Optional scenario filter.

    Returns:
        ResultsListResult: `results`, a list of compact result records (each with a
        `result_id`, `results_file` pointer, and summary).
    """
    return ResultsListResult(
        results=results.list_results(
            _STORE, task_type, asset_id=asset_id, scenario_id=scenario_id
        )
    )


@mcp.tool(title="Get Run")
def get_run(run_id: str) -> Union[RunRecord, ErrorResult]:
    """Fetch one run record by id.

    A run record is a single recipe or plan execution with its config and outcome.

    Args:
        run_id: The run id returned by `run_recipe()` / `run_plan()` (e.g. `run:ab12...`).

    Returns:
        RunRecord: The stored run. Returns ErrorResult if the id is unknown.
    """
    rec = _STORE.get(RUNS_COLLECTION, run_id)
    return RunRecord(**rec) if rec else ErrorResult(error="not found")


@mcp.tool(title="List Runs")
def list_runs(asset_id: Optional[str] = None) -> RunsResult:
    """List run records and plans, optionally filtered by asset.

    Runs are individual executions; plans are multi-step sequences. Use `get_run()` for
    one run's full detail.

    Args:
        asset_id: Optional asset filter. Omit to list runs and plans for every asset.

    Returns:
        RunsResult: `runs` (individual executions) and `plans` (multi-step sequences),
        each a list of stored records.
    """
    sel = {"asset_id": asset_id} if asset_id else {}
    return RunsResult(
        runs=_STORE.find(RUNS_COLLECTION, sel), plans=_STORE.find(PLANS_COLLECTION, sel)
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
