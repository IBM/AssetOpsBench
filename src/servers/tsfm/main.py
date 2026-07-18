"""TSFM MCP server : the model + feature catalogs.

Model cards are catalog DATA, not tools: they live in the CouchDB `model_catalog` collection
(loaded by src/couchdb/init_data.py like every other AssetOpsBench collection) and this surface is
their discovery + lifecycle API. A card is a POINTER — it records how to construct/load a model
(`sktime_class` + `params`, or an `hf_repo` / `artifact_path` / `remote_endpoint` /
`model_checkpoint`), never the weights themselves.

Browse with list_models / search_models / find_models, shortlist with describe_candidates, read
detail with describe_models, then author via model_template + register_model and manage with
update_model / deprecate_model / new_model_version. resolve_model preflights that a card can
actually be loaded.
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
        "The TSFM model catalog. Model cards are catalog data in CouchDB, not tools. A card is a "
        "POINTER to a model: it carries `sktime_class` + `params` (constructed via sktime) and/or "
        "an `hf_repo` / `artifact_path` / `remote_endpoint` / `model_checkpoint`, never weights. "
        "Browse with list_models / search_models / find_models, shortlist with "
        "describe_candidates, read detail with describe_models. Call model_template for the card "
        "shape, register_model / register_finetuned to add, resolve_model to preflight that a card "
        "can be loaded, and update_model / deprecate_model / new_model_version for lifecycle."
    ),
)

# The catalog is an ordinary AssetOpsBench CouchDB collection (model_catalog), loaded by
# src/couchdb/init_data.py. The server reads it; it does not seed.
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

Caveat: this tool is read-only and does not inspect any dataset or model card.

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

This tool summarizes the observed series structure and basic statistical evidence, such as
observation count, channel count, and any detectable temporal characteristics. It is intended
to provide factual context for downstream reasoning, not a prediction or diagnosis.

Caveat: the `channels` argument must reference valid numeric measurement columns only. Metadata
columns such as identifiers, labels, or non-numeric fields should not be passed as channels.
If `channels` is omitted, the tool will infer usable numeric columns automatically.

Args:
    dataset_path: File pointer to the dataset, typically a `file://...` URI.
    timestamp_column: Optional name of the time column. If omitted, the tool may auto-detect
        common timestamp names.
    channels: Optional list of numeric signal columns to profile. If provided, each channel must
        exist in the dataset and be numeric.

Returns:
    ProfileResult: Summary evidence for the dataset, including observation count, channel count,
    inferred temporal characteristics, and the profiled channel list. ErrorResult if the file
    pointer is invalid or the series cannot be profiled.
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

This tool extracts pattern evidence from the series, describing how grouped channels behave over
time across changepoint phases. It reports state labels such as stable, rise, decline, spike,
level_shift, cessation, or oscillation, and may also describe relationships between groups such
as decoupled, co_move, or lead_lag. The output is intended as evidence for later reasoning,
not as a direct fault label.

Caveat: the tool is sensitive to the input shape and channel selection. If `channels` is
provided, it must refer only to valid numeric measurement columns. If `groups` is omitted, the
tool defaults to one group per channel unless `group_rules` is supplied. Large or very wide
datasets may be slower to process, so a compact sample is often better for smoke tests.

Args:
    dataset_path: File pointer to the dataset, typically a `file://...` URI.
    timestamp_column: Optional name of the time column.
    channels: Optional list of numeric signal columns to include in the characterization.
    groups: Optional explicit channel-group mapping of the form `{group_name: [channel_names]}`.
    group_rules: Optional grouping preset or grouping rules name, such as
        `"vibration_temperature"`.

Returns:
    CharacterizeResult: Structured pattern evidence including summary text, number of
    observations, detected groups, phase-level states, and an evidence file pointer. ErrorResult
    if the dataset cannot be loaded or the characterization fails.
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

This tool removes rows that fail the cleaning rules used by the TSFM workflow and returns a
quality summary, including input/output row counts and per-column missing-value statistics.
It is meant to prepare data for downstream forecasting or anomaly tasks.

Caveat: this tool expects a dataset file pointer and a valid timestamp column name. Very sparse
or malformed datasets may be reduced substantially by the cleaning step, and the cleaned output
may be much smaller than the input.

Args:
    dataset_path: File pointer to the dataset, typically a `file://...` URI.
    timestamp_column: Name of the timestamp column to use when interpreting the series.

Returns:
    DataQualityResult: Cleaning summary including a cleaned file pointer, input and output row
    counts, a message, and missing-value statistics. ErrorResult if the dataset cannot be read
    or cleaned.
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

    Use this to browse candidate transform or extractor cards before choosing a
    feature for a workflow.

    Args:
        kind: Optional exact feature kind filter. Use `transform` for executable
            fit/transform programs, `extractor` for scalar extractor metadata, or
            omit for both. Any other value returns ErrorResult.
        status: Optional exact status filter. Defaults to `active`; pass null or
            an empty string to include deprecated and superseded cards.

    Returns:
        FeaturesResult: Matching feature cards as stored in the catalog. Each card
        includes fields such as `feature_id`, `kind`, `status`, `description`,
        and any card-specific metadata.
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

    The broad browse entry point for the model catalog: returns full cards, unranked, with no
    filtering by default. Pair it with `describe_models` for a by-id detail lookup, or use
    `find_models` / `search_models` when you need task ranking or text matching. Only `active`
    cards are returned unless another status is requested.

    Args:
        task_id: Restrict to cards whose `task_ids` include this task, e.g. `tsfm_forecasting`.
            Validated against the known tasks. None returns every task.
        domain: Restrict to a single domain (exact match), e.g. `energy`. Use `list_domains` to
            see valid values. None returns every domain.
        status: Lifecycle status to include; defaults to `active` (deprecated cards are hidden
            unless you pass `deprecated`).

    Returns:
        ModelsResult: `models`, a list of full card dicts matching the filters. ErrorResult if
        `task_id` is not a known task.
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

    Matches `text` case-insensitively against each card's `model_id`, `description`,
    `model_family` and `tags`. Use this when you have a keyword in mind (a family name, vendor,
    or capability such as `anomaly`); use `list_models` to browse everything, or `find_models`
    for a task-ranked shortlist.

    Args:
        text: Required. The substring to match against id / description / family / tags.
        tags: Optional list of tags; when given, a card must also carry these tags to be returned.
        status: Lifecycle status to include; defaults to `active`.

    Returns:
        ModelsResult: `models`, a list of full card dicts whose fields contain `text`. ErrorResult
        if `text` is empty.
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

    The task-aware selector: narrows to cards that support `task_id`, applies the optional
    capability filters, and returns at most `top_k` cards. A card lacking a filtered field is
    excluded (e.g. classical models have no `context_length`, so `min_context_length` drops
    them). Use `describe_candidates` for an unranked shortlist, or `list_models` for the unranked
    full list. A model is an estimator card.

    Args:
        task_id: Required. Only cards whose `task_ids` include this task are considered. Validated
            against the known tasks.
        min_context_length: Keep only models whose `context_length` is at least this value; cards
            without the field are excluded.
        prediction_length: Keep only models whose `prediction_length` covers this horizon; cards
            without the field are excluded.
        domain: Exact-match domain filter, e.g. `energy`. Use `list_domains` for valid values.
        top_k: Maximum number of cards to return; clamped to the range 1..50 (default 5).

    Returns:
        ModelsResult: `models`, the ranked shortlist of full card dicts (up to `top_k`).
        ErrorResult if `task_id` is not a known task.
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

    A HuggingGPT-style candidate list: the cards that support `task_id`, capped at `top_k` and
    presented in CATALOG ORDER (no ranking or scoring is applied — you decide which to use). To
    judge popularity or quality yourself, follow up with `hf_stats` / `gift_status`. Use
    `find_models` instead when you want capability filters and ranking.

    Args:
        task_id: Required. Only cards whose `task_ids` include this task are shortlisted.
            Validated against the known tasks.
        top_k: Maximum number of candidates to return; clamped to the range 1..50 (default 5).
        domain: Optional exact-match domain filter, e.g. `energy`.

    Returns:
        CandidatesResult: `task_id` (echoed) and `candidates`, the list of candidate card dicts in
        catalog order. ErrorResult if `task_id` is not a known task.
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

    The card is schema-validated before storage. A card is a POINTER: it records how to construct
    or load a model, never the weights themselves. Use `register_finetuned` for fine-tune
    checkpoints, and `new_model_version` to supersede a card rather than replace it.

    Registering an existing `model_id` is REJECTED rather than overwriting it; use
    `new_model_version` to supersede a card, or `update_model` to patch one.

    Args:
        model: The card to store. Required keys: `model_id`, `description` (>= 3 chars) and
            `task_ids` (>= 1). Point it at the model with `sktime_class` (+ `params`) and/or
            `hf_repo` / `artifact_path` / `remote_endpoint` / `model_checkpoint`. Call
            `model_template()` for the full shape and a worked example. Unknown keys are kept
            as-is, so typos are stored silently.

    Returns:
        RegisterResult: `status`, the registered `id`, and the stored `card`. ErrorResult if the
        card fails schema validation.
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

    Inherits the base card's `sktime_class` and `params`, then sets `params.model_path` to
    `checkpoint_path` so the fine-tuned weights load at fit time. Records `provenance=finetuned`
    and lineage back to the base, readable via `get_model_lineage`.

    `base_model_id` must already be in the catalog and carry an `sktime_class`; if it does not,
    this errors rather than guessing a wrapper class, which would silently produce a card that
    loads the wrong architecture.

    Args:
        model_id: Id for the new fine-tuned card.
        checkpoint_path: Directory holding the fine-tuned weights; becomes `params.model_path`.
        base_model_id: Id of the card this was fine-tuned from; its wrapper class and params are
            inherited. Should exist in the catalog (see `list_models`).
        context_length: Input window the checkpoint was tuned for.
        prediction_length: Forecast horizon the checkpoint was tuned for.
        description: What it was tuned on and for. At least 3 characters.
        domain: Optional domain tag, such as `energy`. Defaults to `general`.

    Returns:
        CardResult: The stored card, FLAT - its fields sit at the top level, unlike
        `register_model` which nests the card under `card`. ErrorResult on validation failure.
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


@mcp.tool(title="Search Feature Catalog")
def search_features(
    text: str = "",
    tags: Optional[List[str]] = None,
    status: Optional[str] = "active",
) -> Union[FeaturesResult, ErrorResult]:
    """Search feature catalog cards by id, name, description, or tags.

    The search is a case-insensitive literal substring match, not semantic
    retrieval. Use `list_features()` when you need a complete catalog browse.

    Args:
        text: Optional substring to match against `feature_id`, `name`,
            `description`, and tags. Empty string returns all cards allowed by
            the status and tag filters.
        tags: Optional list of tags that must all be present on a card.
        status: Optional exact status filter. Defaults to `active`; pass null or
            an empty string to search every status.

    Returns:
        FeaturesResult: Matching feature cards. The result can be empty when no
        cards satisfy the filters.
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

    Registration is for `kind=transform` cards only. The card is validated
    against the feature schema and its `code` is executed through the feature
    runner to verify required entry points, no in-place mutation, and optional
    invertibility.

    Args:
        feature: Feature card payload. Required fields are `feature_id`,
            `interface`, and `code`; optional fields include `name`,
            `description`, `target_task`, `target_model`, `tags`, and
            `invertible`.
        overwrite: When false, an existing `feature_id` returns ErrorResult.
            When true, the card replaces the existing document.

    Returns:
        RegisterResult: Registration status, feature id, and the stored card.
        Returns ErrorResult for missing payload, schema errors, failed execution
        validation, duplicate ids, or database write failures.
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

    This is a direct metadata patch for catalog maintenance. It does not rerun
    transform-code validation, so use `register_feature()` or
    `new_feature_version()` when changing executable transform code.

    Args:
        feature_id: Exact feature id without the database `feature:` prefix.
            Empty input returns ErrorResult.
        fields: Non-empty mapping of fields to merge into the stored card.

    Returns:
        CardResult: The updated feature card, including `updated_at`. Returns
        ErrorResult when inputs are blank, the feature does not exist, or the
        backing database write fails.
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

    Only `kind=transform` cards can be versioned. The new card is validated like
    `register_feature()`, receives a bumped `version`, points back through
    `parent_feature_id`, and the predecessor is marked `superseded`.

    Args:
        feature_id: Exact transform feature id to version. Empty input returns
            ErrorResult.
        fields: Optional mapping of changes to apply to the successor card.
        new_feature_id: Optional explicit id for the successor. When omitted,
            the id is generated as `<feature_id>_v<version>`.

    Returns:
        CardResult: The newly stored successor feature card. Returns ErrorResult
        for blank or unknown ids, extractor cards, validation failures,
        duplicate successor ids, or database write failures.
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
    resolve_model to preflight that the card loads.

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
            _index_result(
                "tsfm_anomaly_detection", asset_id=asset_id, results_file=results_file,
                model_id=(recipe.get("estimator") or {}).get("model_id"),
                summary={"total_records": res["n_observations"],
                         "anomaly_count": res["n_anomalies"]},
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
        _index_result(
            res["task"], asset_id=asset_id, results_file=results_file,
            summary={"num_classes": int(len(set(y)))} if (res["task"] == "tsfm_classification"
                                                          and y is not None) else {},
            metrics=[{"metric": res.get("metric"), "score": res.get("cv_score")}],
        )
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


# =============================================================================
# Results and runs (the ledger)
# =============================================================================


@mcp.tool(title="Get Result")
def get_result(task_type: str, result_id: str) -> Union[ResultRecord, ErrorResult]:
    """Fetch one persisted result by task_type + result_id (the record a run produced and stored).
    Returns the stored result or an error if no such id exists for that task_type."""
    rec = results.get_result(_STORE, task_type, result_id)
    return ResultRecord(**rec) if rec else ErrorResult(error="not found")


@mcp.tool(title="List Results")
def list_results(
    task_type: str, asset_id: Optional[str] = None, scenario_id: Optional[str] = None
) -> ResultsListResult:
    """List persisted results for a task_type, optionally narrowed by asset_id / scenario_id.
    Compact records; use get_result for the full stored payload of one id."""
    return ResultsListResult(
        results=results.list_results(
            _STORE, task_type, asset_id=asset_id, scenario_id=scenario_id
        )
    )


@mcp.tool(title="Get Run")
def get_run(run_id: str) -> Union[RunRecord, ErrorResult]:
    """Fetch one run record by run_id: a single recipe/plan execution with its config and outcome.
    Returns the stored run or an error if the id is unknown."""
    rec = _STORE.get(RUNS_COLLECTION, run_id)
    return RunRecord(**rec) if rec else ErrorResult(error="not found")


@mcp.tool(title="List Runs")
def list_runs(asset_id: Optional[str] = None) -> RunsResult:
    """List run records and plans, optionally filtered by asset_id. Runs are individual executions;
    plans are multi-step sequences. Use get_run for one run's full detail."""
    sel = {"asset_id": asset_id} if asset_id else {}
    return RunsResult(
        runs=_STORE.find(RUNS_COLLECTION, sel), plans=_STORE.find(PLANS_COLLECTION, sel)
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()