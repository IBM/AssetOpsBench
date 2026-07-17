"""TSFM MCP server for task evidence, model cards, and feature cards.

Catalogs are CouchDB data loaded by src/couchdb/init_data.py; `MODEL_CATALOG_DBNAME` and
`FEATURE_CATALOG_DBNAME` select the collections. Model cards are pointers to loadable models,
never weights.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Union

import pandas as pd
from mcp.server.fastmcp import FastMCP

from .core import tasks as task_spec
from .core.results_models import (
    CandidatesResult,
    CharacterizeResult,
    CardResult,
    DataQualityResult,
    DescribeModelsResult,
    DomainsResult,
    ErrorResult,
    HfStatsResult,
    LineageResult,
    ModelCountResult,
    ModelDescription,
    ModelsResult,
    ModelTemplateResult,
    ProfileResult,
    RegisterResult,
    ResolveResult,
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
from .stores import feature_store

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("tsfm-mcp-server")

mcp = FastMCP(
    "tsfm",
    instructions=(
        "The TSFM server provides task discovery, file-pointer evidence tools, and CouchDB-backed "
        "model and feature catalog tools. Use the model tools to browse, resolve, register, "
        "fine-tune, version, or deprecate model cards; use the feature tools to browse, register, "
        "version, or deprecate transform/extractor cards. Catalog cards are data, not weights; "
        "`MODEL_CATALOG_DBNAME` and `FEATURE_CATALOG_DBNAME` select the backing collections."
    ),
)

# The catalog is an ordinary AssetOpsBench CouchDB collection (MODEL_CATALOG_DBNAME, default
# model_catalog), loaded by src/couchdb/init_data.py. The server reads it; it does not seed.
_STORE = make_store()


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


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
