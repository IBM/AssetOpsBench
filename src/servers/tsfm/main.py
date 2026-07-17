"""TSFM MCP server — the model catalog.

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
)
from .core.store import make_store
from .io import refs
from .reasoning import dataquality as _dq
from .reasoning import patterns, profile
from .stores import model_store

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


# =============================================================================
# Discovery / read
# =============================================================================


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


@mcp.tool(title="Describe Candidates")
def describe_candidates(
    task_id: str, top_k: int = 5, domain: Optional[str] = None
) -> Union[CandidatesResult, ErrorResult]:
    """CANDIDATE models for a task (HuggingGPT-style shortlist, CATALOG ORDER (no ranking)). A
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


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
