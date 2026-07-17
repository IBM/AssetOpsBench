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
    """List the 8 standardized TS-AI TASKS (forecasting, regression, classification, anomaly,
    imputation, evaluation, similarity_search, clustering). Each entry has a plain `description`
    plus its contract (required inputs, output, eval protocol). Start here, then profile_series.
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
