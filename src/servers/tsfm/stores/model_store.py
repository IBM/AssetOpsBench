"""Model store: registry of TS models on the core Store.

Pointer index: weights live at artifact_path / hf_repo / remote_endpoint / model_checkpoint
(toolkit); the catalog points at them. Capabilities:
  read    : get / list / find_models (explainable ranking) / search / get_lineage
  write   : register (schema-validated) / update / deprecate / new_version / register_finetuned
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core import schemas

COLLECTION = "model_catalog"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id(model_id: str) -> str:
    return f"model:{model_id}"


def _next_version(v) -> str:
    """Bump a version string robustly: take its leading integer (else 1) and add 1. Handles
    non-numeric ('r2') and null versions without crashing on int()."""
    head = re.match(r"\d+", str(v or "1"))
    return str((int(head.group()) if head else 1) + 1)


# --------------------------------------------------------------------------- #
# read
# --------------------------------------------------------------------------- #
def get_model(store, model_id: str) -> Optional[dict]:
    return store.get(COLLECTION, _id(model_id))


def list_models(
    store,
    task_id: Optional[str] = None,
    domain: Optional[str] = None,
    modality: Optional[str] = None,
    framework: Optional[str] = None,
    usage_mode: Optional[str] = None,
    status: str = "active",
) -> List[dict]:
    sel: Dict = {}
    if status:
        sel["status"] = status
    if domain:
        sel["domain"] = domain
    if modality:
        sel["modality"] = modality
    if framework:
        sel["framework"] = framework
    if task_id:
        # CouchDB requires a condition object here; a bare scalar is a 400.
        sel["task_ids"] = {"$elemMatch": {"$eq": task_id}}
    if usage_mode:
        sel["usage_modes"] = {"$elemMatch": {"$eq": usage_mode}}
    return store.find(COLLECTION, sel)


def find_models(
    store,
    task_id: str,
    *,
    min_context_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    domain: Optional[str] = None,
    modality: Optional[str] = None,
    usage_mode: Optional[str] = None,
    top_k: int = 5,
    explain: bool = False,
):
    """Filter by task + structured constraints and return the first top_k in CATALOG ORDER (no
    ranking). Filters: domain (exact), min_context_length / prediction_length (models lacking that
    field are excluded). explain=True attaches a `_filter` note. Use hf_stats / gift_status to
    judge quality/popularity yourself."""
    cands = list_models(
        store, task_id=task_id, domain=domain, modality=modality, usage_mode=usage_mode
    )

    def ok(m):
        if min_context_length and (m.get("context_length") or 0) < min_context_length:
            return False
        if prediction_length and (m.get("prediction_length") or 0) < prediction_length:
            return False
        return True

    ranked = [m for m in cands if ok(m)][:top_k]   # catalog order, no ranking
    if explain:
        for r in ranked:
            r["_filter"] = {
                "domain_match": bool(domain and r.get("domain") == domain),
                "context_length": r.get("context_length"),
            }
    return ranked


def search(
    store, text: str = "", *, tags: Optional[List[str]] = None, status: str = "active"
) -> List[dict]:
    """Free-text over id/description/family/tags + optional tag filter."""
    text = (text or "").lower()
    out = []
    for m in list_models(store, status=status):
        hay = " ".join(
            [
                m.get("model_id") or "",
                m.get("description") or "",
                m.get("model_family") or "",
                " ".join(m.get("tags") or []),
            ]
        ).lower()
        if text and text not in hay:
            continue
        if tags and not set(tags) <= set(m.get("tags", [])):
            continue
        out.append(m)
    return out


def describe_candidates(
    store, task_id: str, *, top_k: int = 5, domain: str = None
) -> list:
    """HuggingGPT-style model selection surface: return the first top_k candidate cards for a task
    (catalog order, no ranking) as compact records, the '{{Candidate Models}}' the agent reasons
    over. Use hf_stats / gift_status to judge popularity/quality. Trims tokens for the shortlist.
    """
    cands = list_models(store, task_id=task_id, domain=domain)
    out = []
    for m in cands[:top_k]:   # catalog order, no ranking
        out.append(
            {
                "model_id": m["model_id"],
                "description": m.get("description", ""),
                "family": m.get("model_family") or m.get("family"),
                "sktime_class": m.get("sktime_class"),
                "context_length": m.get("context_length"),
                "tags": m.get("tags", []),
            }
        )
    return out


def get_lineage(store, model_id: str) -> dict:
    """Fine-tune lineage: base-model ancestors + fine-tune descendants, plus the version links
    (supersedes / superseded_by) set by new_version."""
    card = get_model(store, model_id) or {}
    ancestors, cur = [], get_model(store, model_id)
    seen = set()
    while cur and cur.get("base_model_id") and cur["base_model_id"] not in seen:
        seen.add(cur["base_model_id"])
        parent = get_model(store, cur["base_model_id"])
        if not parent:
            break
        ancestors.append(parent["model_id"])
        cur = parent
    descendants = [
        m["model_id"] for m in store.find(COLLECTION, {"base_model_id": model_id})
    ]
    return {
        "model_id": model_id,
        "ancestors": ancestors,
        "root": ancestors[-1] if ancestors else model_id,
        "descendants": descendants,
        "supersedes": card.get("supersedes"),
        "superseded_by": card.get("superseded_by"),
    }


def _hf_model_stats(repo: str) -> dict:
    """Fetch HuggingFace popularity for a repo: {downloads, likes}. Network I/O; a small seam so it
    can be monkeypatched/cached. Requires access to huggingface.co."""
    import requests

    r = requests.get(
        f"https://huggingface.co/api/models/{repo}",
        headers={"Accept": "application/json"},
        timeout=15,
    )
    r.raise_for_status()
    j = r.json()
    return {"downloads": j.get("downloads"), "likes": j.get("likes")}


def _leaderboard_stats(url: str) -> dict:
    """Fetch leaderboard standings as {key: {rank, score}} (key matches leaderboard_id / hf_repo /
    model_id). Network I/O; monkeypatchable. Accepts a list of {key|model|id, rank, score} or a
    mapping key -> {rank, score}."""
    import requests

    r = requests.get(url, headers={"Accept": "application/json"}, timeout=20)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, list):
        return {
            str(e.get("key") or e.get("model") or e.get("id")): {
                "rank": e.get("rank"), "score": e.get("score")
            }
            for e in j
        }
    return j


# --------------------------------------------------------------------------- #
# write
# --------------------------------------------------------------------------- #
def register_model(store, model: dict, *, overwrite: bool = False) -> dict:
    doc = schemas.validate_model(model)  # raises on invalid
    if store.get(COLLECTION, doc["_id"]) and not overwrite:
        raise ValueError(
            f"model '{doc['model_id']}' already exists; use new_model_version to supersede it, "
            "or update_model to patch it"
        )
    doc.setdefault("created_at", _now())
    return store.put(COLLECTION, doc)


def update_model(store, model_id: str, fields: dict) -> dict:
    doc = get_model(store, model_id)
    if not doc:
        raise ValueError(f"no model {model_id}")
    doc.update(fields)
    doc["updated_at"] = _now()
    return store.put(COLLECTION, schemas.validate_model(doc))


def deprecate_model(store, model_id: str, reason: Optional[str] = None) -> dict:
    return update_model(
        store, model_id, {"status": "deprecated", "deprecation_reason": reason}
    )


def new_version(
    store, model_id: str, fields: dict, *, new_model_id: Optional[str] = None
) -> dict:
    """Register a new version; the predecessor is marked superseded and linked."""
    old = get_model(store, model_id)
    if not old:
        raise ValueError(f"no model {model_id}")
    nv = dict(old, **fields)
    nv["version"] = _next_version(old.get("version"))
    nv["model_id"] = new_model_id or f"{model_id}_v{nv['version']}"
    nv["supersedes"] = model_id
    nv.pop("_id", None)
    nv.pop("updated_at", None)
    out = register_model(store, nv, overwrite=True)
    update_model(
        store, model_id, {"status": "superseded", "superseded_by": out["model_id"]}
    )
    return out


def _has_param(sktime_class: str, param: str) -> bool:
    """Does this sktime class's constructor accept `param`? False if it cannot be inspected."""
    import inspect

    try:
        from ..substrate import resolver as R

        target = R._import_target(sktime_class)
        return param in inspect.signature(target.__init__).parameters
    except Exception:
        return False


def _require_model_path_param(sktime_class: str, base_model_id: str) -> None:
    """A checkpoint card is served via params.model_path, so the wrapper must accept one."""
    import inspect

    try:
        from ..substrate import resolver as R

        target = R._import_target(sktime_class)
        sig = inspect.signature(target.__init__).parameters
    except Exception:
        return  # uninspectable (e.g. optional dep absent): do not block on a check we cannot run
    if "model_path" in sig or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values()
    ):
        return
    raise ValueError(
        f"base model '{base_model_id}' uses {sktime_class}, whose constructor takes no "
        f"'model_path'. A fine-tuned card loads its weights via params.model_path, so this card "
        f"would raise TypeError at resolve(). register_finetuned is for checkpoint-backed wrappers "
        f"(TinyTimeMixerForecaster, PatchTSTForecaster, ChronosForecaster, ...)."
    )


def register_finetuned(
    store,
    *,
    model_id: str,
    checkpoint_path: str,
    base_model_id: str,
    context_length: int,
    prediction_length: int,
    description: str,
    domain: str = "general",
    overwrite: bool = True,
) -> dict:
    """Register a fine-tuned model that points the catalog at a checkpoint. Inherits the sktime
    wrapper class from the base model and sets params.model_path to the checkpoint, so the
    fine-tuned weights are resolvable (loaded from checkpoint_path at fit), with lineage to base."""
    base = get_model(store, base_model_id)
    if not base:
        raise ValueError(
            f"base model '{base_model_id}' is not in the catalog; register it first "
            "(see list_models). Inventing a wrapper class here would silently produce a card "
            "that loads the wrong architecture."
        )
    sktime_class = base.get("sktime_class")
    if not sktime_class:
        raise ValueError(
            f"base model '{base_model_id}' has no sktime_class to inherit; the fine-tuned card "
            "would not be resolvable"
        )

    # The card loads weights through params.model_path, so the base's wrapper must actually TAKE a
    # model_path. Without this check we happily emit a card that raises TypeError at resolve():
    #   ThetaForecaster.__init__() got an unexpected keyword argument 'model_path'
    _require_model_path_param(sktime_class, base_model_id)

    params = {**(base.get("params") or {}), "model_path": checkpoint_path}
    # A checkpoint card SERVES: load the fine-tuned weights and predict. It must not train again.
    # sktime's TTM defaults to fit_strategy="minimal", which fine-tunes - so an unpinned checkpoint
    # card would fine-tune the already-fine-tuned weights on every fit.
    if _has_param(sktime_class, "fit_strategy"):
        params["fit_strategy"] = "zero-shot"
    return register_model(
        store,
        {
            "model_id": model_id,
            "sktime_class": sktime_class,
            "params": params,
            "model_checkpoint": checkpoint_path,
            "artifact_path": checkpoint_path,
            "source": "local_artifact",
            "framework": base.get("framework", "tinytimemixer"),
            "modality": "timeseries",
            "provenance": "finetuned",
            # provenance is HISTORY (where these weights came from); training_regime is what
            # happens on the NEXT fit(). Serving a checkpoint is load-and-predict = zero_shot.
            # Leaving this unset let the card infer to fine_tune, which sent run_recipe down the
            # ~20-fold refit path to serve a model that does not train.
            "training_regime": "zero_shot",
            "base_model_id": base_model_id,
            "task_ids": ["tsfm_forecasting"],
            "context_length": context_length,
            "prediction_length": prediction_length,
            "domain": domain,
            "description": description,
            "created_by": "agent.tsfm.finetune",
        },
        overwrite=overwrite,
    )