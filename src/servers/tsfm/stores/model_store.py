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
        sel["task_ids"] = {"$elemMatch": task_id}
    if usage_mode:
        sel["usage_modes"] = {"$elemMatch": usage_mode}
    return store.find(COLLECTION, sel)


def _best_metric(m: dict, name: str):
    vals = [
        x["value"]
        for x in m.get("metrics", [])
        if x.get("metric") == name and x.get("value") is not None
    ]
    return min(vals) if vals else None


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
    """Filter by task + structured constraints, rank, return top_k.

    Ranking (lexicographic, lower=better): domain match → has eval MAE (lower) → longer
    context → fewer-params proxy (shorter id). With explain=True, attaches a `_rank` reason.
    """
    cands = list_models(
        store, task_id=task_id, domain=None, modality=modality, usage_mode=usage_mode
    )

    def ok(m):
        if min_context_length and (m.get("context_length") or 0) < min_context_length:
            return False
        if prediction_length and (m.get("prediction_length") or 0) < prediction_length:
            return False
        return True

    cands = [m for m in cands if ok(m)]

    def score(m):
        domain_match = 0 if (domain and m.get("domain") == domain) else 1
        mae = _best_metric(m, "mae")
        return (
            domain_match,
            mae if mae is not None else float("inf"),
            -(m.get("context_length") or 0),
        )

    ranked = sorted(cands, key=score)[:top_k]
    if explain:
        for r in ranked:
            r["_rank"] = {
                "domain_match": bool(domain and r.get("domain") == domain),
                "mae": _best_metric(r, "mae"),
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
    """HuggingGPT-style model selection surface: return the top_k candidate cards for a task as
    compact {model_id, description, family, context_length} records, ranked by an eval-quality
    prior (lower MAE first), the '{{Candidate Models}}' the agent reasons over to pick/ensemble.
    Trims tokens like HuggingGPT's top-K shortlist.
    """
    cands = list_models(store, task_id=task_id, domain=domain)

    def prior(m):
        mae = _best_metric(m, "mae")
        return (mae if mae is not None else float("inf"),)  # better eval first

    out = []
    for m in sorted(cands, key=prior)[:top_k]:
        out.append(
            {
                "model_id": m["model_id"],
                "description": m.get("description", ""),
                "family": m.get("family") or m.get("model_family"),
                "sktime_class": m.get("sktime_class"),
                "context_length": m.get("context_length"),
                "tags": m.get("tags", []),
            }
        )
    return out


def get_lineage(store, model_id: str) -> dict:
    """Ancestors (base chain) + descendants (fine-tunes of this)."""
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
    }


# --------------------------------------------------------------------------- #
# write
# --------------------------------------------------------------------------- #
def register_model(store, model: dict, *, overwrite: bool = False) -> dict:
    doc = schemas.validate_model(model)  # raises on invalid
    if store.get(COLLECTION, doc["_id"]) and not overwrite:
        raise ValueError(
            f"model '{doc['model_id']}' exists (overwrite=True or new_version)"
        )
    doc.setdefault("created_at", _now())
    return store.put(COLLECTION, doc)


def update_model(store, model_id: str, fields: dict) -> dict:
    doc = get_model(store, model_id)
    if not doc:
        raise ValueError(f"no model {model_id}")
    if "metrics" in fields:  # append, don't replace
        doc["metrics"] = doc.get("metrics", []) + list(fields.pop("metrics"))
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
    metrics: Optional[list] = None,
    overwrite: bool = True,
) -> dict:
    """Agent-decided write-back: point the catalog at a fine-tune checkpoint location."""
    return register_model(
        store,
        {
            "model_id": model_id,
            "model_checkpoint": checkpoint_path,
            "artifact_path": checkpoint_path,
            "source": "local_artifact",
            "framework": "tinytimemixer",
            "modality": "timeseries",
            "provenance": "finetuned",
            "base_model_id": base_model_id,
            "task_ids": ["tsfm_forecasting", "tsfm_forecasting_evaluation"],
            "context_length": context_length,
            "prediction_length": prediction_length,
            "domain": domain,
            "description": description,
            "metrics": metrics or [],
            "created_by": "agent.tsfm.finetune",
        },
        overwrite=overwrite,
    )