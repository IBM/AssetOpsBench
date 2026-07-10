"""Feature store — transforms (EFE) + extractors (FLOps) on the core Store.

Two entry kinds in one catalog:
  - kind="transform": EFE-style fit/transform/inverse programs stored as code (validated +
    EFE validity-gated on register).
  - kind="extractor": FLOps scalar extractors (the 130+ library); the executable lives in
    feature_selection.EXTRACTORS, the catalog indexes them so select_features can pick.

Capabilities:
  read    : get / find_features / list_extractors / search / get_lineage
  write   : register_feature (validated+gated) / update / deprecate / new_version
            register_extractor / register_extractor_library
  learn   : select_features (FLOps, full library) / select_features_from_catalog (by category,
            writes importance back) / discover_lookback
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core import schemas

COLLECTION = "feature_catalog"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id(fid):
    return f"feature:{fid}"


# --------------------------------------------------------------------------- #
# read
# --------------------------------------------------------------------------- #
def get_feature(store, feature_id: str) -> Optional[dict]:
    return store.get(COLLECTION, _id(feature_id))


def find_features(
    store,
    category: Optional[str] = None,
    target_task: Optional[str] = None,
    target_model: Optional[str] = None,
    kind: str = "transform",
    status: str = "active",
) -> List[dict]:
    sel: Dict = {}
    if status:
        sel["status"] = status
    if category:
        sel["scenario_categories"] = {"$elemMatch": category}
    if target_task:
        sel["target_task"] = target_task
    if target_model:
        sel["target_model"] = target_model
    docs = store.find(COLLECTION, sel)
    if kind:
        docs = [d for d in docs if d.get("kind", "transform") == kind]
    return docs


def list_extractors(
    store, category: Optional[str] = None, status: str = "active"
) -> List[dict]:
    return find_features(store, category=category, kind="extractor", status=status)


def search(
    store, text: str = "", *, tags: Optional[List[str]] = None, status: str = "active"
):
    text = (text or "").lower()
    out = []
    for f in store.find(COLLECTION, {"status": status} if status else {}):
        hay = " ".join(
            [
                f.get("feature_id") or "",
                f.get("name") or "",
                " ".join(f.get("tags") or []),
            ]
        ).lower()
        if text and text not in hay:
            continue
        if tags and not set(tags) <= set(f.get("tags", [])):
            continue
        out.append(f)
    return out


def get_lineage(store, feature_id: str) -> dict:
    """Evolution chain: ancestors via parent_feature_id + descendants."""
    ancestors, cur, seen = [], get_feature(store, feature_id), set()
    while cur and cur.get("parent_feature_id") and cur["parent_feature_id"] not in seen:
        seen.add(cur["parent_feature_id"])
        p = get_feature(store, cur["parent_feature_id"])
        if not p:
            break
        ancestors.append(p["feature_id"])
        cur = p
    descendants = [
        f["feature_id"]
        for f in store.find(COLLECTION, {"parent_feature_id": feature_id})
    ]
    return {
        "feature_id": feature_id,
        "ancestors": ancestors,
        "root": ancestors[-1] if ancestors else feature_id,
        "descendants": descendants,
    }


# --------------------------------------------------------------------------- #
# write (transforms)
# --------------------------------------------------------------------------- #
def register_feature(store, feature: dict, *, overwrite: bool = False) -> dict:
    """Schema-validate, then run the EFE validity gate (entry points / no-inplace /
    invertibility) before accepting."""
    doc = schemas.validate_feature(feature)
    from ..engine import feature_runner as fr
    import numpy as np

    X = np.random.RandomState(0).normal(0, 1, size=(40, 3))
    chk = fr.validate_and_run(
        doc, X_fit=X[:30], X_in=X, metadata={"window": 8, "channel_indices": [0, 1, 2]}
    )
    doc.setdefault("validity", {}).update(chk["checks"])
    doc["kind"] = "transform"
    if store.get(COLLECTION, doc["_id"]) and not overwrite:
        raise ValueError(
            f"feature '{doc['feature_id']}' exists (overwrite=True or new_version)"
        )
    doc.setdefault("created_at", _now())
    return store.put(COLLECTION, doc)


def update_feature(store, feature_id: str, fields: dict) -> dict:
    doc = get_feature(store, feature_id)
    if not doc:
        raise ValueError(f"no feature {feature_id}")
    if "metrics" in fields:
        doc["metrics"] = doc.get("metrics", []) + list(fields.pop("metrics"))
    doc.update(fields)
    doc["updated_at"] = _now()
    return store.put(COLLECTION, doc)


def deprecate_feature(store, feature_id: str, reason: Optional[str] = None) -> dict:
    return update_feature(
        store, feature_id, {"status": "deprecated", "deprecation_reason": reason}
    )


def new_version(
    store, feature_id: str, fields: dict, *, new_feature_id: Optional[str] = None
) -> dict:
    old = get_feature(store, feature_id)
    if not old:
        raise ValueError(f"no feature {feature_id}")
    nv = dict(old, **fields)
    nv["version"] = str(int(str(old.get("version", "1")).split(".")[0]) + 1)
    nv["feature_id"] = new_feature_id or f"{feature_id}_v{nv['version']}"
    nv["parent_feature_id"] = feature_id
    nv["generation"] = int(old.get("generation", 0)) + 1
    nv.pop("_id", None)
    nv.pop("updated_at", None)
    out = register_feature(store, nv, overwrite=True)
    update_feature(
        store, feature_id, {"status": "superseded", "superseded_by": out["feature_id"]}
    )
    return out


# --------------------------------------------------------------------------- #
# write (extractors — the FLOps library index)
# --------------------------------------------------------------------------- #
_EXTRACTOR_CATEGORIES = {
    "mean": ["Future State Prediction", "Analysis & Inference"],
    "std": ["Future State Prediction", "Anomaly & Exception Detection"],
    "min": ["Analysis & Inference"],
    "max": ["Analysis & Inference"],
    "range": ["Analysis & Inference"],
    "q25": ["Future State Prediction"],
    "q75": ["Future State Prediction"],
    "kurtosis": ["Anomaly & Exception Detection"],
    "skew": ["Anomaly & Exception Detection"],
    "slope": ["Analysis & Inference", "Future State Prediction"],
    "autocorr1": ["Future State Prediction"],
    "energy": ["Anomaly & Exception Detection"],
    "abs_diff_mean": ["Anomaly & Exception Detection"],
    "spectral_centroid": ["Anomaly & Exception Detection"],
    "dominant_freq_power": ["Anomaly & Exception Detection"],
}


def register_extractor(store, name: str, scenario_categories: List[str]) -> dict:
    from ..reasoning import feature_selection as fsel

    doc = {
        "_id": _id(name),
        "feature_id": name,
        "name": f"FLOps extractor: {name}",
        "description": fsel.describe(name),
        "kind": "extractor",
        "extractor_name": name,
        "modality": "timeseries",
        "interface": "extract",
        "output_type": "scalar",
        "provenance": "library",
        "method": "FLOps",
        "scenario_categories": scenario_categories,
        "metrics": [],
        "tags": ["flops", "extractor", name],
        "status": "active",
        "version": "1",
        "created_by": "seed",
        "created_at": _now(),
    }
    return store.put(COLLECTION, doc)


def register_extractor_library(store) -> int:
    from ..reasoning import feature_selection as fsel

    n = 0
    for name in fsel.EXTRACTORS:
        register_extractor(
            store, name, _EXTRACTOR_CATEGORIES.get(name, ["Analysis & Inference"])
        )
        n += 1
    return n


# --------------------------------------------------------------------------- #
# learn (FLOps selection)
# --------------------------------------------------------------------------- #
def select_features(
    series, *, reference_feature: str = "mean", lookback=None, cd_margin=0.05
):
    """FLOps over the full library (no store needed) — backward-compatible."""
    from ..reasoning import feature_selection as fsel

    return fsel.select_features(
        series,
        reference_feature=reference_feature,
        lookback=lookback,
        cd_margin=cd_margin,
    )


def select_features_from_catalog(
    store,
    series,
    *,
    category: Optional[str] = None,
    reference_feature: str = "mean",
    cd_margin: float = 0.05,
    write_back: bool = False,
) -> dict:
    """FLOps restricted to the catalog's extractors for `category`; optionally write the
    importance score back onto each extractor doc's metrics (the catalog 'learns')."""
    from ..reasoning import feature_selection as fsel

    cands = list_extractors(store, category=category)
    names = [
        c["extractor_name"] for c in cands if c.get("extractor_name") in fsel.EXTRACTORS
    ]
    subset = {n: fsel.EXTRACTORS[n] for n in names} or fsel.EXTRACTORS
    res = fsel.select_features(
        series,
        reference_feature=reference_feature,
        cd_margin=cd_margin,
        extractors=subset,
    )
    res["candidates"] = names
    if write_back:
        for name, sc in res["scores"].items():
            if store.get(COLLECTION, _id(name)):
                update_feature(
                    store,
                    name,
                    {
                        "metrics": [
                            {
                                "metric": "flops_importance",
                                "value": sc,
                                "dataset": "last_run",
                            }
                        ]
                    },
                )
    return res


def discover_lookback(series, max_lw: int = 128) -> int:
    from ..reasoning import feature_selection as fsel

    return fsel.discover_lookback(series, max_lw=max_lw)
