"""Result tables: one collection per task type, derived from the task catalog (tasks.py)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from ..core import tasks as task_spec

# Short id prefixes only (aux). The collection NAME is the single source of truth in tasks.py
# (TSTask.result_collection); this map exists purely to keep result_ids short and stable.
_PREFIX = {
    "tsfm_forecasting": "fr",
    "tsfm_regression": "rr",
    "tsfm_classification": "cr",
    "tsfm_anomaly_detection": "ar",
    "tsfm_imputation": "ir",
    "tsfm_evaluation": "er",
    "tsfm_similarity_search": "sr",
    "tsfm_clustering": "clr",
}

# Summary keys a written result must carry, per task (tasks not listed require nothing extra).
REQUIRED_SUMMARY = {
    "tsfm_forecasting": ["horizon"],
    "tsfm_anomaly_detection": ["total_records", "anomaly_count"],
    "tsfm_classification": ["num_classes"],
    "tsfm_imputation": ["imputed_count"],
    "tsfm_evaluation": [],
}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collection_for(task_type: str) -> str:
    """The result collection for a task, read from the task catalog (single source of truth)."""
    t = task_spec.TASKS.get(task_type)
    if t is None:
        raise ValueError(
            f"unknown task_type '{task_type}'. Known: {list(task_spec.TASKS)}"
        )
    return t.result_collection


def write_result(
    store,
    task_type: str,
    *,
    asset_id: str,
    results_file: str,
    model_id: Optional[str] = None,
    feature_ids: Optional[List[str]] = None,
    dataset: Optional[str] = None,
    scenario_id: Optional[str] = None,
    summary: Optional[dict] = None,
    metrics: Optional[list] = None,
    created_by: str = "agent.tsfm",
) -> dict:
    coll = collection_for(task_type)  # raises on unknown task_type
    prefix = _PREFIX.get(task_type, task_type[:2])
    summary = summary or {}
    missing = [k for k in REQUIRED_SUMMARY.get(task_type, []) if k not in summary]
    if task_type == "tsfm_evaluation" and not metrics:
        missing = ["metrics"]
    if missing:
        raise ValueError(f"{task_type}: missing required {missing}")
    rid = f"{prefix}:{uuid.uuid4().hex[:12]}"
    doc = {
        "_id": rid,
        "result_id": rid,
        "task_type": task_type,
        "asset_id": asset_id,
        "model_id": model_id,
        "feature_ids": feature_ids or [],
        "dataset": dataset,
        "scenario_id": scenario_id,
        "results_file": results_file,
        "summary": summary,
        "metrics": metrics or [],
        "status": "produced",
        "created_by": created_by,
        "produced_at": _now(),
    }
    return store.put(coll, doc)


def get_result(store, task_type, result_id):
    return store.get(collection_for(task_type), result_id)


def list_results(store, task_type, asset_id=None, scenario_id=None):
    sel = {}
    if asset_id:
        sel["asset_id"] = asset_id
    if scenario_id:
        sel["scenario_id"] = scenario_id
    return store.find(collection_for(task_type), sel)
