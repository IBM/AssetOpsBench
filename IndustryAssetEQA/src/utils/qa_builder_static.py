# src/utils/qa_builder_static.py
"""
QA Builder for static diagnostic datasets (e.g., USM/USM-like).

Takes facts stored in EpisodicStore (from static_fact_extractor),
and produces diagnostic QA instances of the form:

{
  "qa_id": "usm_c_c_17",
  "fact_id": "c_17",
  "task_type": "diagnostic",
  "question": "Why is this diagnostic episode labeled 'Waxing' for asset 'c'?",
  "direct_answer": "Because features path1_amp_ratio and narrowband_energy are abnormal in a way characteristic of the 'Waxing' state.",
  "reasoning_answer": "In this episode for asset 'c', path1_amp_ratio = 4.21 and narrowband_energy = 2.90, which are significantly different from typical values in the dataset; this combination matches patterns labeled 'Waxing'.",
  "provenance": {
    "fact_id": "c_17",
    "features": ["path1_amp_ratio", "narrowband_energy"],
    "file": "c.csv",
    "row": 17
  },
  "confidence": 0.82
}

Usage (example):

python -m utils.qa_builder_static \
  --db data/episodic_store.db \
  --out data/c_qa.jsonl \
  --per-label 20

"""

from __future__ import annotations
import json
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from src.utils.episodic_store import EpisodicStore


def _load_all_facts(store: EpisodicStore) -> List[Dict[str, Any]]:
    """Load all facts from the store into memory."""
    cur = store.conn.cursor()
    rows = cur.execute("SELECT fact_json FROM facts").fetchall()
    return [json.loads(r[0]) for r in rows]


def _get_distinct_labels(store: EpisodicStore) -> List[str]:
    cur = store.conn.cursor()
    rows = cur.execute(
        "SELECT DISTINCT label FROM facts WHERE label IS NOT NULL"
    ).fetchall()
    return [r[0] for r in rows if r[0] is not None]


def _compute_feature_stats(facts: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean/std per feature name across all facts for numeric values.
    Returns: {feature_name: {"mean": m, "std": s}}
    """
    values: Dict[str, List[float]] = {}
    for fact in facts:
        for feat in fact.get("features", []):
            name = feat.get("name")
            val = feat.get("value")
            if isinstance(val, (int, float)) and not np.isnan(val):
                values.setdefault(name, []).append(float(val))

    stats: Dict[str, Dict[str, float]] = {}
    for name, vals in values.items():
        if len(vals) == 0:
            continue
        arr = np.array(vals, dtype=float)
        m = float(arr.mean())
        s = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        stats[name] = {"mean": m, "std": s}
    return stats


def _pick_salient_features(
    fact: Dict[str, Any],
    feature_stats: Dict[str, Dict[str, float]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Pick top_k salient features based on |z-score| vs dataset statistics.
    Fall back to the first top_k numeric features if stats are missing.
    Returns list of feature dicts (subset of fact["features"]).
    """
    feats = fact.get("features", [])
    scored: List[Tuple[float, Dict[str, Any]]] = []
    fallback_numeric: List[Dict[str, Any]] = []

    for feat in feats:
        name = feat.get("name")
        val = feat.get("value")
        if not isinstance(val, (int, float)) or (isinstance(val, float) and np.isnan(val)):
            continue
        fallback_numeric.append(feat)
        stats = feature_stats.get(name)
        if not stats:
            continue
        mean = stats["mean"]
        std = stats["std"]
        if std <= 0.0:
            z = 0.0
        else:
            z = (float(val) - mean) / std
        scored.append((abs(z), feat))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    # fallback: just take first top_k numeric features
    return fallback_numeric[:top_k]


def _build_qa_for_fact(
    fact: Dict[str, Any],
    feature_stats: Dict[str, Dict[str, float]],
    dataset_name: str = "usm_static",
) -> Dict[str, Any]:
    """
    Build a single diagnostic QA instance for one fact.
    """
    fact_id = fact.get("fact_id")
    asset_id = fact.get("asset_id") or "asset"
    label = fact.get("label") or "Unknown"
    source_file = fact.get("source_file", "unknown")
    row_index = fact.get("row_index", None)

    salient = _pick_salient_features(fact, feature_stats, top_k=3)
    salient_names = [f["name"] for f in salient]
    # format value snippets for reasoning
    val_snippets: List[str] = []
    for f in salient:
        name = f["name"]
        val = f["value"]
        if isinstance(val, float):
            val_snippets.append(f"{name} = {val:.3f}")
        else:
            val_snippets.append(f"{name} = {val}")

    features_text = ", ".join(salient_names) if salient_names else "a subset of diagnostic features"
    values_text = ", ".join(val_snippets) if val_snippets else "several diagnostic features whose values differ from typical baselines"

    question = f"Why is this diagnostic episode labeled '{label}' for asset '{asset_id}'?"

    direct_answer = (
        f"Because the diagnostic features {features_text} are abnormal in a way "
        f"characteristic of the '{label}' state."
    )

    reasoning_answer = (
        f"In this episode for asset '{asset_id}', {values_text}. "
        f"Compared to typical episodes in this dataset, this pattern is indicative of the '{label}' state, "
        f"which is why this diagnostic snapshot is labeled as '{label}'."
    )

    provenance = {
        "fact_id": fact_id,
        "features": salient_names,
        "file": source_file,
        "row": row_index,
    }

    # simple confidence heuristic: more salient features → higher confidence
    base_conf = 0.70
    conf = min(0.95, base_conf + 0.05 * len(salient))

    qa_id = f"{dataset_name}_{fact_id}"

    qa = {
        "qa_id": qa_id,
        "fact_id": fact_id,
        "task_type": "diagnostic",
        "question": question,
        "direct_answer": direct_answer,
        "reasoning_answer": reasoning_answer,
        "provenance": provenance,
        "confidence": round(conf, 2),
        "label": label,
        "asset_id": asset_id,
    }
    return qa


def build_qa_dataset(
    db_path: str,
    out_path: str,
    per_label: int = 20,
    dataset_name: str = "usm_static",
    labels: Optional[List[str]] = None,
) -> int:
    """
    Build a diagnostic QA dataset from all facts in an EpisodicStore.

    - db_path: path to SQLite episodic_store.db
    - out_path: JSONL file to write QA instances
    - per_label: maximum number of QA instances per label
    - dataset_name: used to prefix qa_id
    - labels: optional list of labels to include (default = all labels in store)

    Returns: number of QA instances written.
    """
    store = EpisodicStore(db_path=db_path)
    try:
        all_facts = _load_all_facts(store)
        if not all_facts:
            raise RuntimeError("No facts found in episodic store.")

        feature_stats = _compute_feature_stats(all_facts)
        all_labels = labels or _get_distinct_labels(store)
        if not all_labels:
            raise RuntimeError("No labels found in episodic store.")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        count = 0
        with open(out_path, "w") as fout:
            for label in all_labels:
                facts_for_label = store.query_by_label(label, limit=per_label)
                for fact in facts_for_label:
                    qa = _build_qa_for_fact(fact, feature_stats, dataset_name=dataset_name)
                    fout.write(json.dumps(qa) + "\n")
                    count += 1
        return count
    finally:
        store.close()


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build diagnostic QA dataset from EpisodicStore.")
    parser.add_argument("--db", required=True, help="Path to episodic_store.db")
    parser.add_argument("--out", required=True, help="Output JSONL path for QA dataset")
    parser.add_argument("--per-label", type=int, default=20, help="Max QA instances per label")
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated list of labels to include (optional)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="usm_static",
        help="Dataset name prefix to use for qa_id",
    )

    args = parser.parse_args()
    labels_list = args.labels.split(",") if args.labels else None
    n = build_qa_dataset(
        db_path=args.db,
        out_path=args.out,
        per_label=args.per_label,
        dataset_name=args.dataset_name,
        labels=labels_list,
    )
    print(f"Wrote {n} QA instances to {args.out}")
