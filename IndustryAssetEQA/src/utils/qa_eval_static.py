# src/utils/qa_eval_static.py
"""
Evaluation utilities for static and time-series diagnostic / counterfactual QA.

Assumes:
  - Gold QA instances in a JSONL file (qa_path), as produced by a QA builder
    (e.g. qa_builder_static.build_qa_dataset or qa_builder_ts_cf.build_counterfactual_qa_dataset).
  - Model predictions in a JSONL file (preds_path) where each line has:
        {
          "qa_id": "<qa id>",
          "answer": {
              "direct_answer": "...",
              "reasoning_answer": "...",
              "provenance": {...},
              "confidence": 0.81,
              # OPTIONAL for counterfactual QAs:
              # "counterfactual": {
              #   "direction": "increase" | "decrease" | "no_change",
              #   ...
              # }
          }
        }

Metrics computed:
  - total: total # of QA instances with predictions
  - structure_ok_rate: fraction of answers with valid JSON structure (via verify_answer)
  - provenance_ok_rate: fraction with valid provenance (fact exists, features valid, file/row match)
  - label_consistency_rate: fraction where the gold label appears in the direct_answer text
  - full_pass_rate: fraction where structure_ok & provenance_ok & label_consistent are all true

  - counterfactual_total: number of QAs with task_type == "counterfactual" that were evaluated
  - counterfactual_direction_accuracy: among those, fraction where the model's
      counterfactual.direction matches the gold counterfactual.direction (case-insensitive)
"""

from __future__ import annotations
import json
import os
from typing import Dict, Any, List, Tuple

from src.utils.episodic_store import EpisodicStore
from src.utils.verifier_static import verify_answer


def load_qa_index(qa_path: str) -> Dict[str, Dict[str, Any]]:
    """Load QA JSONL into a dict keyed by qa_id."""
    idx: Dict[str, Dict[str, Any]] = {}
    with open(qa_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("qa_id")
            if qid is None:
                continue
            idx[qid] = obj
    return idx


def load_preds(preds_path: str) -> Dict[str, Dict[str, Any]]:
    """Load predictions JSONL into a dict keyed by qa_id."""
    preds: Dict[str, Dict[str, Any]] = {}
    with open(preds_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("qa_id")
            ans = obj.get("answer")
            if qid is None or ans is None:
                continue
            preds[qid] = ans
    return preds


def check_label_consistency(
    gold_qa: Dict[str, Any],
    answer: Dict[str, Any]
) -> bool:
    """
    Simple heuristic: the gold label should appear (case-insensitive substring)
    in the model's direct_answer.
    """
    label = gold_qa.get("label")
    if not label:
        return True  # can't check, so don't penalize
    direct_answer = answer.get("direct_answer", "")
    if not isinstance(direct_answer, str):
        return False
    return label.lower() in direct_answer.lower()


def check_counterfactual_direction(
    gold_qa: Dict[str, Any],
    answer: Dict[str, Any],
) -> Tuple[bool, bool]:
    """
    Compare direction of risk change for counterfactual QAs.

    Returns:
      (is_correct, counted)

      - is_correct: True if directions match exactly (case-insensitive).
      - counted:    True if we had both gold and predicted directions and
                    included this example in the metric; False if missing.

    Behavior:
      - If gold has no counterfactual.direction -> (False, False)  (not counted)
      - If gold has direction but prediction has no direction -> (False, True) (counted as incorrect)
      - Otherwise: (gold_dir == pred_dir, True)
    """
    gold_cf = gold_qa.get("counterfactual") or {}
    pred_cf = answer.get("counterfactual") or {}

    gold_dir = (gold_cf.get("direction") or "").strip().lower()
    pred_dir = (pred_cf.get("direction") or "").strip().lower()

    if not gold_dir:
        # no gold direction -> can't evaluate this example
        return False, False
    if not pred_dir:
        # gold exists but prediction missing -> we count it as incorrect
        return False, True

    return (gold_dir == pred_dir), True


def evaluate_static_diagnostic(
    db_path: str,
    qa_path: str,
    preds_path: str,
    out_path: str = None
) -> Dict[str, Any]:
    """
    Evaluate model predictions on static / time-series diagnostic and counterfactual QA.

    Parameters:
      - db_path: episodic_store.db
      - qa_path: gold QA JSONL
      - preds_path: model predictions JSONL
      - out_path: optional path to save detailed per-example results (JSONL)

    Returns:
      dict with aggregate metrics.
    """
    store = EpisodicStore(db_path=db_path)
    try:
        qa_index = load_qa_index(qa_path)
        preds = load_preds(preds_path)

        total = 0
        struct_ok = 0
        prov_ok = 0
        label_ok = 0
        full_ok = 0

        # New counters for counterfactual direction metric
        cf_total = 0
        cf_dir_correct = 0

        detailed: List[Dict[str, Any]] = []

        for qa_id, gold_qa in qa_index.items():
            ans = preds.get(qa_id)
            if ans is None:
                # skip missing prediction
                continue

            total += 1
            v_report = verify_answer(ans, store)
            s_ok = v_report["structure_ok"]
            p_ok = v_report["provenance_ok"]
            l_ok = check_label_consistency(gold_qa, ans)

            if s_ok:
                struct_ok += 1
            if p_ok:
                prov_ok += 1
            if l_ok:
                label_ok += 1
            if s_ok and p_ok and l_ok:
                full_ok += 1

            task_type = gold_qa.get("task_type")
            dir_ok = False  # default

            # Counterfactual direction metric
            if task_type == "counterfactual":
                dir_ok, counted = check_counterfactual_direction(gold_qa, ans)
                if counted:
                    cf_total += 1
                    if dir_ok:
                        cf_dir_correct += 1

            detailed.append({
                "qa_id": qa_id,
                "label": gold_qa.get("label"),
                "task_type": task_type,
                "structure_ok": s_ok,
                "provenance_ok": p_ok,
                "label_consistent": l_ok,
                "counterfactual_dir_correct": bool(task_type == "counterfactual" and dir_ok),
                "verify_report": v_report
            })

        metrics = {
            "total": total,
            "structure_ok_rate": struct_ok / total if total else 0.0,
            "provenance_ok_rate": prov_ok / total if total else 0.0,
            "label_consistency_rate": label_ok / total if total else 0.0,
            "full_pass_rate": full_ok / total if total else 0.0,
            "counterfactual_total": cf_total,
            "counterfactual_direction_accuracy": (
                cf_dir_correct / cf_total if cf_total else 0.0
            ),
        }

        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                for row in detailed:
                    f.write(json.dumps(row) + "\n")

        return metrics
    finally:
        store.close()


# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate static / TS QA predictions (including counterfactuals).")
    parser.add_argument("--db", required=True, help="Path to episodic_store.db")
    parser.add_argument("--qa", required=True, help="Path to gold QA JSONL (e.g. data/c_qa.jsonl or data/pdm_qa_cf.jsonl)")
    parser.add_argument("--preds", required=True, help="Path to predictions JSONL (e.g. data/c_preds.jsonl or data/pdm_preds_cf.jsonl)")
    parser.add_argument("--out", default=None, help="Optional path for per-example evaluation JSONL")
    args = parser.parse_args()

    metrics = evaluate_static_diagnostic(
        db_path=args.db,
        qa_path=args.qa,
        preds_path=args.preds,
        out_path=args.out
    )
    print(json.dumps(metrics, indent=2))
