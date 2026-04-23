# src/utils/qa_builder_ts_cf.py

"""
Build counterfactual QA instances for PdM time-series episodes.

For each failure_window fact (label != "healthy"), we:
  - Define an intervention: reset hours_since_last_maint_<label> = 0.
  - Use the PdM risk model to estimate risk_before, risk_after, delta_risk, direction.
  - Create a QA asking how risk would change under this intervention.

Output JSONL schema (one per line):

{
  "qa_id": "<dataset_name>_cf_<fact_id>",
  "fact_id": "<fact_id>",
  "task_type": "counterfactual",
  "question": "...",
  "direct_answer": "...",
  "reasoning_answer": "...",
  "provenance": {
    "fact_id": "<fact_id>",
    "features": ["hours_since_last_maint_comp3"],
    "file": "PdM_telemetry.csv",
    "row": 0
  },
  "counterfactual": {
    "intervention": "do(hours_since_last_maint_comp3 = 0.0)",
    "risk_before": 1.0,
    "risk_after": 0.000009,
    "delta_risk": -0.999991,
    "direction": "decrease"
  },
  "confidence": 0.9,
  "label": "comp3",
  "asset_id": "machine_56"
}
"""

from __future__ import annotations
import json
from typing import Dict, Any, List, Tuple

from src.utils.causal_sim_pdm import (
    load_facts_jsonl,
    load_risk_model,
    estimate_effect,
)


def find_feat_value(fact: Dict[str, Any], feat_name: str) -> float:
    for feat in fact.get("features", []):
        if feat.get("name") == feat_name:
            return float(feat.get("value", 0.0))
    return 0.0


def build_counterfactual_qa_dataset(
    facts_path: str,
    model_path: str,
    out_path: str,
    dataset_name: str = "pdm_ts",
    per_label: int | None = None,
    min_abs_delta: float = 1e-3,
) -> int:
    """
    Build counterfactual QAs for PdM.

    Args:
      facts_path: path to pdm_facts.jsonl
      model_path: path to trained PdM risk model (pdm_risk_model.joblib)
      out_path:   output JSONL path for counterfactual QAs
      dataset_name: used as prefix in qa_id
      per_label: optional cap on #QAs per label (comp1/2/3/4)
      min_abs_delta: minimum |delta_risk| to keep a QA (filter trivial changes)

    Returns:
      Number of QAs written.
    """
    facts = load_facts_jsonl(facts_path)
    bundle = load_risk_model(model_path)

    counts_per_label: Dict[str, int] = {}
    written = 0

    with open(out_path, "w") as fout:
        for fact in facts:
            label = fact.get("label")
            if not label or label == "healthy":
                continue  # only failures for now

            # optional cap per label
            if per_label is not None:
                c = counts_per_label.get(label, 0)
                if c >= per_label:
                    continue

            fact_id = fact.get("fact_id")
            asset_id = fact.get("asset_id", "unknown_asset")
            start_time = fact.get("start_time")
            end_time = fact.get("end_time")

            # maintenance feature tied to this label, e.g. hours_since_last_maint_comp3
            maint_feat = f"hours_since_last_maint_{label}"
            feat_names = [f.get("name") for f in fact.get("features", [])]
            if maint_feat not in feat_names:
                # skip if this episode doesn't carry that feature
                continue

            current_val = find_feat_value(fact, maint_feat)

            # counterfactual: reset maintenance age for the failed component
            intervention = {maint_feat: 0.0}
            cf_result = estimate_effect(
                fact=fact,
                model_bundle=bundle,
                intervention=intervention,
            )

            # filter out trivial/no-change cases
            if abs(cf_result["delta_risk"]) < min_abs_delta:
                continue

            direction = cf_result["direction"]  # 'increase'/'decrease'/'no_change'
            risk_before = cf_result["risk_before"]
            risk_after = cf_result["risk_after"]
            delta_risk = cf_result["delta_risk"]
            cf_conf = cf_result["confidence"]

            if direction == "decrease":
                direct_answer = "The risk of failure would decrease."
                dir_verb = "decrease"
            elif direction == "increase":
                direct_answer = "The risk of failure would increase."
                dir_verb = "increase"
            else:
                direct_answer = "The risk of failure would remain approximately the same."
                dir_verb = "stay roughly the same"

            reasoning_answer = (
                f"For this episode, the learned risk model currently predicts a failure risk of "
                f"{risk_before:.3f}. If we reset {maint_feat} from {current_val:.1f} hours to 0.0 "
                f"(simulating maintenance right before the window), the predicted risk changes to "
                f"{risk_after:.3f} (Δ = {delta_risk:.3f}), so the overall failure risk is expected "
                f"to {dir_verb}."
            )

            question = (
                f"If maintenance on {label} had been performed immediately before the telemetry window "
                f"from {start_time} to {end_time} for asset '{asset_id}', how would the risk of failure "
                f"in this episode change?"
            )

            provenance = {
                "fact_id": fact_id,
                "features": [maint_feat],
                "file": fact.get("source_file", "unknown"),
                "row": fact.get("row_index", -1),
            }

            cf_block = {
                "intervention": f"do({maint_feat} = 0.0)",
                "risk_before": risk_before,
                "risk_after": risk_after,
                "delta_risk": delta_risk,
                "direction": direction,
            }

            qa_id = f"{dataset_name}_cf_{fact_id}"

            qa_obj = {
                "qa_id": qa_id,
                "fact_id": fact_id,
                "task_type": "counterfactual",
                "question": question,
                "direct_answer": direct_answer,
                "reasoning_answer": reasoning_answer,
                "provenance": provenance,
                "counterfactual": cf_block,
                "confidence": cf_conf,
                "label": label,
                "asset_id": asset_id,
            }

            fout.write(json.dumps(qa_obj) + "\n")
            counts_per_label[label] = counts_per_label.get(label, 0) + 1
            written += 1

    return written


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Build counterfactual QA dataset for PdM.")
    parser.add_argument("--facts", required=True, help="Path to pdm_facts.jsonl")
    parser.add_argument("--model", required=True, help="Path to PdM risk model (pdm_risk_model.joblib)")
    parser.add_argument("--out", required=True, help="Output path for counterfactual QA JSONL")
    parser.add_argument("--dataset-name", default="pdm_ts", help="Prefix for qa_id")
    parser.add_argument("--per-label", type=int, default=None, help="Optional cap per failure label")
    parser.add_argument("--min-abs-delta", type=float, default=1e-3, help="Min |delta_risk| to keep a QA")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = build_counterfactual_qa_dataset(
        facts_path=args.facts,
        model_path=args.model,
        out_path=args.out,
        dataset_name=args.dataset_name,
        per_label=args.per_label,
        min_abs_delta=args.min_abs_delta,
    )
    print(f"Wrote {n} counterfactual QA instances to {args.out}")
