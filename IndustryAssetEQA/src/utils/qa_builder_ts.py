# src/utils/qa_builder_ts.py
"""
QA builder for time-series PdM facts.

Given:
  - pdm_facts.jsonl (from ts_fact_extractor.py)
  - (optionally) a trained risk model and feature order for counterfactual/action QAs

This script can build several QA sets:

  task_type = "diagnostic"
    -> Why is this episode labeled 'comp3'?

  task_type = "counterfactual"
    -> If we had reset hours_since_last_maint_comp3 to 0 at the start of this window,
       how would the risk of any failure change?

  task_type = "descriptive"
    -> During this window, what was the average vibration level?

  task_type = "temporal_count"
    -> During this window, how many distinct error types occurred?

  task_type = "action_recommendation"
    -> For this episode, should we open a work order now or continue monitoring?

Outputs:
  - A JSONL QA file where each line is:
      {
        "qa_id": "...",
        "fact_id": "...",
        "task_type": "...",
        "question": "...",
        "direct_answer": "...",
        "reasoning_answer": "...",
        "provenance": {...},
        "label": "... or structured label ...",
        "asset_id": "...",
        ... (optionally "counterfactual": {...} for CF QAs)
      }

Use:
  python -m src.utils.qa_builder_ts \
    --facts data/pdm_facts.jsonl \
    --out data/pdm_qa_diag.jsonl \
    --mode diagnostic

  python -m src.utils.qa_builder_ts \
    --facts data/pdm_facts.jsonl \
    --out data/pdm_qa_cf.jsonl \
    --mode counterfactual \
    --model data/pdm_risk_model.joblib \
    --feature-order-json data/pdm_feature_order.json

  python -m src.utils.qa_builder_ts \
    --facts data/pdm_facts.jsonl \
    --out data/pdm_qa_desc.jsonl \
    --mode descriptive

  python -m src.utils.qa_builder_ts \
    --facts data/pdm_facts.jsonl \
    --out data/pdm_qa_temporal.jsonl \
    --mode temporal_count

  python -m src.utils.qa_builder_ts \
    --facts data/pdm_facts.jsonl \
    --out data/pdm_qa_action.jsonl \
    --mode action_recommendation \
    --model data/pdm_risk_model.joblib \
    --feature-order-json data/pdm_feature_order.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

import numpy as np
from joblib import load as joblib_load


# --------------------
# Helpers
# --------------------

def get_feat(fact: Dict[str, Any], name: str, default: Optional[float] = None) -> Optional[float]:
    """Get a feature value by name from fact['features']."""
    for feat in fact.get("features", []):
        if feat.get("name") == name:
            return feat.get("value")
    return default


def iter_facts(facts_path: str):
    """Yield fact dicts from a JSONL file."""
    with open(facts_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def load_feature_order(path: str) -> List[str]:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "features" in obj:
        return list(obj["features"])
    if isinstance(obj, list):
        return list(obj)
    raise ValueError(f"Unsupported feature_order JSON format in {path}")


def load_risk_model(model_path: str):
    return joblib_load(model_path)


def compute_any_failure_risk(
    model,
    classes: List[str],
    feature_order: List[str],
    fact: Dict[str, Any],
) -> float:
    """
    Compute P(any failure) = 1 - P(healthy) for a single fact.
    """
    x = [get_feat(fact, name, 0.0) for name in feature_order]
    X = np.asarray(x, dtype=float).reshape(1, -1)
    probs = model.predict_proba(X)[0]  # shape (num_classes,)
    classes_arr = np.asarray(classes)
    # assume "healthy" exists in classes
    if "healthy" not in classes_arr:
        # fallback: treat everything as failure -> risk=1
        return 1.0
    idx_h = int(np.where(classes_arr == "healthy")[0][0])
    p_healthy = float(probs[idx_h])
    return 1.0 - p_healthy


def apply_intervention_to_fact(
    fact: Dict[str, Any],
    intervention: Dict[str, float],
) -> Dict[str, Any]:
    """
    Returns a shallow copy of fact with feature values modified
    according to intervention dict: {feature_name: new_value}.
    """
    new_fact = dict(fact)
    new_feats = []
    for feat in fact.get("features", []):
        name = feat.get("name")
        val = feat.get("value")
        if name in intervention:
            new_feats.append({"name": name, "value": intervention[name]})
        else:
            new_feats.append({"name": name, "value": val})
    new_fact["features"] = new_feats
    return new_fact


# --------------------
# QA builders
# --------------------

def build_diagnostic_qa_for_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnostic QA: explain the failure label for this episode.
    """
    fact_id = fact["fact_id"]
    asset_id = fact.get("asset_id", fact.get("machineID"))
    label = str(fact.get("label", "unknown"))
    source_file = fact.get("source_file", "PdM_telemetry.csv")
    row_index = fact.get("row_index", 0)
    start = fact.get("start_time")
    end = fact.get("end_time")

    qa_id = f"pdm_diag_{fact_id}"

    question = (
        f"Why is this diagnostic episode for asset {asset_id} labeled '{label}' "
        f"over the time window from {start} to {end}?"
    )

    # Pick a few illustrative features
    important_feats = [
        "vibration_mean",
        "pressure_mean",
        "hours_since_last_maint_comp1",
        "hours_since_last_maint_comp2",
        "hours_since_last_maint_comp3",
        "hours_since_last_maint_comp4",
    ]
    used_feats = []
    for name in important_feats:
        val = get_feat(fact, name)
        if val is not None:
            used_feats.append((name, val))

    if not used_feats:
        # fallback: use first 3 features
        feats_all = fact.get("features", [])
        used_feats = [(f["name"], f["value"]) for f in feats_all[:3]]

    feat_str = ", ".join(f"{n}={v:.3f}" for n, v in used_feats)

    direct_answer = (
        f"This episode is labeled '{label}' because the diagnostic features show a pattern "
        f"consistent with the '{label}' failure mode."
    )

    reasoning_answer = (
        f"For asset {asset_id} in the window {start} to {end}, key features are: {feat_str}. "
        f"Compared to typical healthy episodes, this combination is characteristic of the "
        f"'{label}' state, so the episode is labeled '{label}'."
    )

    provenance = {
        "fact_id": fact_id,
        "features": [n for n, _ in used_feats],
        "file": source_file,
        "row": row_index,
    }

    qa = {
        "qa_id": qa_id,
        "fact_id": fact_id,
        "task_type": "diagnostic",
        "question": question,
        "direct_answer": direct_answer,
        "reasoning_answer": reasoning_answer,
        "provenance": provenance,
        "label": label,
        "asset_id": asset_id,
    }
    return qa


def build_descriptive_qa_for_fact(fact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Descriptive QA: ask about a numeric summary in the window.
    Example: average vibration level.
    """
    fact_id = fact["fact_id"]
    asset_id = fact.get("asset_id", fact.get("machineID"))
    source_file = fact.get("source_file", "PdM_telemetry.csv")
    row_index = fact.get("row_index", 0)
    start = fact.get("start_time")
    end = fact.get("end_time")

    vib_mean = get_feat(fact, "vibration_mean")
    if vib_mean is None:
        return None

    qa_id = f"pdm_desc_{fact_id}"

    question = (
        f"During the time window from {start} to {end} for asset {asset_id}, "
        f"what was the average vibration level?"
    )

    direct_answer = f"The average vibration level was approximately {vib_mean:.2f}."
    reasoning_answer = (
        f"In this episode for asset {asset_id}, the feature vibration_mean is "
        f"{vib_mean:.2f}, computed over the telemetry window from {start} to {end}. "
        "This value summarizes the typical vibration during that period."
    )

    provenance = {
        "fact_id": fact_id,
        "features": ["vibration_mean"],
        "file": source_file,
        "row": row_index,
    }

    qa = {
        "qa_id": qa_id,
        "fact_id": fact_id,
        "task_type": "descriptive",
        "question": question,
        "direct_answer": direct_answer,
        "reasoning_answer": reasoning_answer,
        "provenance": provenance,
        "label": f"{vib_mean:.4f}",
        "asset_id": asset_id,
    }
    return qa


def build_temporal_count_qa_for_fact(fact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Temporal/count QA: how many distinct error types in the window?
    Uses features error_count_last_window and distinct_error_types_last_window.
    """
    fact_id = fact["fact_id"]
    asset_id = fact.get("asset_id", fact.get("machineID"))
    start = fact.get("start_time")
    end = fact.get("end_time")
    row_index = fact.get("row_index", 0)

    n_errors = get_feat(fact, "error_count_last_window")
    n_distinct = get_feat(fact, "distinct_error_types_last_window")
    if n_errors is None or n_distinct is None:
        return None

    source_file = fact.get("errors_source_file", "PdM_errors.csv")

    qa_id = f"pdm_temp_{fact_id}"

    question = (
        f"Between {start} and {end} for asset {asset_id}, "
        "how many distinct error types occurred?"
    )

    direct_answer = f"There were {int(n_distinct)} distinct error types in this window."
    reasoning_answer = (
        f"In this episode, the feature distinct_error_types_last_window is "
        f"{int(n_distinct)}, meaning that {int(n_distinct)} different error codes were "
        f"observed between {start} and {end}. The overall error count in this window "
        f"is {int(n_errors)}, but only {int(n_distinct)} of them are unique types."
    )

    provenance = {
        "fact_id": fact_id,
        "features": ["error_count_last_window", "distinct_error_types_last_window"],
        "file": source_file,
        "row": row_index,
    }

    qa = {
        "qa_id": qa_id,
        "fact_id": fact_id,
        "task_type": "temporal_count",
        "question": question,
        "direct_answer": direct_answer,
        "reasoning_answer": reasoning_answer,
        "provenance": provenance,
        "label": str(int(n_distinct)),
        "asset_id": asset_id,
    }
    return qa


def build_counterfactual_qa_for_fact(
    fact: Dict[str, Any],
    model,
    classes: List[str],
    feature_order: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Counterfactual QA: what if we reset hours_since_last_maint_compX to 0?

    We:
      - choose the failing component (if any),
      - define intervention on hours_since_last_maint_compK,
      - compute risk_before and risk_after,
      - derive direction: increase / decrease / no_change,
      - encode this in gold 'counterfactual' label.
    """
    fact_id = fact["fact_id"]
    asset_id = fact.get("asset_id", fact.get("machineID"))
    label = str(fact.get("label", "unknown"))
    start = fact.get("start_time")
    end = fact.get("end_time")
    row_index = fact.get("row_index", 0)
    source_file = fact.get("source_file", "PdM_telemetry.csv")

    # choose component: if label is comp1/comp2/comp3/comp4, use it; else default comp3
    if label in {"comp1", "comp2", "comp3", "comp4"}:
        comp = label
    else:
        comp = "comp3"

    feat_name = f"hours_since_last_maint_{comp}"
    # Only build CF QA if this feature exists in this fact.
    if get_feat(fact, feat_name) is None:
        return None

    # baseline risk
    risk_before = compute_any_failure_risk(model, classes, feature_order, fact)

    # intervention: reset hours_since_last_maint_compX to 0
    intervention = {feat_name: 0.0}
    fact_cf = apply_intervention_to_fact(fact, intervention)
    risk_after = compute_any_failure_risk(model, classes, feature_order, fact_cf)

    delta = risk_after - risk_before
    if abs(delta) < 1e-6:
        direction = "no_change"
    elif delta < 0:
        direction = "decrease"
    else:
        direction = "increase"

    qa_id = f"pdm_cf_{fact_id}"

    question = (
        f"For asset {asset_id} in the time window {start} to {end}, suppose we had "
        f"performed maintenance on {comp} at the start of the window, effectively "
        f"resetting {feat_name} to 0.0. How would the risk of any component failure "
        "change compared to what actually happened?"
    )

    direct_answer = (
        f"The risk of failure would be expected to {direction} under this intervention."
    )

    reasoning_answer = (
        f"Using the learned risk model, the baseline probability of any failure in this "
        f"episode is approximately {risk_before:.3f}. After resetting {feat_name} to 0.0 "
        f"for the same features, the estimated failure risk becomes {risk_after:.3f}. "
        f"This corresponds to a {direction} in risk (delta = {risk_after - risk_before:.3f})."
    )

    provenance = {
        "fact_id": fact_id,
        "features": [feat_name] + feature_order,
        "file": source_file,
        "row": row_index,
    }

    cf_label = {
        "intervention": intervention,
        "direction": direction,
        "risk_before": risk_before,
        "risk_after": risk_after,
    }

    qa = {
        "qa_id": qa_id,
        "fact_id": fact_id,
        "task_type": "counterfactual",
        "question": question,
        "direct_answer": direct_answer,
        "reasoning_answer": reasoning_answer,
        "provenance": provenance,
        "counterfactual": cf_label,
        "label": direction,
        "asset_id": asset_id,
    }
    return qa


def build_action_qa_for_fact(
    fact: Dict[str, Any],
    model,
    classes: List[str],
    feature_order: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Actionable QA: should we open a work order now or continue monitoring?
    Uses risk model and a threshold.
    """
    fact_id = fact["fact_id"]
    asset_id = fact.get("asset_id", fact.get("machineID"))
    start = fact.get("start_time")
    end = fact.get("end_time")
    row_index = fact.get("row_index", 0)
    source_file = fact.get("source_file", "PdM_telemetry.csv")

    risk = compute_any_failure_risk(model, classes, feature_order, fact)
    action = "open_work_order" if risk >= threshold else "monitor"

    qa_id = f"pdm_action_{fact_id}"

    question = (
        f"For asset {asset_id} in the time window {start} to {end}, "
        "should a maintenance work order be opened now, or is it acceptable "
        "to continue monitoring?"
    )

    if action == "open_work_order":
        direct_answer = "A maintenance work order should be opened now."
    else:
        direct_answer = "It is acceptable to continue monitoring for now."

    reasoning_answer = (
        f"The learned risk model estimates the probability of any component failure "
        f"in this episode at approximately {risk:.2f}. With a decision threshold of "
        f"{threshold:.2f}, this places the episode in the '{action}' region, so the "
        f"recommended action is '{action}'."
    )

    provenance = {
        "fact_id": fact_id,
        "features": feature_order,
        "file": source_file,
        "row": row_index,
    }

    qa = {
        "qa_id": qa_id,
        "fact_id": fact_id,
        "task_type": "action_recommendation",
        "question": question,
        "direct_answer": direct_answer,
        "reasoning_answer": reasoning_answer,
        "provenance": provenance,
        "label": action,
        "asset_id": asset_id,
    }
    return qa


# --------------------
# Top-level builders
# --------------------

def build_pdm_diagnostic_qas(facts_path: str, out_path: str) -> None:
    with open(out_path, "w") as fout:
        for fact in iter_facts(facts_path):
            qa = build_diagnostic_qa_for_fact(fact)
            fout.write(json.dumps(qa) + "\n")


def build_pdm_descriptive_qas(facts_path: str, out_path: str) -> None:
    with open(out_path, "w") as fout:
        for fact in iter_facts(facts_path):
            qa = build_descriptive_qa_for_fact(fact)
            if qa is None:
                continue
            fout.write(json.dumps(qa) + "\n")


def build_pdm_temporal_qas(facts_path: str, out_path: str) -> None:
    with open(out_path, "w") as fout:
        for fact in iter_facts(facts_path):
            qa = build_temporal_count_qa_for_fact(fact)
            if qa is None:
                continue
            fout.write(json.dumps(qa) + "\n")


def build_pdm_counterfactual_qas(
    facts_path: str,
    model_path: str,
    feature_order_path: str,
    out_path: str,
) -> None:
    model = load_risk_model(model_path)
    feature_order = load_feature_order(feature_order_path)
    classes = list(model.classes_)

    with open(out_path, "w") as fout:
        for fact in iter_facts(facts_path):
            qa = build_counterfactual_qa_for_fact(fact, model, classes, feature_order)
            if qa is None:
                continue
            fout.write(json.dumps(qa) + "\n")


def build_pdm_action_qas(
    facts_path: str,
    model_path: str,
    feature_order_path: str,
    out_path: str,
    threshold: float = 0.5,
) -> None:
    model = load_risk_model(model_path)
    feature_order = load_feature_order(feature_order_path)
    classes = list(model.classes_)

    with open(out_path, "w") as fout:
        for fact in iter_facts(facts_path):
            qa = build_action_qa_for_fact(fact, model, classes, feature_order, threshold)
            fout.write(json.dumps(qa) + "\n")


# --------------------
# CLI
# --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build QA datasets from PdM facts.")
    parser.add_argument("--facts", required=True, help="Path to pdm_facts.jsonl")
    parser.add_argument("--out", required=True, help="Path to output QA JSONL")
    parser.add_argument(
        "--mode",
        choices=["diagnostic", "descriptive", "temporal_count", "counterfactual", "action_recommendation"],
        required=True,
        help="Type of QA to build.",
    )
    parser.add_argument("--model", default=None, help="Path to risk model (.joblib) for CF/action QAs.")
    parser.add_argument("--feature-order-json", default=None, help="Path to JSON file with feature order for model.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Risk threshold for action_recommendation.")
    args = parser.parse_args()

    if args.mode in {"counterfactual", "action_recommendation"}:
        if not args.model or not args.feature_order_json:
            raise SystemExit("counterfactual/action_recommendation require --model and --feature-order-json")

    if args.mode == "diagnostic":
        build_pdm_diagnostic_qas(args.facts, args.out)
    elif args.mode == "descriptive":
        build_pdm_descriptive_qas(args.facts, args.out)
    elif args.mode == "temporal_count":
        build_pdm_temporal_qas(args.facts, args.out)
    elif args.mode == "counterfactual":
        build_pdm_counterfactual_qas(args.facts, args.model, args.feature_order_json, args.out)
    elif args.mode == "action_recommendation":
        build_pdm_action_qas(args.facts, args.model, args.feature_order_json, args.out, args.threshold)
