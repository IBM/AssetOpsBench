# src/utils/causal_sim_pdm.py

"""
Causal & Counterfactual Simulator for PdM episode facts.

Goal (v0):
  - Fit a simple predictive model P(label | features) on pdm_facts.jsonl
    (labels like 'comp1', 'comp2', 'comp3', 'comp4', 'healthy').
  - Use this model to estimate how an intervention on features changes
    the risk of failure (any failure vs healthy).

This is NOT physics; it is a parametric, data-driven counterfactual:
  risk_before = 1 - P(label='healthy')
  risk_after  = 1 - P(label='healthy' | do(feature := new_value))

We treat "risk" as the probability of any failure class (1 - P(healthy)).

CLI:

1) Fit model:

python -m src.utils.causal_sim_pdm fit \
  --facts data/pdm_facts.jsonl \
  --model-out data/pdm_risk_model.joblib

2) Estimate effect for a single fact + intervention:

python -m src.utils.causal_sim_pdm estimate \
  --facts data/pdm_facts.jsonl \
  --model data/pdm_risk_model.joblib \
  --fact-id <FACT_ID> \
  --intervention-json '{"hours_since_last_maint_comp4": 0}'
"""

from __future__ import annotations
import json
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib


# ----------------------------
# Utilities to load facts
# ----------------------------

def load_facts_jsonl(path: str) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            facts.append(json.loads(line))
    return facts


def collect_feature_names(facts: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for fact in facts:
        for feat in fact.get("features", []):
            n = feat.get("name")
            if n is not None:
                names.add(n)
    return sorted(names)


def fact_to_vector(
    fact: Dict[str, Any],
    feature_names: List[str],
) -> np.ndarray:
    """
    Convert one fact's features into a fixed-length numeric vector
    aligned with feature_names. Missing features -> 0.0.
    """
    feat_map = {f["name"]: f["value"] for f in fact.get("features", []) if "name" in f}
    x = np.zeros(len(feature_names), dtype=float)
    for i, name in enumerate(feature_names):
        val = feat_map.get(name, 0.0)
        try:
            x[i] = float(val)
        except Exception:
            x[i] = 0.0
    return x


def build_feature_matrix_and_labels(
    facts: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build X (feature matrix), y_str (string labels), and feature_names.

    Filters out facts with missing/empty labels.
    """
    feature_names = collect_feature_names(facts)
    X_rows: List[np.ndarray] = []
    y_str: List[str] = []

    for fact in facts:
        label = fact.get("label")
        if label is None:
            continue
        label = str(label).strip()
        if not label:
            continue

        x = fact_to_vector(fact, feature_names)
        X_rows.append(x)
        y_str.append(label)

    if not X_rows:
        raise RuntimeError("No valid labeled facts found to build training data.")

    X = np.stack(X_rows, axis=0)
    y = np.array(y_str, dtype=str)
    return X, y, feature_names


# ----------------------------
# Model training / saving
# ----------------------------

def fit_risk_model(
    facts_path: str,
    model_out_path: str,
    C: float = 1.0,
    max_iter: int = 500,
) -> Dict[str, Any]:
    """
    Fit a multi-class logistic regression model P(label | features) on PdM facts.

    Saves a bundle with:
      - model (sklearn LogisticRegression)
      - label_encoder (sklearn LabelEncoder)
      - feature_names (list of str)

    Returns a summary dict.
    """
    facts = load_facts_jsonl(facts_path)
    X, y_str, feature_names = build_feature_matrix_and_labels(facts)

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # Multi-class logistic regression
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
    )
    clf.fit(X, y)

    bundle = {
        "model": clf,
        "label_encoder": le,
        "feature_names": feature_names,
    }
    joblib.dump(bundle, model_out_path)

    summary = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": list(le.classes_),
        "model_path": model_out_path,
    }
    return summary


def load_risk_model(model_path: str) -> Dict[str, Any]:
    return joblib.load(model_path)


# ----------------------------
# Counterfactual estimation
# ----------------------------

def apply_intervention_vector(
    x: np.ndarray,
    feature_names: List[str],
    intervention: Dict[str, float],
) -> np.ndarray:
    """
    Apply intervention of the form:
      {"feature_name": new_value, ...}
    Returns a new vector x_cf.
    """
    x_cf = x.copy()
    for feat_name, new_val in intervention.items():
        if feat_name not in feature_names:
            # silently ignore unknown features for now
            continue
        idx = feature_names.index(feat_name)
        try:
            x_cf[idx] = float(new_val)
        except Exception:
            # if can't cast, ignore
            pass
    return x_cf


def estimate_effect(
    fact: Dict[str, Any],
    model_bundle: Dict[str, Any],
    intervention: Dict[str, float],
) -> Dict[str, Any]:
    """
    Estimate effect of an intervention on failure risk.

    - fact: one PdM episode fact (with features[] and label).
    - model_bundle: output of load_risk_model(model_path).
    - intervention: dict mapping feature_name -> new_value.

    Returns:
      {
        "focus": "any_failure",
        "classes": [...],
        "risk_before": float,
        "risk_after": float,
        "delta_risk": float,
        "direction": "increase|decrease|no_change",
        "probs_before": {class -> prob},
        "probs_after": {class -> prob},
        "confidence": float
      }
    """
    clf = model_bundle["model"]
    le = model_bundle["label_encoder"]
    feature_names = model_bundle["feature_names"]
    classes = list(le.classes_)

    # Vector for original fact
    x = fact_to_vector(fact, feature_names)
    probs_before = clf.predict_proba(x.reshape(1, -1))[0]  # shape [n_classes]

    # Vector after intervention
    x_cf = apply_intervention_vector(x, feature_names, intervention)
    probs_after = clf.predict_proba(x_cf.reshape(1, -1))[0]

    # Map class -> prob for inspection
    probs_before_map = {cls: float(p) for cls, p in zip(classes, probs_before)}
    probs_after_map = {cls: float(p) for cls, p in zip(classes, probs_after)}

    # Risk of any failure = 1 - P(healthy)
    if "healthy" in classes:
        idx_h = classes.index("healthy")
        p_healthy_before = probs_before[idx_h]
        p_healthy_after = probs_after[idx_h]
    else:
        # if no healthy class, treat all probability as "failure"
        p_healthy_before = 0.0
        p_healthy_after = 0.0

    risk_before = float(1.0 - p_healthy_before)
    risk_after = float(1.0 - p_healthy_after)
    delta = risk_after - risk_before

    if delta > 1e-3:
        direction = "increase"
    elif delta < -1e-3:
        direction = "decrease"
    else:
        direction = "no_change"

    # Simple confidence heuristic: more extreme probs -> higher confidence
    # (just a placeholder; you can refine later)
    # confidence in [0.5, 1.0], higher when risk is near 0 or 1.
    center = 0.5
    dist = abs(risk_before - center)
    confidence = float(0.5 + 0.5 * min(1.0, dist / 0.5))

    result = {
        "focus": "any_failure",
        "classes": classes,
        "risk_before": risk_before,
        "risk_after": risk_after,
        "delta_risk": float(delta),
        "direction": direction,
        "probs_before": probs_before_map,
        "probs_after": probs_after_map,
        "confidence": round(confidence, 3),
    }
    return result


def find_fact_by_id(facts: List[Dict[str, Any]], fact_id: str) -> Dict[str, Any]:
    for fact in facts:
        if fact.get("fact_id") == fact_id:
            return fact
    raise KeyError(f"fact_id {fact_id} not found in facts.")


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Causal & Counterfactual Simulator for PdM facts.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # fit mode
    p_fit = subparsers.add_parser("fit", help="Fit risk model on PdM facts.")
    p_fit.add_argument("--facts", required=True, help="Path to pdm_facts.jsonl")
    p_fit.add_argument("--model-out", required=True, help="Output path for model bundle (e.g. data/pdm_risk_model.joblib)")
    p_fit.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression")
    p_fit.add_argument("--max-iter", type=int, default=500, help="Max iterations for LogisticRegression")

    # estimate mode
    p_est = subparsers.add_parser("estimate", help="Estimate effect of an intervention for one fact.")
    p_est.add_argument("--facts", required=True, help="Path to pdm_facts.jsonl")
    p_est.add_argument("--model", required=True, help="Path to trained model bundle (joblib)")
    p_est.add_argument("--fact-id", required=True, help="fact_id of the episode for which to estimate effect")
    p_est.add_argument(
        "--intervention-json",
        required=True,
        help="JSON string specifying intervention, e.g. '{\"hours_since_last_maint_comp4\": 0}'",
    )

    args = parser.parse_args()

    if args.mode == "fit":
        os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
        summary = fit_risk_model(
            facts_path=args.facts,
            model_out_path=args.model_out,
            C=args.C,
            max_iter=args.max_iter,
        )
        print(json.dumps(summary, indent=2))

    elif args.mode == "estimate":
        facts = load_facts_jsonl(args.facts)
        bundle = load_risk_model(args.model)

        fact = find_fact_by_id(facts, args.fact_id)
        intervention = json.loads(args.intervention_json)

        result = estimate_effect(
            fact=fact,
            model_bundle=bundle,
            intervention=intervention,
        )
        print(json.dumps(result, indent=2))
