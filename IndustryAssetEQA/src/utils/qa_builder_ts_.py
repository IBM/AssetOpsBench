# src/utils/qa_builder_ts.py
"""
QA builder for time-series PdM facts with ontology / FMEA enrichment surfaced
in questions, answers and provenance.

See original docstring in the repository for usage examples.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from joblib import load as joblib_load

# Path to optional ISO / FMEA metadata JSON used as a fallback when facts lack failure_profile.
ISO_FAILURES_PATH = "data/iso_failure_mode_metadata.json"


def _load_iso_failure_index(path: str) -> Dict[str, Any]:
    """Load an optional ISO/FMEA JSON mapping failure labels -> metadata."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as fh:
            # support both list-of-objects and dict-of-objects formats:
            obj = json.load(fh)
            # if it's a list of dicts with 'label' or 'failure_mode' keys, create a dict index
            if isinstance(obj, list):
                idx = {}
                for entry in obj:
                    if not isinstance(entry, dict):
                        continue
                    # prefer canonical keys
                    key = entry.get("label") or entry.get("failure_mode") or entry.get("name")
                    if key:
                        idx[str(key)] = entry
                return idx
            if isinstance(obj, dict):
                return obj
            return {}
    except Exception:
        return {}


# module-level cache
_ISO_FAILURE_IDX = _load_iso_failure_index(ISO_FAILURES_PATH)


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
# Ontology helpers & enrichment access
# --------------------

def _get_ontology_parts(fact: Dict[str, Any]):
    """Helper to return (asset_profile, failure_profile, sensor_profiles, prov_counts).
    If failure_profile is missing in the fact, attempt a fallback lookup in the ISO index.
    This version normalizes nested `iso_metadata` inside failure_profile by merging it up
    so that downstream code can read fields at top-level.
    """
    asset_profile = fact.get("asset_profile") or {}
    failure_profile = fact.get("failure_profile")  # can be None
    sensor_profiles = fact.get("sensor_profiles") or []
    provenance = fact.get("provenance") or {}

    # --- NORMALIZE: if failure_profile has nested iso_metadata, merge it up ---
    if failure_profile and isinstance(failure_profile, dict):
        iso = failure_profile.get("iso_metadata")
        if iso and isinstance(iso, dict):
            # create a shallow copy and merge iso_metadata fields into top-level
            merged = dict(failure_profile)
            merged.update(iso)
            failure_profile = merged

    # fallback: if failure_profile missing/empty, try ISO index by label
    if (failure_profile is None or failure_profile == {}) and fact.get("label"):
        lbl = str(fact.get("label"))
        # try exact match
        if lbl in _ISO_FAILURE_IDX:
            failure_profile = _ISO_FAILURE_IDX[lbl]
        else:
            # try lowercase key variants
            lbl_l = lbl.lower()
            if lbl_l in _ISO_FAILURE_IDX:
                failure_profile = _ISO_FAILURE_IDX[lbl_l]
            else:
                # try keys that contain label as substring
                for k, v in _ISO_FAILURE_IDX.items():
                    try:
                        if lbl_l in str(k).lower():
                            failure_profile = v
                            break
                    except Exception:
                        continue

    # OPTIONAL: if asset_profile missing equipment fields, copy from failure_profile when available.
    # This helps when facts only contain equipment info inside failure_profile/iso_metadata.
    if failure_profile and isinstance(failure_profile, dict):
        # do not mutate original asset_profile object from fact
        if not asset_profile.get("equipment_category") and failure_profile.get("equipment_category"):
            asset_profile = dict(asset_profile)
            asset_profile["equipment_category"] = failure_profile.get("equipment_category")
        if not asset_profile.get("equipment_class_type") and failure_profile.get("equipment_class_type"):
            asset_profile = dict(asset_profile)
            asset_profile["equipment_class_type"] = failure_profile.get("equipment_class_type")

    # telemetry/error/maint counts (optional)
    tele_points = provenance.get("telemetry_points_in_window") or provenance.get("telemetry_points") or None
    errors_in_window = provenance.get("errors_in_window") or None
    maint_events_in_window = provenance.get("maint_events_in_window") or None
    return asset_profile, failure_profile, sensor_profiles, {
        "telemetry_points": tele_points,
        "errors_in_window": errors_in_window,
        "maint_events_in_window": maint_events_in_window,
    }


def _get_sensor_description(sensor_profiles: List[Dict[str, Any]], sensor_name: str) -> Optional[str]:
    for sp in sensor_profiles:
        if sp.get("sensor_name") == sensor_name:
            return sp.get("description")
    return None


# --------------------
# QA builders (patched to include ontology/FMEA where available)
# --------------------

def _brief_failure_profile(failure_profile: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not failure_profile:
        return None
    return {
        "failure_mode": failure_profile.get("failure_mode") or failure_profile.get("dataset_labels") or None,
        "display_name": failure_profile.get("display_name") or failure_profile.get("name") or None,
        "severity": failure_profile.get("severity") or None,
        "associated_sensors": failure_profile.get("associated_sensors") or None,
        "recommended_actions": failure_profile.get("recommended_actions")[:3] if failure_profile.get("recommended_actions") else None,
    }


def _brief_asset_profile(asset_profile: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not asset_profile:
        return None
    return {
        "asset_name": asset_profile.get("asset_name") or None,
        "equipment_category": asset_profile.get("equipment_category") or None,
        "equipment_class_type": asset_profile.get("equipment_class_type") or None,
    }


def build_diagnostic_qa_for_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnostic QA: explain the failure label for this episode.
    Surfaces failure_profile, asset_profile and relevant provenance.
    """
    fact_id = fact["fact_id"]
    asset_id = fact.get("asset_id", fact.get("machineID"))
    label = str(fact.get("label", "unknown"))
    start = fact.get("start_time")
    end = fact.get("end_time")
    qa_id = f"pdm_diag_{fact_id}"

    asset_profile, failure_profile, sensor_profiles, prov_counts = _get_ontology_parts(fact)

    # Construct question — mention failure_profile name if present
    fp_name = None
    if failure_profile and isinstance(failure_profile, dict):
        fp_name = failure_profile.get("name") or failure_profile.get("display_name")
    if fp_name:
        question = (
            f"Why is this diagnostic episode for asset {asset_id} labeled '{label}' "
            f"({fp_name}) over the time window {start} to {end}?"
        )
    else:
        question = (
            f"Why is this diagnostic episode for asset {asset_id} labeled '{label}' "
            f"over the time window {start} to {end}?"
        )

    # Pick features: prefer associated_sensors -> map to their summary stats
    used_feats: List[tuple] = []
    if failure_profile and isinstance(failure_profile, dict):
        assoc = failure_profile.get("associated_sensors", []) or []
        candidate_names: List[str] = []
        for s in assoc:
            # try common prefixes: mean/max/trend
            candidate_names.extend([f"{s}_mean", f"{s}_max", f"{s}_trend"])
        for name in candidate_names:
            val = get_feat(fact, name)
            if val is not None:
                used_feats.append((name, val))
                if len(used_feats) >= 4:
                    break

    if not used_feats:
        # fallback: sample first 3 numeric features
        feats_all = fact.get("features", [])
        used_feats = [(f["name"], f["value"]) for f in feats_all[:3]]

    feat_str = ", ".join(f"{n}={v:.3f}" for n, v in used_feats)

    # Compose direct answer: reference ontology if possible
    direct_answer = f"This episode is labeled '{label}' because the observed telemetry features match indicators of the '{label}' failure mode."
    if fp_name:
        direct_answer = (
            f"This episode is labeled '{label}' ({fp_name}) because the observed features "
            f"match the typical indicators associated with this failure mode."
        )

    # Compose reasoning: inject typical_indicators and recommended actions if present
    reasoning_parts: List[str] = []
    reasoning_parts.append(f"Key observed features: {feat_str}.")
    if failure_profile and isinstance(failure_profile, dict):
        ti = failure_profile.get("typical_indicators")
        if ti and isinstance(ti, dict):
            try:
                example_inds = []
                for k, v in list(ti.items())[:3]:
                    example_inds.append(f"{k}: {v}")
                reasoning_parts.append("Typical indicators include " + "; ".join(example_inds) + ".")
            except Exception:
                pass
        rec = failure_profile.get("recommended_actions")
        if rec and isinstance(rec, list) and len(rec) > 0:
            reasoning_parts.append("Suggested inspection steps: " + "; ".join(rec[:3]) + ".")
        sev = failure_profile.get("severity")
        if sev:
            reasoning_parts.append(f"Severity rating: {sev}.")

    reasoning_answer = " ".join(reasoning_parts)

    provenance = {
        "fact_id": fact_id,
        "features": [n for n, _ in used_feats],
        "file": fact.get("source_file"),
        "row": fact.get("row_index"),
        "asset_profile": _brief_asset_profile(asset_profile),
        "failure_profile_id": failure_profile.get("failure_mode") if failure_profile else None,
        "telemetry_points_in_window": prov_counts["telemetry_points"],
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
        # explicit brief KG excerpts
        "asset_profile_brief": _brief_asset_profile(asset_profile),
        "failure_profile_brief": _brief_failure_profile(failure_profile),
    }
    return qa


def build_descriptive_qa_for_fact(fact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Descriptive QA: ask about a numeric summary in the window.
    Surfaces sensor_profiles and telemetry counts in provenance.
    """
    fact_id = fact["fact_id"]
    asset_id = fact.get("asset_id", fact.get("machineID"))
    start = fact.get("start_time")
    end = fact.get("end_time")
    qa_id = f"pdm_desc_{fact_id}"

    _, _, sensor_profiles, prov_counts = _get_ontology_parts(fact)

    # prefer a sensor we know about: vibration or first numeric column
    vib_mean = get_feat(fact, "vibration_mean")
    sensor_name = "vibration"
    if vib_mean is None:
        # fallback: pick first numeric feature from features list
        feats_all = fact.get("features", [])
        if not feats_all:
            return None
        first_feat = feats_all[0]["name"]
        # infer sensor name prefix (e.g., "s1_mean" => "s1")
        if "_" in first_feat:
            sensor_name = first_feat.split("_")[0]
        vib_mean = feats_all[0]["value"]
    else:
        vib_mean = vib_mean

    # sensor description if available
    sensor_desc = _get_sensor_description(sensor_profiles, sensor_name)
    sensor_desc_txt = f" ({sensor_desc})" if sensor_desc else ""

    question = (
        f"During the time window {start} to {end} for asset {asset_id}, what was the average "
        f"{sensor_name} level{sensor_desc_txt}?"
    )

    direct_answer = f"The average {sensor_name} level was approximately {float(vib_mean):.2f}."
    reasoning_answer = (
        f"In this episode for asset {asset_id}, the feature {sensor_name}_mean is {float(vib_mean):.2f}, "
        f"computed over {prov_counts.get('telemetry_points','an unknown number of')} telemetry points "
        f"in the window {start} to {end}."
    )

    provenance = {
        "fact_id": fact_id,
        "features": [f"{sensor_name}_mean"],
        "file": fact.get("source_file"),
        "row": fact.get("row_index"),
        "telemetry_points_in_window": prov_counts["telemetry_points"],
        "sensor_description": sensor_desc,
    }

    qa = {
        "qa_id": qa_id,
        "fact_id": fact_id,
        "task_type": "descriptive",
        "question": question,
        "direct_answer": direct_answer,
        "reasoning_answer": reasoning_answer,
        "provenance": provenance,
        "label": f"{float(vib_mean):.4f}",
        "asset_id": asset_id,
        # explicit brief KG excerpts (sensor descriptions live in provenance)
        "asset_profile_brief": _brief_asset_profile(fact.get("asset_profile")),
    }
    return qa


def build_temporal_count_qa_for_fact(fact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Temporal/count QA: how many distinct error types in the window?
    Includes provenance counts.
    """
    fact_id = fact["fact_id"]
    asset_id = fact.get("asset_id", fact.get("machineID"))
    start = fact.get("start_time")
    end = fact.get("end_time")
    qa_id = f"pdm_temp_{fact_id}"

    _, _, _, prov_counts = _get_ontology_parts(fact)

    n_errors = get_feat(fact, "error_count_last_window")
    n_distinct = get_feat(fact, "distinct_error_types_last_window")
    if n_errors is None or n_distinct is None:
        return None

    question = (
        f"Between {start} and {end} for asset {asset_id}, how many distinct error types occurred? "
        f"(telemetry points: {prov_counts.get('telemetry_points','unknown')})"
    )

    direct_answer = f"There were {int(n_distinct)} distinct error types in this window."
    reasoning_answer = (
        f"In this episode the feature distinct_error_types_last_window is {int(n_distinct)}, "
        f"and the total error count is {int(n_errors)}. Telemetry points in window: "
        f"{prov_counts.get('telemetry_points','unknown')}."
    )

    provenance = {
        "fact_id": fact_id,
        "features": ["error_count_last_window", "distinct_error_types_last_window"],
        "file": fact.get("source_file"),
        "row": fact.get("row_index"),
        "errors_in_window": prov_counts.get("errors_in_window"),
        "telemetry_points_in_window": prov_counts.get("telemetry_points"),
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
        # keep brief asset info if available
        "asset_profile_brief": _brief_asset_profile(fact.get("asset_profile")),
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
    Augments question and reasoning with failure_profile hints.
    """
    fact_id = fact["fact_id"]
    asset_id = fact.get("asset_id", fact.get("machineID"))
    label = str(fact.get("label", "unknown"))
    start = fact.get("start_time")
    end = fact.get("end_time")
    row_index = fact.get("row_index", 0)

    asset_profile, failure_profile, sensor_profiles, prov_counts = _get_ontology_parts(fact)

    # choose component: prefer labeled component, else fallback
    if label in {"comp1", "comp2", "comp3", "comp4"}:
        comp = label
    else:
        comp = label if label != "healthy" else "comp3"

    feat_name = f"hours_since_last_maint_{comp}"
    if get_feat(fact, feat_name) is None:
        # fallback: try generic 'hours_since_last_maint_any'
        if get_feat(fact, "hours_since_last_maint_any") is None:
            return None
        feat_name = "hours_since_last_maint_any"

    # baseline risk
    risk_before = compute_any_failure_risk(model, classes, feature_order, fact)

    # intervention: reset chosen hours-since-maint to 0
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

    # Augment question with failure_profile if present
    fp_brief = None
    if failure_profile and isinstance(failure_profile, dict):
        fp_brief = failure_profile.get("failure_mode") or failure_profile.get("display_name")
    if fp_brief:
        question = (
            f"For asset {asset_id} in the time window {start} to {end} (failure mode: {fp_brief}), "
            f"suppose we had performed maintenance on {comp} at the start of the window, resetting "
            f"{feat_name} to 0. How would the risk of any component failure change?"
        )
    else:
        question = (
            f"For asset {asset_id} in the time window {start} to {end}, suppose we had performed "
            f"maintenance on {comp} at the start of the window, resetting {feat_name} to 0. "
            "How would the risk of any component failure change?"
        )

    direct_answer = f"The risk of failure would be expected to {direction} under this intervention."

    # Reasoning: include simulator numbers + ontology hints (recommended actions)
    reasoning_parts: List[str] = [
        f"Baseline failure risk (any component) ≈ {risk_before:.3f}. Post-intervention risk ≈ {risk_after:.3f}. Δ={risk_after-risk_before:.3f}."
    ]
    if failure_profile and isinstance(failure_profile, dict):
        rec = failure_profile.get("recommended_actions")
        if rec and isinstance(rec, list) and len(rec) > 0:
            reasoning_parts.append("Actions typically recommended for this failure mode include: " + "; ".join(rec[:3]) + ".")
        assoc = failure_profile.get("associated_sensors")
        if assoc:
            reasoning_parts.append("Most informative sensors for this failure mode: " + ", ".join(assoc[:4]) + ".")
    reasoning_answer = " ".join(reasoning_parts)

    provenance = {
        "fact_id": fact_id,
        "features": [feat_name] + feature_order,
        "file": fact.get("source_file"),
        "row": row_index,
        "telemetry_points_in_window": prov_counts.get("telemetry_points"),
        "failure_profile_id": failure_profile.get("failure_mode") if failure_profile else None,
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
        # explicit brief KG excerpts
        "asset_profile_brief": _brief_asset_profile(asset_profile),
        "failure_profile_brief": _brief_failure_profile(failure_profile),
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
    Uses risk model and surfaces KG-recommended actions and severity as reasoning.
    """
    fact_id = fact["fact_id"]
    asset_id = fact.get("asset_id", fact.get("machineID"))
    start = fact.get("start_time")
    end = fact.get("end_time")
    row_index = fact.get("row_index", 0)

    asset_profile, failure_profile, sensor_profiles, prov_counts = _get_ontology_parts(fact)

    risk = compute_any_failure_risk(model, classes, feature_order, fact)
    action = "open_work_order" if risk >= threshold else "monitor"

    qa_id = f"pdm_action_{fact_id}"

    # Compose question mentioning asset category if available
    asset_cat = asset_profile.get("equipment_category") if asset_profile else None
    if asset_cat:
        question = (
            f"For {asset_cat} {asset_id} in the time window {start} to {end}, should a maintenance "
            "work order be opened now, or is it acceptable to continue monitoring?"
        )
    else:
        question = (
            f"For asset {asset_id} in the time window {start} to {end}, should a maintenance "
            "work order be opened now, or is it acceptable to continue monitoring?"
        )

    if action == "open_work_order":
        direct_answer = "A maintenance work order should be opened now."
    else:
        direct_answer = "It is acceptable to continue monitoring for now."

    # reasoning: risk numbers + severity + suggested actions from KG
    reasoning_parts: List[str] = [
        f"The learned risk model estimates probability of any failure ≈ {risk:.2f} (threshold={threshold:.2f})."
    ]
    if failure_profile and isinstance(failure_profile, dict):
        sev = failure_profile.get("severity")
        recs = failure_profile.get("recommended_actions")
        if sev:
            reasoning_parts.append(f"Failure severity: {sev}.")
        if recs and isinstance(recs, list) and len(recs) > 0:
            reasoning_parts.append("Recommended diagnostics/actions: " + "; ".join(recs[:3]) + ".")
    reasoning_answer = " ".join(reasoning_parts)

    provenance = {
        "fact_id": fact_id,
        "features": feature_order,
        "file": fact.get("source_file"),
        "row": row_index,
        "telemetry_points_in_window": prov_counts.get("telemetry_points"),
        "failure_profile_id": failure_profile.get("failure_mode") if failure_profile else None,
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
        # explicit brief KG excerpts
        "asset_profile_brief": _brief_asset_profile(asset_profile),
        "failure_profile_brief": _brief_failure_profile(failure_profile),
    }
    return qa


# --------------------
# Top-level builders (unchanged behavior; they now call patched builders)
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
