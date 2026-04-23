# src/utils/qa_builder_ts_cf_.py
"""
Build counterfactual QA instances for PdM time-series episodes with minimal,
safe inclusion of ontology / FMEA details into the question, reasoning, and provenance.

This is a minimal patch over the original builder: it does NOT change the
counterfactual estimator or decision logic. It simply surfaces available
ontology fields already present in each fact (asset_profile, failure_profile,
sensor_profiles, recommended_actions, associated_sensors, severity, etc.)
into the generated QA JSON so downstream prompt builders or evaluators can use them.

Usage (example):
python -m src.utils.qa_builder_ts_cf \
  --facts data/pdm_facts.jsonl \
  --model data/pdm_risk_model.joblib \
  --out data/pdm_qa_cf.jsonl
"""
from __future__ import annotations
import json
from typing import Dict, Any, List, Optional

from src.utils.causal_sim_pdm import (
    load_facts_jsonl,
    load_risk_model,
    estimate_effect,
)


# --------------------
# Helpers
# --------------------

def _find_feat_value(fact: Dict[str, Any], feat_name: str) -> Optional[float]:
    for feat in fact.get("features", []):
        if feat.get("name") == feat_name:
            try:
                return float(feat.get("value"))
            except Exception:
                return None
    return None


def _merge_iso_metadata_into_failure_profile(fp: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    If a failure_profile contains an 'iso_metadata' dict, merge its keys up into
    the top-level failure_profile (without overwriting existing top-level keys).
    Returns a sanitized dict (or None).
    """
    if not fp or not isinstance(fp, dict):
        return fp
    iso = fp.get("iso_metadata")
    if not iso or not isinstance(iso, dict):
        return fp
    merged = dict(fp)  # shallow copy
    for k, v in iso.items():
        if merged.get(k) is None:
            merged[k] = v
    return merged


def _brief_failure_profile(failure_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a compact dict with the most useful KG fields (safe to be None)."""
    fp = _merge_iso_metadata_into_failure_profile(failure_profile) if failure_profile else None
    if not fp or not isinstance(fp, dict):
        return {}
    # Try multiple key names commonly used
    failure_mode = fp.get("failure_mode") or fp.get("failure_label") or fp.get("dataset_labels")
    display_name = fp.get("display_name") or fp.get("name") or fp.get("dataset_labels")
    short_description = (
        fp.get("description")
        or (fp.get("iso_metadata", {}) and fp.get("iso_metadata", {}).get("description"))
        or None
    )
    associated = fp.get("associated_sensors") or (fp.get("iso_metadata", {}) and fp.get("iso_metadata", {}).get("associated_sensors"))
    # normalized associated to list if single string
    if isinstance(associated, str):
        associated = [associated]
    typical_indicators = fp.get("typical_indicators") or (fp.get("iso_metadata", {}) and fp.get("iso_metadata", {}).get("typical_indicators"))
    recommended = fp.get("recommended_actions") or (fp.get("iso_metadata", {}) and fp.get("iso_metadata", {}).get("recommended_actions"))
    severity = fp.get("severity") or (fp.get("iso_metadata", {}) and fp.get("iso_metadata", {}).get("severity"))
    return {
        "failure_mode": failure_mode,
        "display_name": display_name,
        "short_description": short_description,
        "associated_sensors": associated,
        "typical_indicators": typical_indicators,
        "recommended_actions": recommended,
        "severity": severity,
    }


def _brief_asset_profile(asset_profile: Optional[Dict[str, Any]], failure_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a compact asset brief. If equipment fields are missing from asset_profile,
    attempt to copy them from failure_profile (which may have iso_metadata merged).
    """
    ap = {}
    if asset_profile and isinstance(asset_profile, dict):
        ap = dict(asset_profile)  # shallow copy
    # If asset fields missing, try to pull from failure_profile iso metadata
    if (not ap.get("equipment_category") or not ap.get("equipment_class_type") or not ap.get("unit_subunit")) and failure_profile:
        fp = _merge_iso_metadata_into_failure_profile(failure_profile)
        if isinstance(fp, dict):
            ec = fp.get("equipment_category") or fp.get("equipment_category")
            ect = fp.get("equipment_class_type") or fp.get("equipment_class_type")
            us = fp.get("unit_subunit") or fp.get("unit_subunit")
            if ec and not ap.get("equipment_category"):
                ap["equipment_category"] = ec
            if ect and not ap.get("equipment_class_type"):
                ap["equipment_class_type"] = ect
            if us and not ap.get("unit_subunit"):
                ap["unit_subunit"] = us
    # ensure asset_name present in brief
    asset_name = ap.get("asset_name") or ap.get("model")
    if asset_name:
        ap["asset_name"] = asset_name
    return {
        "equipment_category": ap.get("equipment_category"),
        "equipment_class_type": ap.get("equipment_class_type"),
        "unit_subunit": ap.get("unit_subunit"),
        "asset_name": ap.get("asset_name"),
    }


def _sensor_brief_list(sensor_profiles: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    out = []
    if not sensor_profiles:
        return out
    for sp in sensor_profiles:
        if not isinstance(sp, dict):
            continue
        out.append({
            "sensor_name": sp.get("sensor_name"),
            "description": sp.get("description"),
        })
    return out


# --------------------
# Counterfactual builder (existing)
# --------------------

def build_counterfactual_qa_dataset(
    facts_path: str,
    model_path: str,
    out_path: str,
    dataset_name: str = "pdm_ts",
    per_label: int | None = None,
    min_abs_delta: float = 1e-3,
) -> int:
    """
    Build counterfactual QAs for PdM with ontology/FMEA context surfaced.

    Minimal, safe behavior:
      - If asset_profile / failure_profile / sensor_profiles exist in the fact,
        include compact, sanitized snippets of them in the QA output:
          * asset_profile_brief
          * failure_profile_brief
          * sensor_profiles_brief
      - Add ontology-derived hints into the question and reasoning text where available.
      - Do not change the intervention/risk estimation logic.
    """
    facts = load_facts_jsonl(facts_path)
    bundle = load_risk_model(model_path)

    counts_per_label: Dict[str, int] = {}
    written = 0

    with open(out_path, "w") as fout:
        for fact in facts:
            label = fact.get("label")
            if not label or str(label).lower() == "healthy":
                continue  # only failure episodes

            # optional cap per label
            if per_label is not None:
                c = counts_per_label.get(label, 0)
                if c >= per_label:
                    continue

            fact_id = fact.get("fact_id")
            asset_id = fact.get("asset_id", "unknown_asset")
            start_time = fact.get("start_time")
            end_time = fact.get("end_time")

            # maintenance-related feature name (e.g., hours_since_last_maint_comp3)
            maint_feat = f"hours_since_last_maint_{label}"
            feat_names = [f.get("name") for f in fact.get("features", [])]
            if maint_feat not in feat_names:
                # fallback to a generic maintenance-age feature if present
                if "hours_since_last_maint_any" in feat_names:
                    maint_feat = "hours_since_last_maint_any"
                else:
                    # skip if no maintenance feature present
                    continue

            current_val = _find_feat_value(fact, maint_feat)

            # Run the estimator (unchanged)
            cf_result = estimate_effect(
                fact=fact,
                model_bundle=bundle,
                intervention={maint_feat: 0.0},
            )

            # filter trivial deltas
            if abs(cf_result.get("delta_risk", 0.0)) < min_abs_delta:
                continue

            direction = cf_result.get("direction", "no_change")
            risk_before = cf_result.get("risk_before", 0.0)
            risk_after = cf_result.get("risk_after", 0.0)
            delta_risk = cf_result.get("delta_risk", 0.0)
            cf_conf = cf_result.get("confidence", 0.0)

            # Extract ontology snippets if present in fact; merge iso_metadata into failure_profile
            raw_failure_profile = fact.get("failure_profile")
            failure_profile = _merge_iso_metadata_into_failure_profile(raw_failure_profile) if raw_failure_profile else raw_failure_profile
            asset_profile = fact.get("asset_profile")
            sensor_profiles = fact.get("sensor_profiles")

            asset_brief = _brief_asset_profile(asset_profile, failure_profile)
            failure_brief = _brief_failure_profile(failure_profile)
            sensors_brief = _sensor_brief_list(sensor_profiles)

            # Associated sensors from failure_profile if present (ensure list or None)
            assoc_sensors = failure_brief.get("associated_sensors") or None
            if isinstance(assoc_sensors, (list, tuple)):
                assoc_sensors = list(assoc_sensors)
            elif isinstance(assoc_sensors, str):
                assoc_sensors = [assoc_sensors]
            else:
                assoc_sensors = None

            # Compose question text: include failure display_name if available
            fp_name = failure_brief.get("display_name") or failure_brief.get("failure_mode")
            equipment_cat = asset_brief.get("equipment_category")
            # format current_val safely
            current_val_txt = f"{current_val}" if current_val is not None else "unknown"
            if fp_name and equipment_cat:
                question = (
                    f"For {equipment_cat} {asset_id} (failure mode: {fp_name}) in the window "
                    f"{start_time} to {end_time}, if maintenance targeting {label} had been "
                    f"performed immediately before the window (resetting {maint_feat} from "
                    f"{current_val_txt} to 0), how would the risk of failure change?"
                )
            elif fp_name:
                question = (
                    f"For asset {asset_id} (failure mode: {fp_name}) in the window "
                    f"{start_time} to {end_time}, if maintenance targeting {label} had been "
                    f"performed immediately before the window (resetting {maint_feat} from "
                    f"{current_val_txt} to 0), how would the risk of failure change?"
                )
            else:
                question = (
                    f"For asset {asset_id} in the window {start_time} to {end_time}, if maintenance "
                    f"on {label} had been performed immediately before the window (resetting "
                    f"{maint_feat} from {current_val_txt} to 0), how would the failure risk change?"
                )

            # Compose a KG-aware reasoning answer (append KG hints when available)
            reasoning_parts: List[str] = []
            reasoning_parts.append(
                f"The learned risk model estimates baseline P(failure) ≈ {risk_before:.3f} "
                f"and post-intervention P(failure) ≈ {risk_after:.3f} (Δ = {delta_risk:.3f})."
            )
            # Add failure-profile hints
            if failure_brief:
                if failure_brief.get("short_description"):
                    reasoning_parts.append(f"Failure profile: {failure_brief.get('short_description')}")
                if assoc_sensors:
                    reasoning_parts.append(f"This failure mode is typically associated with sensors: {', '.join(assoc_sensors)}.")
                if failure_brief.get("recommended_actions"):
                    recs = failure_brief.get("recommended_actions")
                    # include up to 3 recommended actions as a hint
                    try:
                        reasoning_parts.append("Typical recommended actions: " + "; ".join(recs[:3]) + ".")
                    except Exception:
                        pass
                if failure_brief.get("severity"):
                    reasoning_parts.append(f"Reported severity: {failure_brief.get('severity')}.")
            # Add asset hints
            if asset_brief:
                if asset_brief.get("equipment_category"):
                    reasoning_parts.append(f"Equipment category: {asset_brief.get('equipment_category')}.")
                if asset_brief.get("equipment_class_type"):
                    reasoning_parts.append(f"Equipment class/type: {asset_brief.get('equipment_class_type')}.")
            # Add a short sensor description if available for the top associated sensor
            if sensors_brief and assoc_sensors:
                top = assoc_sensors[0]
                for sp in sensors_brief:
                    if sp.get("sensor_name") == top and sp.get("description"):
                        reasoning_parts.append(f"Sensor hint ({top}): {sp.get('description')}")
                        break

            reasoning_answer = " ".join(reasoning_parts)

            # Build provenance: include ontology identifiers / brief snippets
            provenance: Dict[str, Any] = {
                "fact_id": fact_id,
                "features": [maint_feat],
                "file": fact.get("source_file", "unknown"),
                "row": fact.get("row_index", -1),
                "telemetry_points_in_window": fact.get("provenance", {}).get("telemetry_points_in_window"),
                "errors_in_window": fact.get("provenance", {}).get("errors_in_window"),
                # surface brief ontology snippets (safe, small)
                "asset_profile_brief": asset_brief or None,
                "failure_profile_brief": failure_brief or None,
                "sensor_profiles_brief": sensors_brief or None,
            }

            # Direct answer (unchanged)
            if direction == "decrease":
                direct_answer = "The risk of failure would decrease."
            elif direction == "increase":
                direct_answer = "The risk of failure would increase."
            else:
                direct_answer = "The risk of failure would remain approximately the same."

            cf_block = {
                "intervention": f"do({maint_feat} = 0.0)",
                "risk_before": risk_before,
                "risk_after": risk_after,
                "delta_risk": delta_risk,
                "direction": direction,
                "probs_before": cf_result.get("probs_before"),
                "probs_after": cf_result.get("probs_after"),
            }

            qa_id = f"{dataset_name}_cf_{fact_id}"

            qa_obj: Dict[str, Any] = {
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
                # surface explicit small fields for downstream use
                "asset_profile_brief": asset_brief or None,
                "failure_profile_brief": failure_brief or None,
                "associated_sensors": assoc_sensors or None,
            }

            fout.write(json.dumps(qa_obj) + "\n")
            counts_per_label[label] = counts_per_label.get(label, 0) + 1
            written += 1

    return written


# --------------------
# Action recommendation builder (NEW)
# --------------------

def build_action_qa_dataset(
    facts_path: str,
    model_path: str,
    out_path: str,
    dataset_name: str = "pdm_ts",
    per_label: int | None = None,
    threshold: float = 0.5,
) -> int:
    """
    Build action recommendation QAs: should we open a work order now or continue monitoring?
    This function reuses the estimator API to obtain a baseline risk estimate for each fact
    and surfaces ontology / FMEA snippets similarly to the counterfactual builder.
    """
    facts = load_facts_jsonl(facts_path)
    bundle = load_risk_model(model_path)

    counts_per_label: Dict[str, int] = {}
    written = 0

    with open(out_path, "w") as fout:
        for fact in facts:
            label = fact.get("label")
            # include healthy episodes too — action can be about monitoring or opening work order
            if not label:
                # if there's no label at all, still proceed (asset-level action)
                label = "unknown"

            # optional cap per label
            if per_label is not None:
                c = counts_per_label.get(label, 0)
                if c >= per_label:
                    continue

            fact_id = fact.get("fact_id")
            asset_id = fact.get("asset_id", "unknown_asset")
            start_time = fact.get("start_time")
            end_time = fact.get("end_time")

            # Extract ontology snippets; normalize failure_profile iso metadata
            raw_failure_profile = fact.get("failure_profile")
            failure_profile = _merge_iso_metadata_into_failure_profile(raw_failure_profile) if raw_failure_profile else raw_failure_profile
            asset_profile = fact.get("asset_profile")
            sensor_profiles = fact.get("sensor_profiles")

            asset_brief = _brief_asset_profile(asset_profile, failure_profile)
            failure_brief = _brief_failure_profile(failure_profile)
            sensors_brief = _sensor_brief_list(sensor_profiles)

            # baseline risk: call estimator with empty intervention to obtain baseline
            try:
                est_result = estimate_effect(fact=fact, model_bundle=bundle, intervention={})
                # prefer explicit risk_before if present, otherwise fall back to 'risk' or 1 - healthy etc
                risk = est_result.get("risk_before", est_result.get("risk", None))
                probs_before = est_result.get("probs_before")
            except Exception:
                # If estimator fails for some facts, skip
                continue

            if risk is None:
                # skip if we can't compute baseline risk
                continue

            action = "open_work_order" if risk >= threshold else "monitor"

            # Compose question: mention equipment type if known to give context
            equipment_cat = asset_brief.get("equipment_category")
            if equipment_cat:
                question = (
                    f"For {equipment_cat} {asset_id} in the time window {start_time} to {end_time}, "
                    "should a maintenance work order be opened now, or is it acceptable to continue monitoring?"
                )
            else:
                question = (
                    f"For asset {asset_id} in the time window {start_time} to {end_time}, "
                    "should a maintenance work order be opened now, or is it acceptable to continue monitoring?"
                )

            # Direct answer
            if action == "open_work_order":
                direct_answer = "A maintenance work order should be opened now."
            else:
                direct_answer = "It is acceptable to continue monitoring for now."

            # Reasoning: include risk numbers and KG hints
            reasoning_parts: List[str] = [
                f"The learned risk model estimates probability of any failure ≈ {float(risk):.2f} (threshold={threshold:.2f})."
            ]
            if failure_brief:
                if failure_brief.get("severity"):
                    reasoning_parts.append(f"Failure severity: {failure_brief.get('severity')}.")
                if failure_brief.get("recommended_actions"):
                    recs = failure_brief.get("recommended_actions")
                    try:
                        reasoning_parts.append("Recommended diagnostics/actions: " + "; ".join(recs[:3]) + ".")
                    except Exception:
                        pass
                if failure_brief.get("associated_sensors"):
                    reasoning_parts.append("Most informative sensors: " + ", ".join(failure_brief.get("associated_sensors")[:4]) + ".")
            if asset_brief and asset_brief.get("equipment_class_type"):
                reasoning_parts.append(f"Equipment class/type: {asset_brief.get('equipment_class_type')}.")
            reasoning_answer = " ".join(reasoning_parts)

            provenance: Dict[str, Any] = {
                "fact_id": fact_id,
                "features": [f.get("name") for f in fact.get("features", [])],
                "file": fact.get("source_file", "unknown"),
                "row": fact.get("row_index", -1),
                "telemetry_points_in_window": fact.get("provenance", {}).get("telemetry_points_in_window"),
                "errors_in_window": fact.get("provenance", {}).get("errors_in_window"),
                "failure_profile_id": (failure_profile.get("failure_mode") if isinstance(failure_profile, dict) else None) or (failure_profile.get("failure_label") if isinstance(failure_profile, dict) else None),
                "asset_profile_brief": asset_brief or None,
                "failure_profile_brief": failure_brief or None,
            }

            qa_id = f"{dataset_name}_action_{fact_id}"
            qa_obj: Dict[str, Any] = {
                "qa_id": qa_id,
                "fact_id": fact_id,
                "task_type": "action_recommendation",
                "question": question,
                "direct_answer": direct_answer,
                "reasoning_answer": reasoning_answer,
                "provenance": provenance,
                "label": action,
                "asset_id": asset_id,
                "confidence_estimator": est_result.get("confidence", None),
                "risk": float(risk),
                "probs_before": probs_before,
                "asset_profile_brief": asset_brief or None,
                "failure_profile_brief": failure_brief or None,
            }

            fout.write(json.dumps(qa_obj) + "\n")
            counts_per_label[label] = counts_per_label.get(label, 0) + 1
            written += 1

    return written


# --------------------
# CLI
# --------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Build ontology-enriched counterfactual or action QA for PdM.")
    parser.add_argument("--facts", required=True, help="Path to pdm_facts.jsonl")
    parser.add_argument("--model", required=True, help="Path to PdM risk model (joblib or model bundle)")
    parser.add_argument("--out", required=True, help="Output path for QA JSONL")
    parser.add_argument(
        "--mode",
        choices=["counterfactual", "action_recommendation"],
        default="counterfactual",
        help="Type of QA to build.",
    )
    parser.add_argument("--dataset-name", default="pdm_ts", help="Prefix for qa_id")
    parser.add_argument("--per-label", type=int, default=None, help="Optional cap per failure label")
    parser.add_argument("--min-abs-delta", type=float, default=1e-3, help="Min |delta_risk| to keep a counterfactual QA")
    parser.add_argument("--threshold", type=float, default=0.5, help="Risk threshold for action_recommendation")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.mode == "counterfactual":
        n = build_counterfactual_qa_dataset(
            facts_path=args.facts,
            model_path=args.model,
            out_path=args.out,
            dataset_name=args.dataset_name,
            per_label=args.per_label,
            min_abs_delta=args.min_abs_delta,
        )
        print(f"Wrote {n} counterfactual QA instances to {args.out}")
    else:
        n = build_action_qa_dataset(
            facts_path=args.facts,
            model_path=args.model,
            out_path=args.out,
            dataset_name=args.dataset_name,
            per_label=args.per_label,
            threshold=args.threshold,
        )
        print(f"Wrote {n} action recommendation QA instances to {args.out}")
