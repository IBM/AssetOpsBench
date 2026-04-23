# src/scripts/eval_single_instance.py
import sys
import os
# ensure project root is on sys.path so "import src..." works
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import json
import re

from src.utils.nli_checks import (
    entailment_pass_rate,
    reasoning_answer_alignment,
    extract_conclusion_sentence,
)

# ---------- Example inputs (replace with reading JSONL or args) ----------
gold = {
"qa_id": "pdm_ts_action_pdm_m56_comp3_2015-01-02T03", "fact_id": "pdm_m56_comp3_2015-01-02T03", "task_type": "action_recommendation", "question": "For rotating_equipment machine_56 in the time window 2015-01-01 03:00:00 to 2015-01-02 03:00:00, should a maintenance work order be opened now, or is it acceptable to continue monitoring?", "direct_answer": "A maintenance work order should be opened now.", "reasoning_answer": "The learned risk model estimates probability of any failure ≈ 1.00 (threshold=0.50). Failure severity: high. Recommended diagnostics/actions: check discharge and downstream valves for incorrect positions or sticking; inspect filters, strainers, and lines for blockage; review process conditions and control setpoints. Most informative sensors: pressure. Equipment class/type: electric_motor_driven_rotating_machine.", "provenance": {"fact_id": "pdm_m56_comp3_2015-01-02T03", "features": ["volt_mean", "volt_std", "volt_min", "volt_max", "volt_trend", "rotate_mean", "rotate_std", "rotate_min", "rotate_max", "rotate_trend", "pressure_mean", "pressure_std", "pressure_min", "pressure_max", "pressure_trend", "vibration_mean", "vibration_std", "vibration_min", "vibration_max", "vibration_trend", "error_count_last_window", "distinct_error_types_last_window", "hours_since_last_maint_comp3", "machine_age", "model_model1"], "file": "PdM_telemetry.csv", "row": 0, "telemetry_points_in_window": 22, "errors_in_window": 0, "failure_profile_id": "comp3", "asset_profile_brief": {"equipment_category": "rotating_equipment", "equipment_class_type": "electric_motor_driven_rotating_machine", "unit_subunit": ["process_path", "discharge_line", "valves"], "asset_name": "model1"}, "failure_profile_brief": {"failure_mode": "comp3", "display_name": "comp3", "short_description": "Restriction, valve malfunction, or process-side blockage causing sustained high discharge pressure while rotational speed and vibration remain near normal.", "associated_sensors": ["pressure"], "typical_indicators": {"pressure_mean": "significantly above healthy baseline", "pressure_trend": "often positive or sustained high", "rotate_mean": "near nominal", "vibration_mean": "near nominal or slightly increased"}, "recommended_actions": ["check discharge and downstream valves for incorrect positions or sticking", "inspect filters, strainers, and lines for blockage", "review process conditions and control setpoints"], "severity": "high"}}, "label": "open_work_order", "asset_id": "machine_56", "confidence_estimator": 1.0, "risk": 1.0, "probs_before": {"comp1": 5.666517784187859e-26, "comp2": 1.6855443558067901e-53, "comp3": 1.0, "comp4": 4.746799049527151e-27, "healthy": 1.1467891488958534e-43}, "asset_profile_brief": {"equipment_category": "rotating_equipment", "equipment_class_type": "electric_motor_driven_rotating_machine", "unit_subunit": ["process_path", "discharge_line", "valves"], "asset_name": "model1"}, "failure_profile_brief": {"failure_mode": "comp3", "display_name": "comp3", "short_description": "Restriction, valve malfunction, or process-side blockage causing sustained high discharge pressure while rotational speed and vibration remain near normal.", "associated_sensors": ["pressure"], "typical_indicators": {"pressure_mean": "significantly above healthy baseline", "pressure_trend": "often positive or sustained high", "rotate_mean": "near nominal", "vibration_mean": "near nominal or slightly increased"}, "recommended_actions": ["check discharge and downstream valves for incorrect positions or sticking", "inspect filters, strainers, and lines for blockage", "review process conditions and control setpoints"], "severity": "high"}}



model_out = {
    "qa_id": "pdm_ts_action_pdm_m56_comp3_2015-01-02T03",
    "answer": {"direct_answer": "It is acceptable to continue monitoring the machine at this time; immediate maintenance work order is not necessary.", "reasoning_answer": "The diagnostic features for machine_56 show stable and moderate values without alarming trends. The volt_mean is 169.06 with a small positive volt_trend of 0.22, indicating no sudden voltage drop or spike. The rotate_mean is 448.35 with a rotate_trend of 0.48, showing a slight increase but within a reasonable range given the rotate_std of 46.33. Pressure values are stable with a mean of 124.67 and a minimal pressure_trend of 0.007. Vibration_mean is 39.26 with a small positive vibration_trend of 0.05, which does not indicate excessive wear or imminent failure. Overall, no feature shows critical deviation or rapid deterioration that would justify opening a maintenance work order immediately.", "provenance": {"fact_id": "pdm_m56_comp3_2015-01-02T03", "features": ["volt_mean", "volt_trend", "rotate_mean", "rotate_trend", "pressure_mean", "pressure_trend", "vibration_mean", "vibration_trend"], "file": "PdM_telemetry.csv", "row": 0},
               "confidence": 0.85}


}
# -----------------------------------------------------------------------

# ---------- helpers ----------
def extract_numeric_from_direct_answer(text: str):
    """
    Extract the most-likely numeric answer from a direct_answer string.
    Prefer decimals (pick last decimal), otherwise pick last integer.
    """
    if not text:
        return None
    # decimals first
    decimals = re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", text)
    if decimals:
        try:
            return float(decimals[-1])
        except:
            pass
    # fallback: integers (pick last)
    ints = re.findall(r"-?\d+(?:[eE][-+]?\d+)?", text)
    if ints:
        try:
            return float(ints[-1])
        except:
            pass
    return None

def sentence_splitter(text):
    sents = re.split(r'(?<=[\.\?\!])\s+', (text or "").strip())
    return [s.strip() for s in sents if s.strip()]

def premise_from_gold_with_features(gold):
    """
    Build a compact premise that includes explicit feature=value pairs (if present)
    so the NLI model can see numeric evidence.
    """
    parts = []
    parts.append(f"fact_id={gold.get('fact_id')}")
    parts.append(f"asset_id={gold.get('asset_id')}")
    if gold.get("label") is not None:
        parts.append(f"label={gold.get('label')}")
    prov = gold.get("provenance", {}) or {}
    if prov.get("file"):
        parts.append(f"file={prov.get('file')}")
    if prov.get("row") is not None:
        parts.append(f"row={prov.get('row')}")
    # try to extract numeric value for listed features from reasoning or label
    reasoning = gold.get("reasoning_answer","") or ""
    for feat in prov.get("features", []):
        # look for "<feat> ... <number>" in reasoning
        m = re.search(rf"{re.escape(feat)}[^0-9\-]*(-?\d+\.\d+|-?\d+)", reasoning)
        if m:
            parts.append(f"{feat}={m.group(1)}")
    # short sensor description
    if prov.get("sensor_description"):
        parts.append(f"sensor_desc: {prov.get('sensor_description')[:200]}")
    return " ; ".join(parts)

# ---------- Build premise and inputs for NLI ----------
premise = premise_from_gold_with_features(gold)
reasoning_text = model_out["answer"]["reasoning_answer"]
reasoning_sents = sentence_splitter(reasoning_text)
direct_answer_text = model_out["answer"]["direct_answer"]

print("Premise (short):", premise)
print("Reasoning sentences:", reasoning_sents)
print("Direct answer text:", direct_answer_text)
print("---- Running NLI checks ----")

# ---------- 1) Entailment pass rate (evidence -> reasoning sentences) ----------
entail_rate = entailment_pass_rate(premise, reasoning_sents)
print("Entailment pass rate:", entail_rate)

# ---------- 2) Reasoning -> direct answer alignment ----------
conclusion = extract_conclusion_sentence(reasoning_text)
aligned, entail_prob = reasoning_answer_alignment(reasoning_text, direct_answer_text)
print("Conclusion sentence:", conclusion)
print("Aligned:", aligned, "entail_prob:", entail_prob)

# ---------- 3) Numeric check for descriptive tasks (recommended) ----------
gold_num = None
if gold.get("label"):
    try:
        gold_num = float(gold["label"])
    except Exception:
        gold_num = None
if gold_num is None:
    gold_num = extract_numeric_from_direct_answer(gold.get("direct_answer",""))

model_num = extract_numeric_from_direct_answer(direct_answer_text)

if gold_num is not None and model_num is not None:
    rel_err = abs(model_num - gold_num) / (abs(gold_num) + 1e-12)
    within_tol = (rel_err <= 0.05) or (abs(model_num - gold_num) <= 1e-3)
    print(f"Gold={gold_num:.6f}, Model={model_num:.6f}, rel_err={rel_err:.6f}, within_tol={within_tol}")
else:
    print("Numeric value not found in gold or model direct answer; fallback to label_consistency or string match.")
    within_tol = False

# ---------- 4) Decide final judgments for this instance ----------
structure_ok = True  # assume parsing handled upstream
provenance_ok = (model_out["answer"]["provenance"].get("fact_id") == gold.get("fact_id"))
label_consistent = within_tol if (gold_num is not None and model_num is not None) else (gold.get("label") == str(model_num) if model_num is not None else False)

print("Provenance ok:", provenance_ok)
print("Label consistent (numeric tolerance):", label_consistent)
