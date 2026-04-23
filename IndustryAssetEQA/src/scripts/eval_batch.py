# src/scripts/eval_batch.py
"""
Batch evaluator:
- Reads gold.jsonl and model.jsonl (qa_id must match)
- Task-aware evaluation: descriptive uses numeric tolerance as primary
- Optional NLI checks (default enabled if transformers/torch available)
- Writes results.csv and prints aggregates
"""
import sys
import os
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import argparse
import json
import re
import csv
from typing import Dict, Any, List, Optional
from collections import defaultdict
from tqdm import tqdm

# NLI utilities (may download model on first run)
from src.utils.nli_checks import (
    entailment_pass_rate,
    reasoning_answer_alignment,
    extract_conclusion_sentence,
)

# ----------------- helpers -----------------
def extract_numeric_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    decimals = re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", text)
    if decimals:
        try:
            return float(decimals[-1])
        except:
            pass
    ints = re.findall(r"-?\d+(?:[eE][-+]?\d+)?", text)
    if ints:
        try:
            return float(ints[-1])
        except:
            pass
    return None

def sentence_splitter(text: str) -> List[str]:
    sents = re.split(r'(?<=[\.\?\!])\s+', (text or "").strip())
    return [s.strip() for s in sents if s.strip()]

def premise_from_gold_with_features(gold: Dict[str,Any]) -> str:
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
    reasoning = gold.get("reasoning_answer","") or ""
    for feat in prov.get("features", []):
        m = re.search(rf"{re.escape(feat)}[^0-9\-]*(-?\d+\.\d+|-?\d+)", reasoning)
        if m:
            parts.append(f"{feat}={m.group(1)}")
    if prov.get("sensor_description"):
        parts.append(f"sensor_desc: {prov.get('sensor_description')[:200]}")
    return " ; ".join(parts)

def numeric_equal(a: Optional[float], b: Optional[float], rel_tol: float = 0.05, abs_tol: float = 1e-3) -> bool:
    if a is None or b is None:
        return False
    if abs(a - b) <= abs_tol:
        return True
    return (abs(a - b) / max(abs(b), 1e-12)) <= rel_tol

def load_jsonl_index(path: str) -> Dict[str, Dict]:
    idx = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("qa_id") or obj.get("id")
            if qid:
                idx[qid] = obj
    return idx

# ----------------- per-instance evaluation -----------------
def evaluate_instance(gold: Dict[str,Any], model_out: Dict[str,Any], nli_online: bool = True) -> Dict[str,Any]:
    qa_id = gold.get("qa_id")
    out = {"qa_id": qa_id}

    ans = model_out.get("answer", {}) if model_out else {}
    structure_ok = all(k in ans for k in ("direct_answer","reasoning_answer","provenance","confidence"))
    out["structure_ok"] = bool(structure_ok)

    prov_model = ans.get("provenance", {}) if ans else {}
    provenance_ok = prov_model.get("fact_id") == gold.get("fact_id")
    out["provenance_ok"] = bool(provenance_ok)

    task_type = (gold.get("task_type") or "").lower()

    # extract numeric gold/model values
    gold_num = None
    if gold.get("label"):
        try:
            gold_num = float(gold["label"])
        except:
            gold_num = None
    if gold_num is None:
        gold_num = extract_numeric_from_text(gold.get("direct_answer","") or "")

    model_direct = ans.get("direct_answer","") if ans else ""
    model_num = extract_numeric_from_text(model_direct)

    # prepare premise
    premise = premise_from_gold_with_features(gold)

    entail_rate = None
    alignment_prob = None
    alignment_bool = None
    label_consistent = False

    if task_type == "descriptive":
        # primary numeric check
        if gold_num is not None and model_num is not None:
            label_consistent = numeric_equal(model_num, gold_num)
        else:
            def norm(s): return re.sub(r"\s+"," ", (s or "").strip().lower())
            label_consistent = (norm(gold.get("direct_answer","")) == norm(model_direct))
        # optional NLI diagnostics
        if nli_online:
            sents = sentence_splitter(ans.get("reasoning_answer","") if ans else "")
            entail_rate = entailment_pass_rate(premise, sents)
            concl = extract_conclusion_sentence(ans.get("reasoning_answer","") or "")
            concl_num = extract_numeric_from_text(concl)
            if concl_num is not None and model_num is not None:
                alignment_bool = numeric_equal(concl_num, model_num)
                alignment_prob = 1.0 if alignment_bool else 0.0
            else:
                aligned, p = reasoning_answer_alignment(ans.get("reasoning_answer","") or "", model_direct)
                alignment_bool = bool(aligned)
                alignment_prob = float(p)
    else:
        # diagnostic / counterfactual / action / temporal
        if task_type == "diagnostic":
            def normalize_label(s): return re.sub(r"[_\-\s]+"," ", (s or "").strip().lower())
            gold_label = normalize_label(gold.get("label") or gold.get("direct_answer",""))
            model_label = normalize_label(model_direct)
            label_consistent = (gold_label in model_label) or (model_label in gold_label)
        else:
            # fallback text-match
            def norm(s): return re.sub(r"\s+"," ", (s or "").strip().lower())
            label_consistent = (norm(gold.get("direct_answer","")) == norm(model_direct))
        if nli_online:
            sents = sentence_splitter(ans.get("reasoning_answer","") if ans else "")
            entail_rate = entailment_pass_rate(premise, sents)
            aligned, p = reasoning_answer_alignment(ans.get("reasoning_answer","") or "", model_direct)
            alignment_bool = bool(aligned)
            alignment_prob = float(p)

    out.update({
        "label_consistent": bool(label_consistent),
        "entailment_pass_rate": (None if entail_rate is None else float(entail_rate)),
        "reasoning_alignment_bool": (None if alignment_bool is None else bool(alignment_bool)),
        "reasoning_alignment_prob": (None if alignment_prob is None else float(alignment_prob)),
        "gold_num": (None if gold_num is None else float(gold_num)),
        "model_num": (None if model_num is None else float(model_num)),
    })

    out["full_pass"] = bool(structure_ok and provenance_ok and label_consistent)
    return out

# ----------------- batch runner -----------------
def run_batch(gold_path: str, model_path: str, out_csv: str = "results.csv", nli_online: bool = True):
    gold_idx = load_jsonl_index(gold_path)
    model_idx = load_jsonl_index(model_path)

    rows = []
    stats = defaultdict(int)
    entail_rates = []
    alignment_probs = []

    keys = list(gold_idx.keys())
    for qa_id in tqdm(keys, desc="Evaluating", unit="qa"):
        gold = gold_idx[qa_id]
        model = model_idx.get(qa_id)
        if model is None:
            stats["missing_model"] += 1
            continue
        res = evaluate_instance(gold, model, nli_online=nli_online)
        rows.append(res)
        stats["count"] += 1
        stats["structure_ok"] += int(res["structure_ok"])
        stats["provenance_ok"] += int(res["provenance_ok"])
        stats["label_consistent"] += int(res["label_consistent"])
        stats["full_pass"] += int(res["full_pass"])
        if res["entailment_pass_rate"] is not None:
            entail_rates.append(res["entailment_pass_rate"])
        if res["reasoning_alignment_prob"] is not None:
            alignment_probs.append(res["reasoning_alignment_prob"])

    # write CSV
    fieldnames = [
        "qa_id","structure_ok","provenance_ok","label_consistent","entailment_pass_rate",
        "reasoning_alignment_bool","reasoning_alignment_prob","gold_num","model_num","full_pass"
    ]
    with open(out_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    n = stats["count"]
    print(f"Processed {n} matched QA instances (model missing for {stats['missing_model']} gold rows).")
    if n == 0:
        return
    def frac(k): return stats[k] / n
    print(f"Structure OK: {frac('structure_ok'):.3f}, Provenance OK: {frac('provenance_ok'):.3f}, Label Consistent: {frac('label_consistent'):.3f}, Full Pass: {frac('full_pass'):.3f}")
    if entail_rates:
        print(f"Mean entailment pass rate (NLI): {sum(entail_rates)/len(entail_rates):.3f}")
    if alignment_probs:
        print(f"Mean reasoning->answer entailment prob (NLI): {sum(alignment_probs)/len(alignment_probs):.3f}")
    print(f"Wrote per-instance results to {out_csv}")

# ----------------- CLI -----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch evaluator: gold JSONL + model JSONL -> CSV + summary (supports slicing)")
    p.add_argument("--gold", required=True, help="Gold JSONL (SME-validated) file path")
    p.add_argument("--model", required=True, help="Model outputs JSONL file path")
    p.add_argument("--out", default="results.csv", help="Output CSV path")
    p.add_argument("--offline", action="store_true", help="Skip NLI checks (useful if HF model not available)")
    p.add_argument("--start", type=int, default=0, help="Start index (inclusive) into ordered gold list (0-based)")
    p.add_argument("--end", type=int, default=None, help="End index (exclusive) into ordered gold list (0-based). If omitted, runs to end.")
    args = p.parse_args()

    # Load indices once
    gold_idx = load_jsonl_index(args.gold)
    model_idx = load_jsonl_index(args.model)
    keys = list(gold_idx.keys())

    # normalize start/end
    start = max(0, args.start or 0)
    end = args.end if args.end is not None else len(keys)
    end = min(len(keys), end)
    if start >= end:
        print(f"Empty slice: start={start} end={end} (no items). Exiting.")
        raise SystemExit(0)

    print(f"Evaluating gold[{start}:{end}] -> {end-start} QA instances (indices into ordered gold file).")
    # run the batch on the slice
    # we reuse run_batch internals but process only keys[start:end]
    # small local runner to avoid duplicating file writes
    rows = []
    stats = defaultdict(int)
    entail_rates = []
    alignment_probs = []

    slice_keys = keys[start:end]
    for qa_id in tqdm(slice_keys, desc="Evaluating", unit="qa"):
        gold = gold_idx[qa_id]
        model = model_idx.get(qa_id)
        if model is None:
            stats["missing_model"] += 1
            continue
        res = evaluate_instance(gold, model, nli_online=(not args.offline))
        rows.append(res)
        stats["count"] += 1
        stats["structure_ok"] += int(res["structure_ok"])
        stats["provenance_ok"] += int(res["provenance_ok"])
        stats["label_consistent"] += int(res["label_consistent"])
        stats["full_pass"] += int(res["full_pass"])
        if res["entailment_pass_rate"] is not None:
            entail_rates.append(res["entailment_pass_rate"])
        if res["reasoning_alignment_prob"] is not None:
            alignment_probs.append(res["reasoning_alignment_prob"])

    # write CSV (append if file exists so you can run multiple chunks)
    header = [
        "qa_id","structure_ok","provenance_ok","label_consistent","entailment_pass_rate",
        "reasoning_alignment_bool","reasoning_alignment_prob","gold_num","model_num","full_pass"
    ]
    write_mode = "a" if (os.path.exists(args.out) and os.path.getsize(args.out) > 0) else "w"
    with open(args.out, write_mode, newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=header)
        if write_mode == "w":
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in header})

    n = stats["count"]
    print(f"Processed {n} matched QA instances in slice (model missing for {stats['missing_model']} gold rows).")
    if n > 0:
        def frac(k): return stats[k] / n
        print(f"Structure OK: {frac('structure_ok'):.3f}, Provenance OK: {frac('provenance_ok'):.3f}, Label Consistent: {frac('label_consistent'):.3f}, Full Pass: {frac('full_pass'):.3f}")
    if entail_rates:
        print(f"Mean entailment pass rate (NLI): {sum(entail_rates)/len(entail_rates):.3f}")
    if alignment_probs:
        print(f"Mean reasoning->answer entailment prob (NLI): {sum(alignment_probs)/len(alignment_probs):.3f}")
    print(f"Wrote per-instance results (appended) to {args.out}")

