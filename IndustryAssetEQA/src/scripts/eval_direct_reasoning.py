#!/usr/bin/env python3
"""
Task-aware direct + reasoning evaluator.

Saves CSV with columns indicating direct match and reasoning checks.

Usage:
python -m src.scripts.eval_direct_reasoning --gold gold.jsonl --preds preds.jsonl --out results_direct_reasoning.csv --start 0 --end 100 --offline
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
from typing import Optional, Dict, Any, List, Set
from collections import defaultdict
from tqdm import tqdm

# Try to import NLI utilities (optional)
try:
    from src.utils.nli_checks import (
        entailment_pass_rate,
        reasoning_answer_alignment,
        extract_conclusion_sentence,
    )
    _HAVE_NLI = True
except Exception:
    _HAVE_NLI = False

# Tunable thresholds
ENTAILMENT_PASS_THRESH = 0.3   # used when NLI available
TOKEN_OVERLAP_THRESH = 0.3     # fallback token Jaccard threshold
CONCLUSION_ALIGN_THRESH = 0.30  # used for conclusion->direct entailment

# ---------------- helpers ----------------
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

def sentence_splitter(text: str) -> List[str]:
    sents = re.split(r'(?<=[\.\?\!])\s+', (text or "").strip())
    return [s.strip() for s in sents if s.strip()]

def extract_numeric_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    decimals = re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", text)
    if decimals:
        try:
            return float(decimals[-1])
        except:
            return None
    ints = re.findall(r"-?\d+(?:[eE][-+]?\d+)?", text)
    if ints:
        try:
            return float(ints[-1])
        except:
            return None
    return None

def numeric_equal(a: Optional[float], b: Optional[float], rel_tol: float = 0.05, abs_tol: float = 1e-3) -> bool:
    if a is None or b is None:
        return False
    if abs(a - b) <= abs_tol:
        return True
    return (abs(a - b) / max(abs(b), 1e-12)) <= rel_tol

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def infer_direction_from_text(t: str) -> Optional[str]:
    if not t:
        return None
    s = t.lower()
    if re.search(r"\b(decreas|reduc|lower|drop|declin)\b", s):
        return "decrease"
    if re.search(r"\b(increas|higher|raise|rise)\b", s):
        return "increase"
    if re.search(r"\b(no change|unchang|remain|unchanged|no_change)\b", s):
        return "no_change"
    return None

def token_set(s: str) -> Set[str]:
    s = normalize_text(s)
    toks = re.findall(r"\w+", s)
    return set(toks)

def jaccard_overlap(a: str, b: str) -> float:
    A = token_set(a)
    B = token_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = A.intersection(B)
    union = A.union(B)
    return len(inter) / len(union)

# ---------------- evaluation ----------------
def evaluate_pair(gold: Dict[str,Any], pred: Dict[str,Any], use_nli: bool = True) -> Dict[str,Any]:
    qa_id = gold.get("qa_id")
    result: Dict[str,Any] = {"qa_id": qa_id}

    gold_direct = (gold.get("direct_answer") or "").strip()
    gold_reasoning = (gold.get("reasoning_answer") or "").strip()
    gold_task = (gold.get("task_type") or "").lower()

    pred_direct = (pred.get("answer",{}).get("direct_answer") or "").strip()
    pred_reasoning = (pred.get("answer",{}).get("reasoning_answer") or "").strip()

    # Structure check: for CF accept direct or structured block
    if gold_task == "counterfactual":
        gold_cf_exists = bool(gold.get("counterfactual"))
        pred_cf_exists = bool(pred.get("answer",{}).get("counterfactual") or pred.get("counterfactual"))
        structure_ok = bool(gold_direct or gold_cf_exists) and (bool(pred_direct) or pred_cf_exists)
    else:
        structure_ok = bool(gold_direct and gold_reasoning and pred_direct and pred_reasoning)
    result["structure_ok"] = structure_ok

    # numeric extraction
    gold_num = None
    if gold.get("label"):
        try:
            gold_num = float(gold["label"])
        except:
            gold_num = None
    if gold_num is None:
        gold_num = extract_numeric_from_text(gold_direct)
    model_num = extract_numeric_from_text(pred_direct)

    # --- Direct match (task aware, CF special-case)
    direct_match = False
    if gold_task == "counterfactual":
        cf = gold.get("counterfactual", {}) or {}
        gold_dir = cf.get("direction")
        gold_delta = None
        if "delta_risk" in cf:
            try:
                gold_delta = float(cf["delta_risk"])
            except:
                gold_delta = None
        elif "risk_before" in cf and "risk_after" in cf:
            try:
                gold_delta = float(cf.get("risk_after")) - float(cf.get("risk_before"))
            except:
                gold_delta = None

        pred_cf = pred.get("answer", {}).get("counterfactual") or pred.get("counterfactual") or {}
        pred_dir = pred_cf.get("direction")
        pred_delta = None
        if "delta_risk" in pred_cf:
            try:
                pred_delta = float(pred_cf["delta_risk"])
            except:
                pred_delta = None
        elif "risk_before" in pred_cf and "risk_after" in pred_cf:
            try:
                pred_delta = float(pred_cf.get("risk_after")) - float(pred_cf.get("risk_before"))
            except:
                pred_delta = None

        if pred_dir is None:
            pred_dir = infer_direction_from_text(pred_direct) or infer_direction_from_text(pred_reasoning)
        if gold_dir is None and gold_delta is not None:
            gold_dir = "decrease" if gold_delta < 0 else ("increase" if gold_delta > 0 else "no_change")
        if pred_dir is None and pred_delta is not None:
            pred_dir = "decrease" if pred_delta < 0 else ("increase" if pred_delta > 0 else "no_change")

        if gold_dir and pred_dir:
            direct_match = (gold_dir == pred_dir)
        elif (gold_delta is not None) and (pred_delta is not None):
            direct_match = numeric_equal(gold_delta, pred_delta)
        else:
            direct_match = normalize_text(gold_direct) == normalize_text(pred_direct)

        result["gold_cf_direction"] = gold_dir
        result["pred_cf_direction"] = pred_dir
        result["gold_delta_risk"] = gold_delta
        result["pred_delta_risk"] = pred_delta

    else:
        if gold_num is not None and model_num is not None:
            direct_match = numeric_equal(gold_num, model_num)
        else:
            direct_match = normalize_text(gold_direct) == normalize_text(pred_direct)

    result["direct_match"] = direct_match
    result["gold_num"] = gold_num
    result["model_num"] = model_num

    # --- Reasoning checks: two forms
    # 1) semantic entailment (if NLI available)
    # 2) token-overlap fallback
    reasoning_ok = False
    reasoning_entail_gold_to_model = None
    reasoning_entail_model_to_gold = None
    reasoning_overlap = None
    conclusion_entails_gold_direct_bool = None
    conclusion_entails_gold_direct_prob = None

    gold_sents = sentence_splitter(gold_reasoning)
    pred_sents = sentence_splitter(pred_reasoning)

    if (use_nli and _HAVE_NLI) and (gold_sents or pred_sents):
        # compute mutual entailment rates
        reasoning_entail_gold_to_model = entailment_pass_rate(gold_reasoning, pred_sents) if pred_sents else 0.0
        reasoning_entail_model_to_gold = entailment_pass_rate(pred_reasoning, gold_sents) if gold_sents else 0.0

        # conclusion alignment (model conclusion -> gold direct)
        aligned, prob = reasoning_answer_alignment(pred_reasoning, gold_direct)
        conclusion_entails_gold_direct_bool = aligned
        conclusion_entails_gold_direct_prob = prob

        # decide: require both directions >= threshold AND conclusion alignment
        if (reasoning_entail_gold_to_model >= ENTAILMENT_PASS_THRESH and
                reasoning_entail_model_to_gold >= ENTAILMENT_PASS_THRESH):
            reasoning_ok = True
        else:
            reasoning_ok = False

    else:
        # fallback lexical overlap:
        # compare whole reasoning texts using Jaccard token overlap
        reasoning_overlap = jaccard_overlap(gold_reasoning, pred_reasoning)
        reasoning_ok = (reasoning_overlap >= TOKEN_OVERLAP_THRESH)
        # difficulty: set conclusion fields to None (no NLI)
        conclusion_entails_gold_direct_bool = None
        conclusion_entails_gold_direct_prob = None

    # export reasoning results
    result["reasoning_ok"] = reasoning_ok
    result["reasoning_entail_gold_to_model"] = reasoning_entail_gold_to_model
    result["reasoning_entail_model_to_gold"] = reasoning_entail_model_to_gold
    result["reasoning_overlap"] = reasoning_overlap
    result["conclusion_entails_gold_direct_bool"] = conclusion_entails_gold_direct_bool
    result["conclusion_entails_gold_direct_prob"] = conclusion_entails_gold_direct_prob

    return result

# ---------------- batch runner ----------------
def run(gold_path: str, preds_path: str, out_csv: str,
        start: int = 0, end: Optional[int] = None,
        offline: bool = False):

    gold_idx = load_jsonl_index(gold_path)
    pred_idx = load_jsonl_index(preds_path)

    keys = list(gold_idx.keys())
    total = len(keys)

    start = max(0, start)
    end = total if end is None else min(total, end)

    if start >= end:
        print("Invalid slice. Nothing to evaluate.")
        return

    print(f"Evaluating indices [{start}:{end}] out of {total}")

    rows = []
    stats = defaultdict(int)
    missing = []

    slice_keys = keys[start:end]

    for qa_id in tqdm(slice_keys, desc="Evaluating", unit="qa"):
        gold = gold_idx[qa_id]
        pred = pred_idx.get(qa_id)
        if pred is None:
            stats["missing_pred"] += 1
            missing.append(qa_id)
            continue

        res = evaluate_pair(gold, pred, use_nli=not offline)
        rows.append(res)

        stats["count"] += 1
        stats["structure_ok"] += int(bool(res.get("structure_ok")))
        stats["direct_match"] += int(bool(res.get("direct_match")))
        stats["reasoning_ok"] += int(bool(res.get("reasoning_ok")))

    # write CSV (ensure stable columns)
    fieldnames = []
    if rows:
        # gather unique keys in deterministic order
        keys_order = ["qa_id", "structure_ok", "direct_match", "gold_num", "model_num",
                      "reasoning_ok", "reasoning_entail_gold_to_model", "reasoning_entail_model_to_gold",
                      "reasoning_overlap", "conclusion_entails_gold_direct_bool", "conclusion_entails_gold_direct_prob"]
        # add any extra keys present
        extra = [k for k in rows[0].keys() if k not in keys_order]
        fieldnames = keys_order + extra
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if stats["count"] > 0:
        print("\nSummary:")
        print("Structure OK:", stats["structure_ok"] / stats["count"])
        print("Direct match:", stats["direct_match"] / stats["count"])
        print("Reasoning OK:", stats["reasoning_ok"] / stats["count"])
    if stats.get("missing_pred"):
        print(f"Missing predictions for {stats['missing_pred']} gold items. Example missing ids: {missing[:5]}")

    print(f"\nSaved results to {out_csv}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", default="results_direct_reasoning.csv")
    parser.add_argument("--offline", action="store_true", help="disable NLI entailment checks")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    run(
        gold_path=args.gold,
        preds_path=args.preds,
        out_csv=args.out,
        start=args.start,
        end=args.end,
        offline=args.offline
    )
