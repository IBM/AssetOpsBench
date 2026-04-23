#!/usr/bin/env python3
import sys
import os

# ensure project root is on sys.path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import argparse
import json
import re
import csv
from typing import Optional, Dict, Any, List
from collections import defaultdict
from tqdm import tqdm

try:
    from src.utils.nli_checks import (
        entailment_pass_rate,
        reasoning_answer_alignment,
        extract_conclusion_sentence,
    )
    _HAVE_NLI = True
except Exception:
    _HAVE_NLI = False


# ----------------- helpers -----------------

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
        return float(decimals[-1])
    ints = re.findall(r"-?\d+(?:[eE][-+]?\d+)?", text)
    if ints:
        return float(ints[-1])
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


# ----------------- evaluation -----------------

def evaluate_pair(gold: Dict[str,Any], pred: Dict[str,Any], use_nli: bool = True) -> Dict[str,Any]:

    qa_id = gold.get("qa_id")
    result = {"qa_id": qa_id}

    gold_direct = (gold.get("direct_answer") or "").strip()
    gold_reasoning = (gold.get("reasoning_answer") or "").strip()

    pred_direct = (pred.get("answer",{}).get("direct_answer") or "").strip()
    pred_reasoning = (pred.get("answer",{}).get("reasoning_answer") or "").strip()

    # Structure check
    structure_ok = bool(gold_direct and gold_reasoning and pred_direct and pred_reasoning)
    result["structure_ok"] = structure_ok

    # Direct answer match (numeric aware)
    gold_num = None
    if gold.get("label"):
        try:
            gold_num = float(gold["label"])
        except:
            gold_num = None
    if gold_num is None:
        gold_num = extract_numeric_from_text(gold_direct)

    model_num = extract_numeric_from_text(pred_direct)

    if gold_num is not None and model_num is not None:
        direct_match = numeric_equal(gold_num, model_num)
    else:
        direct_match = normalize_text(gold_direct) == normalize_text(pred_direct)

    result["direct_match"] = direct_match
    result["gold_num"] = gold_num
    result["model_num"] = model_num

    # Reasoning semantic checks (optional NLI)
    if use_nli and _HAVE_NLI:
        pred_sents = sentence_splitter(pred_reasoning)
        gold_sents = sentence_splitter(gold_reasoning)

        if pred_sents:
            result["reasoning_entail_gold_to_model"] = entailment_pass_rate(gold_reasoning, pred_sents)
        else:
            result["reasoning_entail_gold_to_model"] = 0.0

        if gold_sents:
            result["reasoning_entail_model_to_gold"] = entailment_pass_rate(pred_reasoning, gold_sents)
        else:
            result["reasoning_entail_model_to_gold"] = 0.0

        aligned, prob = reasoning_answer_alignment(pred_reasoning, gold_direct)
        result["conclusion_entails_gold_direct_bool"] = aligned
        result["conclusion_entails_gold_direct_prob"] = prob
    else:
        result["reasoning_entail_gold_to_model"] = None
        result["reasoning_entail_model_to_gold"] = None
        result["conclusion_entails_gold_direct_bool"] = None
        result["conclusion_entails_gold_direct_prob"] = None

    return result


# ----------------- batch runner -----------------

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

    slice_keys = keys[start:end]

    for qa_id in tqdm(slice_keys, desc="Evaluating", unit="qa"):
        gold = gold_idx[qa_id]
        pred = pred_idx.get(qa_id)
        if pred is None:
            stats["missing_pred"] += 1
            continue

        res = evaluate_pair(gold, pred, use_nli=not offline)
        rows.append(res)

        stats["count"] += 1
        stats["structure_ok"] += int(res["structure_ok"])
        stats["direct_match"] += int(res["direct_match"])

    # write CSV
    fieldnames = rows[0].keys() if rows else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if stats["count"] > 0:
        print("\nSummary:")
        print("Structure OK:", stats["structure_ok"] / stats["count"])
        print("Direct match:", stats["direct_match"] / stats["count"])

    print(f"\nSaved results to {out_csv}")


# ----------------- CLI -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", default="results_direct_reasoning.csv")
    parser.add_argument("--offline", action="store_true")
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
