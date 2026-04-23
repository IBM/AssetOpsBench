# src/utils/semantic_verifier.py
"""
Semantic verifier utilities.

Provides:
  - feature_presence_check(answer, fact, min_features=1)
  - numeric_answer_check(answer, gold_qa, fact, rel_tol=0.05, abs_tol=1e-3)
  - embed_similarity_check(answer_text, gold_text, model_name='all-MiniLM-L6-v2', threshold=0.7)

Embedding-based similarity uses sentence-transformers if available; otherwise it returns None.
"""
from __future__ import annotations
import re
import math
from typing import Dict, Any, List, Optional, Tuple

# Try optional import for embeddings
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _EMBED_AVAILABLE = True
except Exception:
    _EMBED_AVAILABLE = False
    SentenceTransformer = None
    st_util = None


_num_regex = re.compile(r"[-+]?\d*\.\d+|\d+")


def _extract_numbers_from_text(text: str) -> List[float]:
    if not isinstance(text, str):
        return []
    return [float(m.group(0)) for m in _num_regex.finditer(text)]


def feature_presence_check(
    answer: Dict[str, Any],
    fact: Dict[str, Any],
    min_features: int = 1,
) -> Dict[str, Any]:
    """
    Check if features cited in provenance appear in the textual answer.

    Returns dict:
      {
        "ok": bool,            # whether threshold passed
        "n_features_cited": int,
        "n_features_present_in_text": int,
        "missing_features": [ ... ],
      }
    Behavior:
      - We look for feature names (exact token) OR flattened tokens (e.g., 'vibration_mean' -> 'vibration mean')
      - We search both direct_answer and reasoning_answer text fields.
    """
    prov = answer.get("provenance", {}) or {}
    feats = prov.get("features", []) or []
    text = ""
    da = answer.get("direct_answer") or ""
    ra = answer.get("reasoning_answer") or ""
    text = f"{da}\n{ra}".lower()

    found = []
    missing = []
    for f in feats:
        if not f or not isinstance(f, str):
            continue
        f_norm = f.lower()
        # direct presence
        if f_norm in text:
            found.append(f)
            continue
        # try whitespace variant
        f_ws = f_norm.replace("_", " ")
        if f_ws in text:
            found.append(f)
            continue
        # try hyphen variant
        if f_norm.replace("_", "-") in text:
            found.append(f)
            continue
        # not found
        missing.append(f)

    n_present = len(found)
    n_total = len(feats)
    ok = n_present >= min_features if n_total > 0 else True

    return {
        "ok": ok,
        "n_features_cited": n_total,
        "n_features_present_in_text": n_present,
        "missing_features": missing,
    }


def numeric_answer_check(
    answer: Dict[str, Any],
    gold_qa: Dict[str, Any],
    fact: Dict[str, Any],
    rel_tol: float = 0.05,
    abs_tol: float = 1e-3,
) -> Dict[str, Any]:
    """
    For descriptive / numeric tasks, try to verify that the numeric direct_answer
    matches the gold label (if gold label is numeric) or that numbers mentioned in
    reasoning_answer match the fact's features.

    Returns:
      {
        "ok": bool or None,            # None if not applicable
        "gold_numeric": Optional[float],
        "pred_numbers": [ ... ],
        "closest_pred": Optional[float],
        "error_abs": Optional[float],
        "error_rel": Optional[float],
        "within_tolerance": bool or None,
        "note": str
      }

    Logic:
      - If gold_qa['label'] is numeric (parseable), compare it to any numbers found in direct_answer first;
        if none found, check reasoning_answer numbers.
      - Accept if |pred - gold| <= max(abs_tol, rel_tol * |gold|)
    """
    label = gold_qa.get("label")
    if label is None:
        return {"ok": None, "note": "no gold label"}

    # try parse gold numeric
    try:
        gold_num = float(label)
    except Exception:
        return {"ok": None, "note": "gold label not numeric"}

    # parse numbers from direct_answer, reasoning_answer
    da = answer.get("direct_answer") or ""
    ra = answer.get("reasoning_answer") or ""
    pred_nums = _extract_numbers_from_text(f"{da}\n{ra}")

    if not pred_nums:
        return {
            "ok": False,
            "gold_numeric": gold_num,
            "pred_numbers": [],
            "closest_pred": None,
            "error_abs": None,
            "error_rel": None,
            "within_tolerance": False,
            "note": "no numeric value found in answer"
        }

    # choose closest to gold
    closest = min(pred_nums, key=lambda x: abs(x - gold_num))
    err_abs = abs(closest - gold_num)
    err_rel = err_abs / (abs(gold_num) + 1e-12)

    within = err_abs <= max(abs_tol, rel_tol * abs(gold_num))

    return {
        "ok": within,
        "gold_numeric": gold_num,
        "pred_numbers": pred_nums,
        "closest_pred": closest,
        "error_abs": err_abs,
        "error_rel": err_rel,
        "within_tolerance": within,
        "note": f"tolerance used: rel_tol={rel_tol}, abs_tol={abs_tol}"
    }


def embed_similarity_check(
    answer_text: str,
    gold_text: str,
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Compute embedding cosine similarity between answer_text and gold_text.

    Returns:
      {
        "available": bool,     # False if sentence-transformers not installed
        "similarity": float or None,
        "passes_threshold": bool or None,
        "note": str
      }
    """
    if not _EMBED_AVAILABLE:
        return {"available": False, "similarity": None, "passes_threshold": None,
                "note": "sentence-transformers not available"}

    if not model_name:
        model_name = "all-MiniLM-L6-v2"
    try:
        model = SentenceTransformer(model_name)
        emb1 = model.encode(answer_text, convert_to_tensor=True)
        emb2 = model.encode(gold_text, convert_to_tensor=True)
        sim = float(st_util.cos_sim(emb1, emb2).cpu().item())
        passes = sim >= float(threshold)
        return {"available": True, "similarity": sim, "passes_threshold": passes, "note": f"threshold={threshold}"}
    except Exception as e:
        return {"available": False, "similarity": None, "passes_threshold": None, "note": f"error: {e}"}
