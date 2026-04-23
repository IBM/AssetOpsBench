# src/utils/verifier_static.py
"""
Verifier for static diagnostic QA answers.

Given:
  - an EpisodicStore
  - a gold QA instance (optional, for consistency checks)
  - a model answer JSON (as dict)

It checks:
  - JSON structure (required keys)
  - provenance.fact_id exists in the store
  - provenance.features are valid feature names for that fact
  - provenance.file,row match fact metadata (if provided)

This patched version is defensive and JSONL-aware:
 - provenance.file/row are optional (no longer treated as always required)
 - normalizes feature names
 - validates confidence for NaN/Inf
 - safer row casting and clearer issues
 - returns extra debug fields in provenance result
 - CLI accepts single JSON, JSON array, or JSONL (NDJSON)
"""
from __future__ import annotations
import json
import math
import sys
from typing import Dict, Any, List, Optional, Set

from src.utils.episodic_store import EpisodicStore

# Top-level keys that we expect
REQUIRED_TOP_KEYS: Set[str] = {"direct_answer", "reasoning_answer", "provenance", "confidence"}

# Provenance keys we will *check for if present*; file/row are optional but preferred
RECOMMENDED_PROV_KEYS: Set[str] = {"fact_id", "features", "file", "row"}


def _norm_str(s: Any) -> str:
    """Normalize strings for comparison (strip + lower). If not str, convert to str."""
    if s is None:
        return ""
    try:
        return str(s).strip().lower()
    except Exception:
        return str(s)


def _is_number(x: Any) -> bool:
    """Return True if x can be converted to a finite float."""
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def verify_structure(answer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check that all required keys are present and of roughly correct type.

    Returns dict with:
      - ok: bool
      - issues: list[str]
      - hints: optional dict with extra small debug info
    """
    issues: List[str] = []
    hints: Dict[str, Any] = {}

    if not isinstance(answer, dict):
        return {"ok": False, "issues": ["answer must be a JSON object/dict"], "hints": {}}

    # Missing top-level keys
    missing = REQUIRED_TOP_KEYS - set(answer.keys())
    if missing:
        issues.append(f"Missing top-level keys: {sorted(missing)}")

    # direct_answer / reasoning_answer type checks (prefer strings)
    da = answer.get("direct_answer")
    ra = answer.get("reasoning_answer")
    if da is not None and not isinstance(da, (str, int, float, bool)):
        issues.append("direct_answer should be a string or scalar value")
    if ra is not None and not isinstance(ra, (str, int, float, bool)):
        issues.append("reasoning_answer should be a string or scalar value")

    # provenance must be object
    prov = answer.get("provenance")
    if prov is None:
        issues.append("provenance is missing")
    else:
        if not isinstance(prov, dict):
            issues.append("provenance must be an object/dict")
        else:
            # features should be a list when present
            feats = prov.get("features", None)
            if feats is not None and not isinstance(feats, list):
                issues.append("provenance.features must be a list if present")
            else:
                # if list, ensure items are strings (or convertible)
                if isinstance(feats, list):
                    non_strings = [f for f in feats if not isinstance(f, (str, int, float, bool))]
                    if non_strings:
                        issues.append("provenance.features contains non-scalar entries (expected strings/ids)")

            # file and row are allowed to be missing; no structural error if missing

    # confidence checks
    if "confidence" not in answer:
        issues.append("confidence is missing")
    else:
        conf = answer.get("confidence")
        if not _is_number(conf):
            issues.append("confidence must be a finite numeric value between 0 and 1")
        else:
            c = float(conf)
            if not (0.0 <= c <= 1.0):
                issues.append("confidence must be between 0 and 1 inclusive")

    return {"ok": len(issues) == 0, "issues": issues, "hints": hints}


def verify_provenance_against_store(
    answer: Dict[str, Any],
    store: EpisodicStore,
) -> Dict[str, Any]:
    """
    Check that:
      - provenance.fact_id exists (if provided)
      - provenance.features are real features of that fact (if provided)
      - provenance.file and provenance.row match fact metadata if provided

    Returns:
      {
        "ok": bool,
        "issues": [...],
        "expected_features": [...],     # normalized expected feature names (from store)
        "provided_features": [...],     # normalized provided feature names (from answer)
        "fact_present": bool,
        "fact_summary": { "source_file": ..., "row_index": ... }  # when fact exists
      }
    """
    issues: List[str] = []
    prov = answer.get("provenance", {}) if isinstance(answer, dict) else {}
    fact_id = prov.get("fact_id")
    features = prov.get("features", []) or []
    file_ = prov.get("file", None)
    row = prov.get("row", None)

    result: Dict[str, Any] = {
        "ok": True,
        "issues": [],
        "expected_features": [],
        "provided_features": [_norm_str(f) for f in features],
        "fact_present": False,
        "fact_summary": {},
    }

    if fact_id is None:
        issues.append("provenance.fact_id is missing")
        result["ok"] = False
        result["issues"] = issues
        return result

    # attempt to fetch fact from store
    try:
        fact = store.get_fact(str(fact_id))
    except Exception as e:
        issues.append(f"error when querying EpisodicStore for fact_id '{fact_id}': {e}")
        result["ok"] = False
        result["issues"] = issues
        return result

    if fact is None:
        issues.append(f"provenance.fact_id '{fact_id}' not found in EpisodicStore")
        result["ok"] = False
        result["issues"] = issues
        return result

    # fact exists
    result["fact_present"] = True
    fact_file = fact.get("source_file")
    fact_row = fact.get("row_index")
    result["fact_summary"] = {"source_file": fact_file, "row_index": fact_row}

    # Build expected feature name set defensively
    expected_names: Set[str] = set()
    raw_feats = fact.get("features", []) or []
    for item in raw_feats:
        # support both {"name": "..."} and plain string entries
        if isinstance(item, dict):
            name = item.get("name")
            if name is not None:
                expected_names.add(_norm_str(name))
        else:
            expected_names.add(_norm_str(item))
    result["expected_features"] = sorted(expected_names)

    # Check file match if provided
    if file_ is not None and fact_file is not None and _norm_str(file_) != _norm_str(fact_file):
        issues.append(f"provenance.file='{file_}' does not match fact.source_file='{fact_file}'")

    # Safe integer cast/comparison for row
    if row is not None and fact_row is not None:
        try:
            if int(row) != int(fact_row):
                issues.append(f"provenance.row={row} does not match fact.row_index={fact_row}")
        except (TypeError, ValueError):
            issues.append(f"provenance.row={row} is not an integer")

    # Validate features provided exist in expected features
    invalid_feats = []
    for f in features or []:
        if _norm_str(f) not in expected_names:
            invalid_feats.append(f)
    if invalid_feats:
        issues.append(f"provenance.features contain unknown feature(s): {sorted(invalid_feats)}")

    if issues:
        result["ok"] = False
        result["issues"] = issues
    else:
        result["ok"] = True
        result["issues"] = []

    return result


def verify_answer(
    answer: Dict[str, Any],
    store: EpisodicStore,
) -> Dict[str, Any]:
    """
    Full verification entrypoint.

    Returns:
      {
        "structure_ok": bool,
        "provenance_ok": bool,
        "structure_issues": [...],
        "provenance_issues": [...],
        "provenance_debug": {...}  # includes expected/provided features and fact summary
      }
    """
    s = verify_structure(answer)
    p = verify_provenance_against_store(answer, store)

    return {
        "structure_ok": s.get("ok", False),
        "provenance_ok": p.get("ok", False),
        "structure_issues": s.get("issues", []),
        "provenance_issues": p.get("issues", []),
        "provenance_debug": {
            "expected_features": p.get("expected_features", []),
            "provided_features": p.get("provided_features", []),
            "fact_present": p.get("fact_present", False),
            "fact_summary": p.get("fact_summary", {}),
        },
    }


# CLI: verify one or many answer JSON/JSONL records
if __name__ == "__main__":
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description="Verify model answer JSON or JSONL against EpisodicStore.")
    parser.add_argument("--db", required=True, help="Path to episodic_store.db")
    parser.add_argument("--answer", required=True, help="Path to JSON file or JSONL (NDJSON) with model answers")
    parser.add_argument("--out", required=False, help="Optional output JSONL path for reports (default stdout)")
    parser.add_argument("--verbose", action="store_true", help="Print additional debugging info")
    args = parser.parse_args()

    # load answers: support single JSON object, JSON array, or JSONL (one JSON object per line)
    try:
        with open(args.answer, "r") as f:
            text = f.read()
    except Exception as e:
        print(json.dumps({"error": f"Failed to read answer file: {e}"}))
        if args.verbose:
            traceback.print_exc()
        sys.exit(2)

    records = []
    text_stripped = text.strip()
    if not text_stripped:
        print(json.dumps({"error": "Answer file is empty"}))
        sys.exit(2)

    # Try parsing as a single JSON value (object or array)
    parsed_single = None
    try:
        parsed_single = json.loads(text_stripped)
    except Exception:
        parsed_single = None

    if parsed_single is not None:
        # If it's a list, treat each member as a record; otherwise single record
        if isinstance(parsed_single, list):
            records = parsed_single
        else:
            records = [parsed_single]
    else:
        # Fallback: treat file as JSONL (one JSON object per non-empty line)
        records = []
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(json.dumps({"error": f"JSON decode error in line {i}: {e}"}))
                if args.verbose:
                    traceback.print_exc()
                sys.exit(2)
            records.append(rec)

    # Open store
    try:
        store = EpisodicStore(db_path=args.db)
    except Exception as e:
        print(json.dumps({"error": f"Failed to open EpisodicStore at '{args.db}': {e}"}))
        if args.verbose:
            traceback.print_exc()
        sys.exit(3)

    # Prepare output
    out_f = None
    try:
        if args.out:
            out_f = open(args.out, "w")
        else:
            out_f = sys.stdout

        for idx, rec in enumerate(records):
            # support two shapes:
            # 1) {"qa_id": "...", "answer": {...}}
            # 2) direct answer object {...}
            qa_id = rec.get("qa_id") if isinstance(rec, dict) else f"idx_{idx}"
            ans = None
            if isinstance(rec, dict) and "answer" in rec and isinstance(rec["answer"], dict):
                qa_id = rec.get("qa_id", qa_id)
                ans = rec["answer"]
            elif isinstance(rec, dict) and any(k in rec for k in ("direct_answer", "reasoning_answer", "provenance")):
                ans = rec
            else:
                # Unexpected shape — treat as error but continue
                out_record = {"qa_id": qa_id, "error": "Record does not contain an 'answer' dict or expected keys"}
                print(json.dumps(out_record), file=out_f)
                continue

            try:
                report = verify_answer(ans, store)
            except Exception as e:
                report = {"error": f"verification exception: {e}"}
                if args.verbose:
                    traceback.print_exc()

            out_record = {"qa_id": qa_id, "report": report}
            print(json.dumps(out_record), file=out_f)

    finally:
        try:
            if out_f and out_f is not sys.stdout:
                out_f.close()
        except Exception:
            pass
        try:
            store.close()
        except Exception:
            pass

    # If any record failed (structure_ok or provenance_ok false) exit with code 1
    any_fail = False
    for rec in records:
        # extract qa_id and answer as above to recompute minimal check without re-running verifier
        if isinstance(rec, dict) and "answer" in rec and isinstance(rec["answer"], dict):
            ans = rec["answer"]
        elif isinstance(rec, dict) and any(k in rec for k in ("direct_answer", "reasoning_answer", "provenance")):
            ans = rec
        else:
            continue
        s = verify_structure(ans)
        p = verify_provenance_against_store(ans, store)
        if not (s.get("ok") and p.get("ok")):
            any_fail = True
            break

    if any_fail:
        sys.exit(1)
    sys.exit(0)
