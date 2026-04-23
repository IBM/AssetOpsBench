# src/utils/static_fact_extractor.py
"""
Static Fact Extractor for diagnostic CSVs (USM-like).
- Reads a CSV where each row is a diagnostic snapshot for an asset instance.
- Produces JSONL of facts where each fact represents one diagnostic episode (row).
- Attaches provenance (dataset, file, row_id, asset_id) and per-feature entries.

API:
- extract_facts_from_csv(path, asset_id=None, label_cols=None, out_path=None)
- extract_fact_from_row(row, feature_cols, label_col, asset_id, row_index, source_file)
- infer_schema_from_df(df, possible_label_names=["class","label","fault","health_state"])
- CLI supports --input, --asset, --label-col, --out

Output:
A JSONL file (one JSON object per episode) with fields:
{
  "fact_id": "<asset>_<row_index>",
  "dataset": "<dataset name>",
  "source_file": "<input csv>",
  "asset_id": "<asset if provided or inferred>",
  "row_index": int,
  "features": [{"name": "<col>", "value": <num|string>}...],
  "label": "<label string or null>",
  "provenance": { "file": "<input csv>", "row": <row_index> },
  "confidence": 1.0
}
"""

from __future__ import annotations
import csv
import json
import os
import uuid
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd


def infer_schema_from_df(df: pd.DataFrame, possible_label_names: List[str] = None
                         ) -> Tuple[List[str], Optional[str]]:
    """
    Return feature columns and label column (if detected).
    Default heuristics:
      - If a column name in possible_label_names exists, choose it as label.
      - Else if 'class' in columns -> label
      - Else attempt to find a small-nunique column (<= 20 distinct) labelled-like.
    """
    if possible_label_names is None:
        possible_label_names = ["class", "label", "fault", "health_state", "state", "target"]
    cols = list(df.columns)
    label_col = None
    for name in possible_label_names:
        if name in cols:
            label_col = name
            break
    if label_col is None:
        # fallback: find a column with small number of unique values and non-numeric types or small unique count
        for c in cols:
            if df[c].nunique(dropna=True) <= 20 and df[c].dtype == object:
                label_col = c
                break
        # last resort, pick 'class' if present anywhere-case-insensitive
        for c in cols:
            if c.lower() == "class":
                label_col = c
                break
    feature_cols = [c for c in cols if c != label_col]
    return feature_cols, label_col


def extract_fact_from_row(row: pd.Series,
                          feature_cols: List[str],
                          label_col: Optional[str],
                          asset_id: Optional[str],
                          row_index: int,
                          source_file: str,
                          dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert a DataFrame row into a fact dict with provenance.
    """
    # build features list
    features = []
    for c in feature_cols:
        # normalize types: try numeric conversion, else keep original
        val = row.get(c, None)
        try:
            # allow NaN -> None
            if pd.isna(val):
                v = None
            else:
                v = float(val) if _is_number_like(val) else val
        except Exception:
            v = val
        features.append({"name": c, "value": v})

    label_value = None
    if label_col:
        label_value = row.get(label_col, None)
        if pd.isna(label_value):
            label_value = None

    fact_id = f"{asset_id or 'asset'}_{row_index}"

    fact = {
        "fact_id": fact_id,
        "dataset": dataset_name or "static_diagnostic",
        "source_file": os.path.basename(source_file),
        "asset_id": asset_id,
        "row_index": int(row_index),
        "features": features,
        "label": None if label_value is None else str(label_value),
        "provenance": {"file": os.path.basename(source_file), "row": int(row_index)},
        "confidence": 1.0
    }
    return fact


def _is_number_like(x) -> bool:
    """
    Return True if value looks numeric (int/float or numeric string).
    """
    try:
        if x is None:
            return False
        if isinstance(x, (int, float)):
            return True
        s = str(x).strip()
        # allow scientific and decimal values
        float(s)
        return True
    except Exception:
        return False


def extract_facts_from_csv(path: str,
                           asset_id: Optional[str] = None,
                           label_col: Optional[str] = None,
                           dataset_name: Optional[str] = None,
                           out_path: Optional[str] = None,
                           sample_limit: Optional[int] = None) -> str:
    """
    Read CSV and produce JSONL facts file. Returns out_path.

    Parameters:
      - path: path to CSV file
      - asset_id: optional asset id string; if None, set to filename prefix
      - label_col: optional label column name; if None, the function will infer it
      - dataset_name: optional dataset name to store in fact
      - out_path: output jsonl path; default: replace .csv -> _facts.jsonl
      - sample_limit: if provided, only process first N rows (helpful for dev)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if sample_limit is not None:
        df = df.head(sample_limit)

    feature_cols, inferred_label = infer_schema_from_df(df)
    # explicit label overrides inferred
    label_col = label_col or inferred_label

    # if user didn't provide asset_id, derive from filename
    if asset_id is None:
        base = os.path.basename(path)
        asset_id = os.path.splitext(base)[0]

    if out_path is None:
        out_path = os.path.splitext(path)[0] + "_facts.jsonl"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as fout:
        for idx, row in df.reset_index(drop=True).iterrows():
            fact = extract_fact_from_row(row, feature_cols, label_col, asset_id, idx, path, dataset_name)
            fout.write(json.dumps(fact) + "\n")

    return out_path


# --------------------
# CLI helper
# --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Static Fact Extractor for diagnostic CSVs")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--asset", "-a", default=None, help="Asset id (optional); default uses filename")
    parser.add_argument("--label-col", "-l", default=None, help="Label column name (optional)")
    parser.add_argument("--dataset", "-d", default=None, help="Dataset name to embed in facts (optional)")
    parser.add_argument("--out", "-o", default=None, help="Output JSONL path (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (optional)")
    args = parser.parse_args()
    out = extract_facts_from_csv(args.input, asset_id=args.asset, label_col=args.label_col,
                                dataset_name=args.dataset, out_path=args.out, sample_limit=args.limit)
    print(f"Wrote facts to {out}")
