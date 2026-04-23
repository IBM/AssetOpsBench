#!/usr/bin/env python3
# scripts/normalize_pdm_facts.py
"""
Normalize PdM facts JSONL:
 - If failure_profile.iso_metadata exists, merge it up into failure_profile.
 - If asset_profile is missing equipment_category / equipment_class_type, copy them
   from failure_profile (after merge).
 - Skip facts where 'healthy' appears in fact_id or label (case-insensitive).
Writes normalized output to a new JSONL file.
Usage:
    python scripts/normalize_pdm_facts.py --in pdm_facts.jsonl --out pdm_facts_normalized.jsonl
"""

import argparse
import json
from typing import Dict, Any


def normalize_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    # Safety: do not modify facts that are explicitly 'healthy'
    fact_id = str(fact.get("fact_id", "")).lower()
    label = str(fact.get("label", "")).lower()
    if "healthy" in fact_id or "healthy" in label:
        return fact  # return unchanged

    fp = fact.get("failure_profile")
    # merge iso_metadata up if present
    if isinstance(fp, dict):
        iso = fp.get("iso_metadata")
        if isinstance(iso, dict):
            # merge (iso overrides nothing; prefer top-level values if present)
            merged = dict(fp)
            # ensure keys in iso are available as top-level in failure_profile
            for k, v in iso.items():
                # only set if not already present OR you can choose to override
                if merged.get(k) is None:
                    merged[k] = v
            fact["failure_profile"] = merged
            fp = merged

    # ensure asset_profile exists
    ap = fact.get("asset_profile") or {}
    if not isinstance(ap, dict):
        ap = {}

    # copy equipment fields if missing on asset_profile
    if isinstance(fp, dict):
        ec = fp.get("equipment_category")
        ect = fp.get("equipment_class_type")
        changed = False
        if ec and not ap.get("equipment_category"):
            ap = dict(ap)  # shallow copy
            ap["equipment_category"] = ec
            changed = True
        if ect and not ap.get("equipment_class_type"):
            ap = dict(ap)
            ap["equipment_class_type"] = ect
            changed = True
        if changed:
            fact["asset_profile"] = ap

    return fact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True, help="input jsonl facts file")
    parser.add_argument("--out", dest="outfile", required=True, help="output normalized jsonl file")
    args = parser.parse_args()

    n_in = 0
    n_out = 0
    with open(args.infile, "r") as fin, open(args.outfile, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                fact = json.loads(line)
            except Exception:
                # preserve line if it can't be parsed (optional)
                fout.write(line + "\n")
                continue
            fact_norm = normalize_fact(fact)
            fout.write(json.dumps(fact_norm) + "\n")
            n_out += 1

    print(f"Processed {n_in} facts -> wrote {n_out} normalized facts to {args.outfile}")


if __name__ == "__main__":
    main()
