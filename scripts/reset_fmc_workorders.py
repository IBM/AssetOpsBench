#!/usr/bin/env python
"""Reset the ``wo_fmc`` work orders in CouchDB back to their seed state.

The failure-code write-back scenarios (S2/S4/S5) mutate ``wo_fmc`` records in
the ``workorder`` database.  This restores every record's ``failure_code`` to
the value in the seed CSV (``src/couchdb/sample_data/work_order/wo_fmc.csv``):
TRN- records keep their historical codes, TST- records go back to blank.  Only
records that have drifted from the seed are written.

For a full rebuild of the entire ``workorder`` DB from all CSVs instead, use:
    cd src && uv run python -m couchdb.init_wo --drop

Usage:
    uv run python scripts/reset_fmc_workorders.py
    uv run python scripts/reset_fmc_workorders.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))

_CSV = _SRC / "couchdb" / "sample_data" / "work_order" / "wo_fmc.csv"


def _seed_codes() -> dict[str, str | None]:
    """Map wo_id → seed failure_code (None for blank) from the CSV."""
    seed: dict[str, str | None] = {}
    with open(_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("failure_code") or "").strip()
            seed[row["wo_id"]] = code or None
    return seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report drift without writing."
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")

    import pandas as pd

    from servers.wo.data import load, write_failure_codes

    seed = _seed_codes()
    blanks = sum(1 for v in seed.values() if v is None)
    print(f"seed: {len(seed)} wo_fmc records ({len(seed) - blanks} labelled, {blanks} blank)")

    df = load("wo_fmc")
    if df is None:
        print("CouchDB unavailable or wo_fmc not loaded — nothing to reset.")
        sys.exit(1)

    def _norm(v) -> str | None:
        if pd.isna(v) or not str(v).strip():
            return None
        return str(v).strip()

    current = {str(r["wo_id"]): _norm(r.get("failure_code")) for _, r in df.iterrows()}
    drift = {wo_id: code for wo_id, code in seed.items() if current.get(wo_id) != code}

    print(f"drifted from seed: {len(drift)} record(s)")
    if not drift:
        print("already at seed state — nothing to do.")
        return
    if args.dry_run:
        for wo_id in list(drift)[:20]:
            print(f"  {wo_id}: {current.get(wo_id)!r} -> {drift[wo_id]!r}")
        if len(drift) > 20:
            print(f"  … and {len(drift) - 20} more")
        return

    status = write_failure_codes(drift)
    if status is None:
        print("CouchDB unavailable — reset aborted.")
        sys.exit(1)
    restored = sum(1 for ok in status.values() if ok)
    print(f"reset {restored}/{len(drift)} wo_fmc record(s) to seed state.")


if __name__ == "__main__":
    main()
