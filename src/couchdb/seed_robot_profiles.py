"""Seed robot asset profile documents into the iot CouchDB database.

Creates one profile document per known asset containing robot-inspection fields
(navigation, gauge truth, calibration, dispatch state).  These documents are
deliberately stored WITHOUT an ``asset_id`` field so existing IoT server
queries (``{"asset_id": {"$exists": true}}``) are completely unaffected.

Document shape:
    _id        = "profile:{normalized_asset_id}"   e.g. "profile:chiller_6"
    doc_type   = "asset_robot_profile"
    display_name = "Chiller 6"                     original asset_id string
    + 9 robot fields (see ROBOT_FIELD_DEFAULTS)

Usage:
    python src/couchdb/seed_robot_profiles.py             # apply
    python src/couchdb/seed_robot_profiles.py --dry-run   # preview only
    python src/couchdb/seed_robot_profiles.py --verify    # check DB state
"""

import argparse
import json
import os
import sys

import couchdb3
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Connection — identical pattern to src/servers/iot/main.py
# ---------------------------------------------------------------------------
COUCHDB_URL      = os.environ.get("COUCHDB_URL")
COUCHDB_DBNAME   = os.environ.get("IOT_DBNAME")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD")

# ---------------------------------------------------------------------------
# Robot field defaults (in-scope fields only)
# gauge_value is stored here but MUST NEVER be returned by any MCP tool.
# ---------------------------------------------------------------------------
ROBOT_FIELD_DEFAULTS: dict = {
    "physical_location":   None,
    "gauge_value":         0.0,    # ground truth — NEVER expose to agent via MCP
    "gauge_range":         [0, 100],
    "panel_stuck_prob":    0.12,
    "human_present":       False,
    "never_read":          False,
    "real_gauge_images":   [],
    "reading_consistency": None,
    "sensor_physical_gap": None,
}

ROBOT_FIELDS = list(ROBOT_FIELD_DEFAULTS.keys())

# ---------------------------------------------------------------------------
# Known assets — derived from sample_data/iot/*.json
# physical_location values are placeholder coordinates until floor-plan data arrives.
# ---------------------------------------------------------------------------
ASSETS = [
    {
        "display_name":    "Chiller 6",
        "profile_id":      "profile:chiller_6",
        "physical_location": {"x": 52.3, "y": 18.1, "z": 0.0, "room_id": "cooling_3B"},
        "gauge_range":     [0, 100],
    },
    {
        "display_name":    "Metro Pump 1",
        "profile_id":      "profile:metro_pump_1",
        "physical_location": {"x": 14.0, "y": 32.5, "z": 0.0, "room_id": "pump_room_A"},
        "gauge_range":     [0, 200],
    },
    {
        "display_name":    "Hydraulic Pump 1",
        "profile_id":      "profile:hydraulic_pump_1",
        "physical_location": {"x": 28.7, "y": 11.0, "z": 0.0, "room_id": "pump_room_B"},
        "gauge_range":     [0, 350],
    },
    {
        "display_name":    "Motor 01",
        "profile_id":      "profile:motor_01",
        "physical_location": {"x": 7.2, "y": 44.8, "z": 0.0, "room_id": "motor_bay_1"},
        "gauge_range":     [0, 60],
    },
]

# ---------------------------------------------------------------------------
# Indexes for Robot MCP tool query performance
# ---------------------------------------------------------------------------
ROBOT_INDEXES = [
    {
        "name":   "idx_robot_never_read",
        "fields": ["doc_type", "never_read"],
        "reason": "scenario generator: never-read gauge cases",
    },
]


def _connect() -> couchdb3.Database:
    if not COUCHDB_URL or not COUCHDB_DBNAME:
        sys.exit("ERROR: COUCHDB_URL and IOT_DBNAME must be set.")
    return couchdb3.Database(
        COUCHDB_DBNAME,
        url=COUCHDB_URL,
        user=COUCHDB_USERNAME,
        password=COUCHDB_PASSWORD,
    )


def _build_doc(asset: dict) -> dict:
    doc = {
        "_id":          asset["profile_id"],
        "doc_type":     "asset_robot_profile",
        "display_name": asset["display_name"],
    }
    doc.update(ROBOT_FIELD_DEFAULTS)
    # Per-asset overrides
    doc["physical_location"] = asset["physical_location"]
    doc["gauge_range"]        = asset["gauge_range"]
    return doc


def run(dry_run: bool = False) -> None:
    """Upsert all robot profile documents and create indexes."""
    db = _connect()

    print(f"{'[DRY RUN] ' if dry_run else ''}Seeding robot profiles into '{COUCHDB_DBNAME}'...\n")

    for asset in ASSETS:
        doc_id = asset["profile_id"]
        new_doc = _build_doc(asset)

        try:
            existing = db.get(doc_id)
        except Exception:
            existing = None

        if existing is None:
            action = "CREATE"
            final_doc = new_doc
        else:
            # Patch only fields that are missing (never overwrite existing values)
            patched = False
            final_doc = dict(existing)
            for field in ROBOT_FIELDS:
                if field not in final_doc:
                    final_doc[field] = new_doc[field]
                    patched = True
            action = "PATCH" if patched else "SKIP (already complete)"

        if dry_run:
            print(f"  [{action}] {doc_id}")
            if action != "SKIP (already complete)":
                print(f"           {json.dumps(new_doc, indent=10)}\n")
        else:
            if action == "SKIP (already complete)":
                print(f"  [SKIP]   {doc_id} — all robot fields already present")
            else:
                db.save(final_doc)
                print(f"  [{action}]   {doc_id}")

    if not dry_run:
        _ensure_indexes()

    print("\nDone." if not dry_run else "\n[Dry run complete — no writes performed.]")


def _ensure_indexes() -> None:
    # couchdb3 does not expose create_index; use the HTTP API directly (same
    # pattern as src/couchdb/loader.py _create_indexes).
    auth = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
    base = (COUCHDB_URL or "").rstrip("/")
    print("\nCreating indexes...")
    for idx in ROBOT_INDEXES:
        url = f"{base}/{COUCHDB_DBNAME}/_index"
        payload = {
            "index": {"fields": idx["fields"]},
            "name":  idx["name"],
            "type":  "json",
        }
        try:
            resp = requests.post(url, json=payload, auth=auth, timeout=10)
            resp.raise_for_status()
            result = resp.json().get("result", "ok")
            print(f"  [{result.upper()}] {idx['name']}")
        except Exception as e:
            print(f"  [WARN] {idx['name']}: {e}")


def verify() -> bool:
    """Check that all 4 profiles exist with all 9 robot fields. Returns True if OK."""
    db = _connect()
    print(f"Verifying robot profiles in '{COUCHDB_DBNAME}'...\n")
    all_ok = True

    for asset in ASSETS:
        doc_id = asset["profile_id"]
        try:
            doc = db.get(doc_id)
        except Exception:
            doc = None

        if doc is None:
            print(f"  [MISSING] {doc_id}")
            all_ok = False
            continue

        missing = [f for f in ROBOT_FIELDS if f not in doc]
        if missing:
            print(f"  [INCOMPLETE] {doc_id} — missing fields: {missing}")
            all_ok = False
        else:
            present = {f: doc[f] for f in ROBOT_FIELDS}
            print(f"  [OK] {doc_id}")
            for k, v in present.items():
                flag = "  *** GROUND TRUTH — never expose ***" if k == "gauge_value" else ""
                print(f"       {k}: {v}{flag}")
        print()

    if all_ok:
        print("All profiles verified successfully.")
    else:
        print("VERIFICATION FAILED — run seed_robot_profiles.py to fix.")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed robot asset profiles into CouchDB iot DB.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    group.add_argument("--verify",  action="store_true", help="Check profiles exist and are complete")
    args = parser.parse_args()

    if args.verify:
        ok = verify()
        sys.exit(0 if ok else 1)
    else:
        run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
