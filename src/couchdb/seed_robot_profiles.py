"""Seed robot documents into the iot CouchDB database.

Creates three document types per robot deployment:

  profile:{asset_id}      — per-asset physical layout and panel state (5 fields)
  robot_state:{robot_id}  — per-robot battery and pose state
  waypoints               — singleton inspection waypoint map

All documents deliberately omit the ``asset_id`` field so existing IoT server
queries (``{"asset_id": {"$exists": true}}``) are completely unaffected.

Scenario-specific state (e.g. low battery, localisation failure) is loaded by the
eval harness via init_data.py + the shared robot JSON files in
AssetOpsBenchScenarioGeneration/RobotInspection/shared/robot/.

Usage:
    python src/couchdb/seed_robot_profiles.py             # seed all (nominal state)
    python src/couchdb/seed_robot_profiles.py --dry-run   # preview
    python src/couchdb/seed_robot_profiles.py --verify    # check seeded state
    python src/couchdb/seed_robot_profiles.py --reset     # restore nominal state
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import couchdb3
import requests
from dotenv import load_dotenv

load_dotenv()

COUCHDB_URL      = os.environ.get("COUCHDB_URL")
COUCHDB_DBNAME   = os.environ.get("IOT_DBNAME")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD")
ROBOT_ID         = os.environ.get("ROBOT_ID", "spot_1")

# ---------------------------------------------------------------------------
# PART 1 — Asset profile documents (5 fields per asset)
# ---------------------------------------------------------------------------

PROFILE_DEFAULTS: dict = {
    "physical_location": None,   # navigate_to target; None until floor-plan loaded
    "gauge_path":        None,   # capture_image returns this; None until field data collected
    "gauge_range":       [0, 100],
    "gauge_description": "",
    "panel_stuck":       False,  # open_panel + capture_image occlusion
}
PROFILE_FIELDS = list(PROFILE_DEFAULTS.keys())

ASSETS = [
    {
        "display_name":      "Chiller 6",
        "profile_id":        "profile:chiller_6",
        "waypoint_id":       "wp_chiller_6_panel",
        "physical_location": {"x": 52.3, "y": 18.1, "z": 0.0, "room_id": "cooling_3B"},
        "gauge_range":       [0, 400],
        "gauge_description": "Compressor outlet pressure gauge, analog dial, 0–400 bar",
    },
    {
        "display_name":      "Metro Pump 1",
        "profile_id":        "profile:metro_pump_1",
        "waypoint_id":       "wp_metro_pump_1_panel",
        "physical_location": {"x": 14.0, "y": 32.5, "z": 0.0, "room_id": "pump_room_A"},
        "gauge_range":       [0.0, 1.5],
        "gauge_description": "Flow rate gauge, analog dial, 0–1.5 m³/s",
    },
    {
        "display_name":      "Hydraulic Pump 1",
        "profile_id":        "profile:hydraulic_pump_1",
        "waypoint_id":       "wp_hydraulic_pump_1_panel",
        "physical_location": {"x": 28.7, "y": 11.0, "z": 0.0, "room_id": "pump_room_B"},
        "gauge_range":       [0, 350],
        "gauge_description": "Hydraulic pressure gauge, analog dial, 0–350 bar",
    },
    {
        "display_name":      "Motor 01",
        "profile_id":        "profile:motor_01",
        "waypoint_id":       "wp_motor_01_panel",
        "physical_location": {"x": 7.2, "y": 44.8, "z": 0.0, "room_id": "motor_bay_1"},
        "gauge_range":       [0, 200],
        "gauge_description": "Motor temperature gauge, analog dial, 0–200°C",
    },
]

# ---------------------------------------------------------------------------
# PART 2 — Robot state document
# ---------------------------------------------------------------------------

def _nominal_robot_state() -> dict:
    return {
        "_id":                         f"robot_state:{ROBOT_ID}",
        "doc_type":                    "robot_state",
        "robot_id":                    ROBOT_ID,
        "battery_charge_pct":          85.0,
        "battery_low_threshold":       20.0,
        "battery_estimated_runtime_s": 5400.0,
        "at_charge_station":           False,
        "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "frame": "map"},
        "localization_ok":             True,
        "pose_drift_m":                0.0,
        "fault_state":                 None,
        "last_updated":                datetime.now(timezone.utc).isoformat(),
    }

ROBOT_STATE_FIELDS = [
    "battery_charge_pct", "battery_low_threshold", "battery_estimated_runtime_s",
    "at_charge_station", "pose", "localization_ok", "pose_drift_m", "fault_state",
]

# ---------------------------------------------------------------------------
# PART 3 — Waypoints singleton document
# ---------------------------------------------------------------------------

NOMINAL_WAYPOINTS: dict = {
    "_id":      "waypoints",
    "doc_type": "waypoints",
    "waypoints": [
        {
            "waypoint_id":          "wp_chiller_6_panel",
            "waypoint_name":        "Chiller 6 — Panel C-06",
            "asset_id":             "chiller_6",
            "location_description": "Plant B / Mechanical Room 2 / Panel C-06",
            "x": 52.3, "y": 18.1,
            "active": True,
        },
        {
            "waypoint_id":          "wp_metro_pump_1_panel",
            "waypoint_name":        "Metro Pump 1 — Panel MP-01",
            "asset_id":             "metro_pump_1",
            "location_description": "Plant D / Pump Hall / Panel MP-01",
            "x": 14.0, "y": 32.5,
            "active": True,
        },
        {
            "waypoint_id":          "wp_hydraulic_pump_1_panel",
            "waypoint_name":        "Hydraulic Pump 1 — Panel HP-01",
            "asset_id":             "hydraulic_pump_1",
            "location_description": "Plant C / Hydraulic Room / Panel HP-01",
            "x": 28.7, "y": 11.0,
            "active": True,
        },
        {
            "waypoint_id":          "wp_motor_01_panel",
            "waypoint_name":        "Motor 01 — Panel M-01",
            "asset_id":             "motor_01",
            "location_description": "Plant A / Drive Bay 1 / Panel M-01",
            "x": 7.2, "y": 44.8,
            "active": True,
        },
        {
            "waypoint_id":          "wp_dock_main",
            "waypoint_name":        "Main Charging Dock",
            "asset_id":             None,
            "location_description": "Building A / Charging Bay 1",
            "x": 0.0, "y": 0.0,
            "active": True,
        },
    ],
}

# ---------------------------------------------------------------------------
# CouchDB helpers
# ---------------------------------------------------------------------------

ALL_INDEXES = [
    {"name": "idx_robot_state_by_robot", "fields": ["doc_type", "robot_id"]},
    {"name": "idx_waypoints_singleton",  "fields": ["doc_type"]},
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


def _upsert(db, new_doc: dict, doc_id: str, required_fields: list, dry_run: bool) -> str:
    try:
        existing = db.get(doc_id)
    except Exception:
        existing = None
    if existing is None:
        action, final_doc = "CREATE", new_doc
    else:
        patched   = False
        final_doc = dict(existing)
        for field in required_fields:
            if field not in final_doc:
                final_doc[field] = new_doc.get(field)
                patched = True
        action = "PATCH" if patched else "SKIP"
    if not dry_run and action != "SKIP":
        db.save(final_doc)
    return action


def _ensure_indexes(db_name: str) -> None:
    auth = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
    base = (COUCHDB_URL or "").rstrip("/")
    print("\nCreating indexes...")
    for idx in ALL_INDEXES:
        url     = f"{base}/{db_name}/_index"
        payload = {"index": {"fields": idx["fields"]}, "name": idx["name"], "type": "json"}
        try:
            resp = requests.post(url, json=payload, auth=auth, timeout=10)
            resp.raise_for_status()
            result = resp.json().get("result", "ok")
            print(f"  [{result.upper()}] {idx['name']}")
        except Exception as e:
            print(f"  [WARN] {idx['name']}: {e}")


# ---------------------------------------------------------------------------
# Main operations
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    db   = _connect()
    label = "[DRY RUN] " if dry_run else ""
    print(f"{label}Seeding robot documents into '{COUCHDB_DBNAME}'...\n")

    print("── Asset profiles ──")
    for asset in ASSETS:
        doc_id  = asset["profile_id"]
        new_doc = {
            "_id": doc_id, "doc_type": "asset_robot_profile",
            "display_name": asset["display_name"],
            "waypoint_id":  asset["waypoint_id"],
            **PROFILE_DEFAULTS,
            "physical_location": asset["physical_location"],
            "gauge_range":       asset["gauge_range"],
            "gauge_description": asset["gauge_description"],
        }
        action = _upsert(db, new_doc, doc_id, PROFILE_FIELDS, dry_run)
        print(f"  [{action}] {doc_id}")
        if dry_run and action != "SKIP":
            print(f"           {json.dumps({k: v for k, v in new_doc.items() if k not in ('_id', '_rev')}, indent=10)}\n")

    print("\n── Robot state ──")
    state_doc = _nominal_robot_state()
    action    = _upsert(db, state_doc, state_doc["_id"], ROBOT_STATE_FIELDS, dry_run)
    print(f"  [{action}] robot_state:{ROBOT_ID}")

    print("\n── Waypoints ──")
    wps_action = _upsert(db, NOMINAL_WAYPOINTS, "waypoints", ["waypoints"], dry_run)
    print(f"  [{wps_action}] waypoints")

    if not dry_run:
        _ensure_indexes(COUCHDB_DBNAME)
    print("\nDone." if not dry_run else "\n[Dry run complete — no writes performed.]")



def reset(dry_run: bool = False) -> None:
    db    = _connect()
    label = "[DRY RUN] " if dry_run else ""
    print(f"{label}Restoring nominal state...\n")
    state_id = f"robot_state:{ROBOT_ID}"
    try:
        existing = db.get(state_id)
    except Exception:
        existing = None
    if existing:
        nominal = {**_nominal_robot_state(), "_rev": existing["_rev"]}
        if not dry_run:
            db.save(nominal)
        print(f"  [{'WOULD RESET' if dry_run else 'RESET'}] {state_id}")
    try:
        wps_doc = db.get("waypoints")
    except Exception:
        wps_doc = None
    if wps_doc:
        reset_wps = {**NOMINAL_WAYPOINTS, "_rev": wps_doc["_rev"]}
        if not dry_run:
            db.save(reset_wps)
        print(f"  [{'WOULD RESET' if dry_run else 'RESET'}] waypoints")
    print("\nNominal state restored." if not dry_run else "\n[Dry run — no writes.]")


def verify() -> bool:
    db = _connect()
    print(f"Verifying robot documents in '{COUCHDB_DBNAME}'...\n")
    all_ok = True

    print("── Asset profiles ──")
    for asset in ASSETS:
        doc_id = asset["profile_id"]
        try:
            doc = db.get(doc_id)
        except Exception:
            doc = None
        if doc is None:
            print(f"  [MISSING] {doc_id}"); all_ok = False; continue
        missing = [f for f in PROFILE_FIELDS if f not in doc]
        if missing:
            print(f"  [INCOMPLETE] {doc_id} — missing: {missing}"); all_ok = False
        else:
            print(f"  [OK] {doc_id}")
            for k in PROFILE_FIELDS:
                print(f"       {k}: {doc[k]!r}")
        print()

    print("── Robot state ──")
    state_id = f"robot_state:{ROBOT_ID}"
    try:
        state = db.get(state_id)
    except Exception:
        state = None
    if state is None:
        print(f"  [MISSING] {state_id}"); all_ok = False
    else:
        missing = [f for f in ROBOT_STATE_FIELDS if f not in state]
        if missing:
            print(f"  [INCOMPLETE] {state_id} — missing: {missing}"); all_ok = False
        else:
            print(f"  [OK] {state_id}")
            print(f"       battery={state['battery_charge_pct']}% | "
                  f"localization_ok={state['localization_ok']} | "
                  f"drift={state['pose_drift_m']} m")
    print()

    print("── Waypoints ──")
    try:
        wps_doc = db.get("waypoints")
    except Exception:
        wps_doc = None
    if wps_doc is None:
        print("  [MISSING] waypoints"); all_ok = False
    else:
        wps = wps_doc.get("waypoints", [])
        active = sum(1 for w in wps if w.get("active"))
        print(f"  [OK] waypoints — {len(wps)} total, {active} active")
        for wp in wps:
            status = "ACTIVE" if wp.get("active") else "INACTIVE (FM-11)"
            print(f"       {wp['waypoint_id']} → {wp.get('asset_id', 'dock')} [{status}]")

    if all_ok:
        print("\nAll documents verified.")
    else:
        print("\nVERIFICATION FAILED — re-run seed_robot_profiles.py.")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed robot documents into CouchDB iot DB.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--verify",  action="store_true")
    group.add_argument("--reset",   action="store_true")
    args = parser.parse_args()

    if args.verify:
        sys.exit(0 if verify() else 1)
    elif args.reset:
        reset(dry_run=args.dry_run)
    else:
        run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
