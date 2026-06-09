"""Entry point: load a scenario's data into CouchDB — or the default when none is given.

The scenario id maps directly to a manifest file: scenarios_data/scenario_<id>.json.
No scenarios .jsonl lookup is needed — if that file exists it's used, otherwise the
default manifest (scenarios_data/default.json) is loaded.

Each manifest key becomes a database of the same name, loaded by the generic loader
(loader.py); how each is parsed/keyed/indexed comes from collections.json.

    python -m couchdb.init_data 1       # load scenarios_data/scenario_1.json
    python -m couchdb.init_data         # no scenario → default manifest

    from couchdb.init_data import init_data
    init_data(1)      # scenario 1
    init_data()       # default

Databases are rebuilt from scratch each call.
"""

import argparse
import json
import logging
import os

from dotenv import load_dotenv

try:                       # works as a package (python -m couchdb.init_data / imports)
    from . import loader
except ImportError:        # works as a script (python3 /couchdb/init_data.py)
    import loader

load_dotenv()

logger = logging.getLogger("init_data")

_HERE = os.path.dirname(os.path.abspath(__file__))

SCENARIOS_DATA_DIR = os.environ.get("SCENARIOS_DATA_DIR", os.path.join(_HERE, "scenarios_data"))
DEFAULT_MANIFEST_FILE = os.environ.get(
    "DEFAULT_MANIFEST", os.path.join(SCENARIOS_DATA_DIR, "default", "manifest.json"))


# --------------------------------------------------------------------------- #
# Scenario → manifest (filename convention, no scenarios .jsonl needed)
# --------------------------------------------------------------------------- #
def _load_default_manifest() -> tuple:
    """Return (manifest, base_dir) for the default manifest (scenarios_data/default/manifest.json)."""
    if not os.path.isfile(DEFAULT_MANIFEST_FILE):
        raise FileNotFoundError(
            f"default manifest not found: {DEFAULT_MANIFEST_FILE}. "
            "Create scenarios_data/default/manifest.json (or set DEFAULT_MANIFEST).")
    with open(DEFAULT_MANIFEST_FILE) as f:
        return json.load(f), os.path.dirname(DEFAULT_MANIFEST_FILE)


def manifest_path(scenario_id) -> str:
    folder = os.path.join(SCENARIOS_DATA_DIR, f"scenario_{scenario_id}", "manifest.json")
    return folder if os.path.isfile(folder) else os.path.join(SCENARIOS_DATA_DIR, f"scenario_{scenario_id}.json")


def _resolve_manifest(scenario_id) -> tuple:
    """Return (manifest, base_dir).

    scenario_id None → the default manifest. Otherwise the scenario's own manifest
    (folder form scenario_<id>/manifest.json, or legacy flat scenario_<id>.json),
    raising FileNotFoundError if it doesn't exist — an unknown id is never silently
    treated as the default.

    base_dir is the manifest's own folder; the loader resolves its relative data paths
    against that folder, then its parent (so a sibling shared/ corpus is reachable), then
    the couchdb dir. None → couchdb dir only (legacy flat manifests, unchanged).
    """
    if scenario_id is None:
        return _load_default_manifest()
    path = manifest_path(scenario_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no manifest for scenario {scenario_id}: expected "
            f"{os.path.join(SCENARIOS_DATA_DIR, f'scenario_{scenario_id}', 'manifest.json')} "
            f"or {os.path.join(SCENARIOS_DATA_DIR, f'scenario_{scenario_id}.json')}")
    base_dir = os.path.dirname(path) if os.path.basename(path) == "manifest.json" else None
    with open(path) as f:
        return json.load(f), base_dir


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #
def all_databases() -> list:
    """The databases this loader manages = the default manifest's keys (db name = key)."""
    try:
        manifest, _ = _load_default_manifest()
        return list(manifest.keys())
    except Exception:
        return []


def reset(managed_only: bool = False) -> list:
    """Drop databases for a clean state. Returns the dropped names.

    Default: drop every user database (CouchDB GET /_all_dbs, system DBs excluded).
    ``managed_only=True`` drops only the default-manifest collections.
    """
    targets = all_databases() if managed_only else loader.list_databases()
    dropped = []
    for db in targets:
        code = loader.drop_database(db)
        logger.info("Dropped database '%s' (%s).", db, code)
        dropped.append(db)
    return dropped


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def init_data(scenario_id=None, force: bool = True, reset_first: bool = False,
              managed_only: bool = False) -> dict:
    """Load a scenario's data (or the default) into CouchDB. Returns {collection: (db, n)}.

    Resolves the manifest first, so an unknown ``scenario_id`` raises FileNotFoundError
    before anything is dropped. ``reset_first=True`` then drops databases so collections
    absent from the manifest are left empty rather than carrying over.
    """
    manifest, base_dir = _resolve_manifest(scenario_id)   # validate first (raises on unknown id)
    if reset_first:
        reset(managed_only=managed_only)
    results = {}
    for key, spec in manifest.items():
        results[key] = loader.load_collection(key, spec, drop=force, base_dir=base_dir)   # database name = key
        logger.info("Scenario %s: '%s' → %s (%d docs).", scenario_id, key, *results[key])
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Load CouchDB data for a scenario (default if omitted).")
    p.add_argument("scenario", nargs="?", default=None,
                   help="Scenario id → scenarios_data/scenario_<id>.json (omit for default).")
    p.add_argument("--reuse", action="store_true", help="Reuse instead of reloading from scratch.")
    p.add_argument("--reset", action="store_true", help="Drop databases first, then load (clean start).")
    p.add_argument("--reset-only", action="store_true", help="Drop databases and exit (no load).")
    p.add_argument("--managed-only", action="store_true",
                   help="With --reset/--reset-only: drop only the default-manifest collections.")
    a = p.parse_args()

    if a.reset_only:
        for db in reset(managed_only=a.managed_only):
            print(f"dropped\t{db}")
        return

    for key, (db, n) in init_data(a.scenario, force=not a.reuse,
                                  reset_first=a.reset, managed_only=a.managed_only).items():
        print(f"{key}\t{db}\t{n}")


if __name__ == "__main__":
    main()