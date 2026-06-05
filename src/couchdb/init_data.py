"""Entry point: read a scenario's JSON manifest and load its data into CouchDB.

init_data resolves which manifest to use (the scenario's, or the default) and hands
each collection to the generic, config-driven loader (loader.py). Every manifest key
becomes a database of the same name; how each is parsed/keyed/indexed comes from
collections.json.

    python -m couchdb.init_data 42      # load scenario 42's data
    python -m couchdb.init_data         # no scenario → default data (also used at container startup)

    from couchdb.init_data import init_data
    init_data(42)     # scenario 42
    init_data()       # default

Manifest at scenarios_data/scenario_<id>.json (and scenarios_data/default.json), e.g.::

    {"work_order": "sample_data/work_order/workorders.csv",
     "iot": ["sample_data/iot/chiller_6.json", "sample_data/iot/metro_pump_1.json"]}

If the scenario's ``dataset`` field is ``default`` (or it has no manifest), the default
manifest is loaded. Databases are rebuilt from scratch each call.
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
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

SCENARIOS_DATA_DIR = os.environ.get("WO_SCENARIOS_DATA_DIR", os.path.join(_HERE, "scenarios_data"))
WO_DEFAULT_DATASET = os.environ.get("WO_DEFAULT_DATASET", "default")
WO_SCENARIO_FIELD = os.environ.get("WO_SCENARIO_FIELD", "dataset")
WO_SCENARIOS_FILE = os.environ.get(
    "WO_SCENARIOS_FILE", os.path.join(_REPO_ROOT, "scenarios", "huggingface", "all_utterance.jsonl"))
DEFAULT_MANIFEST_FILE = os.environ.get(
    "WO_DEFAULT_MANIFEST", os.path.join(SCENARIOS_DATA_DIR, "default.json"))


# --------------------------------------------------------------------------- #
# Scenario → manifest
# --------------------------------------------------------------------------- #
def _load_default_manifest() -> dict:
    if not os.path.isfile(DEFAULT_MANIFEST_FILE):
        raise FileNotFoundError(
            f"default manifest not found: {DEFAULT_MANIFEST_FILE}. "
            "Create scenarios_data/default.json (or set WO_DEFAULT_MANIFEST).")
    with open(DEFAULT_MANIFEST_FILE) as f:
        return json.load(f)


def _is_default(dataset_value) -> bool:
    return dataset_value is None or str(dataset_value).strip() in ("", WO_DEFAULT_DATASET)


def _scenario_row(scenario_id, scenarios_path=None) -> dict:
    path = scenarios_path or WO_SCENARIOS_FILE
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"scenarios file not found: {path!r}. Set WO_SCENARIOS_FILE.")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and str(json.loads(line).get("id")) == str(scenario_id):
                return json.loads(line)
    raise KeyError(f"scenario id {scenario_id} not found in {path}")


def _manifest_path(scenario_id, dataset_value):
    candidates = []
    if dataset_value and not _is_default(dataset_value):
        candidates.append(os.path.join(SCENARIOS_DATA_DIR, f"{dataset_value}.json"))
    candidates.append(os.path.join(SCENARIOS_DATA_DIR, f"scenario_{scenario_id}.json"))
    return next((c for c in candidates if os.path.isfile(c)), None)


def _resolve_manifest(scenario_id, scenarios_path=None) -> dict:
    if scenario_id is None:
        return _load_default_manifest()
    row = _scenario_row(scenario_id, scenarios_path)
    dataset_value = row.get(WO_SCENARIO_FIELD)
    path = None if _is_default(dataset_value) else _manifest_path(scenario_id, dataset_value)
    if path is None:
        return _load_default_manifest()
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #
def all_databases() -> list:
    """The databases this loader manages = the default manifest's keys (db name = key)."""
    try:
        return list(_load_default_manifest().keys())
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
def init_data(scenario_id=None, scenarios_path: str = None, force: bool = True,
              reset_first: bool = False, managed_only: bool = False) -> dict:
    """Load a scenario's data (or the default) into CouchDB. Returns {collection: (db, n)}.

    ``reset_first=True`` drops databases first so collections absent from the manifest
    are left empty rather than carrying over.
    """
    if reset_first:
        reset(managed_only=managed_only)
    manifest = _resolve_manifest(scenario_id, scenarios_path)
    results = {}
    for key, spec in manifest.items():
        results[key] = loader.load_collection(key, spec, drop=force)   # database name = key
        logger.info("Scenario %s: '%s' → %s (%d docs).", scenario_id, key, *results[key])
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Load CouchDB data for a scenario (default if omitted).")
    p.add_argument("scenario", nargs="?", default=None, help="Scenario id (omit to load default data).")
    p.add_argument("--scenarios", default=None, help="Scenarios .jsonl path (else WO_SCENARIOS_FILE).")
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

    for key, (db, n) in init_data(a.scenario, scenarios_path=a.scenarios, force=not a.reuse,
                                  reset_first=a.reset, managed_only=a.managed_only).items():
        print(f"{key}\t{db}\t{n}")


if __name__ == "__main__":
    main()