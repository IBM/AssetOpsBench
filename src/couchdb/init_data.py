"""Entry point: read a scenario's JSON manifest and load its data into CouchDB.

init_data reads the manifest and hands each collection to its existing loader:

    manifest["work_order"]  →  init_wo            (the 'workorder' database)
    manifest["iot"]         →  init_asset_data    (the 'iot' database)
    manifest["vibration"]   →  init_asset_data    (the 'vibration' database)

    python -m couchdb.init_data 42      # load scenario 42's data
    python -m couchdb.init_data         # no scenario → default data (also used at container startup)

    from couchdb.init_data import init_data
    init_data(42)     # scenario 42
    init_data()       # default

Manifest at scenarios_data/scenario_<id>.json, e.g.::

    {"work_order": "sample_data/work_order/workorders.csv",
     "iot": ["sample_data/iot/chiller_6.json", "sample_data/iot/metro_pump_1.json"]}

If the scenario's ``dataset`` field is ``default`` (or it has no manifest), the
DEFAULT_MANIFEST below is loaded — the single definition of "default" data, used by
couchdb_setup.sh at container startup too. Databases are rebuilt from scratch.
"""

import argparse
import glob
import json
import logging
import os

from dotenv import load_dotenv

try:                       # works as a package (python -m couchdb.init_data / imports)
    from . import init_asset_data, init_wo
except ImportError:        # works as a script (python3 /couchdb/init_data.py)
    import init_asset_data
    import init_wo

load_dotenv()

logger = logging.getLogger("init_data")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

SCENARIOS_DATA_DIR = os.environ.get("WO_SCENARIOS_DATA_DIR", os.path.join(_HERE, "scenarios_data"))
WO_DEFAULT_DATASET = os.environ.get("WO_DEFAULT_DATASET", "default")
WO_SCENARIO_FIELD = os.environ.get("WO_SCENARIO_FIELD", "dataset")
WO_SCENARIOS_FILE = os.environ.get(
    "WO_SCENARIOS_FILE", os.path.join(_REPO_ROOT, "scenarios", "huggingface", "all_utterance.jsonl"))
VIBRATION_DBNAME = os.environ.get("VIBRATION_DBNAME", "vibration")

# The single definition of the default data (what couchdb_setup.sh used to hardcode).
DEFAULT_MANIFEST = {
    "work_order": "default",
    "iot": [
        "sample_data/iot/chiller_6.json",
        "sample_data/iot/metro_pump_1.json",
        "sample_data/iot/hydraulic_pump_1.json",
    ],
    "vibration": "sample_data/iot/motor_01.json",
}


# --------------------------------------------------------------------------- #
# Collection loaders (delegate to the existing per-collection modules)
# --------------------------------------------------------------------------- #
def _load_work_order(spec, drop) -> tuple:
    return init_wo.load_work_order(spec, drop=drop)   # init_wo handles CSV/JSON/dir/"default"


def _asset_files(spec) -> list:
    """Resolve a sensor-data spec ("default" / path / dir / list) to JSON file paths."""
    def resolve(s):
        if not isinstance(s, str):
            return []
        if s.strip().lower() == "default":
            return [init_asset_data.ASSET_DATA_FILE]
        p = s if os.path.isabs(s) else os.path.join(_HERE, s)   # relative to couchdb/
        if os.path.isdir(p):
            return sorted(glob.glob(os.path.join(p, "*.json")))
        return [p]
    if isinstance(spec, str):
        return resolve(spec)
    if isinstance(spec, list):
        out = []
        for item in spec:
            out += resolve(item)
        return out
    return []


def _load_asset(spec, drop, db: str) -> tuple:
    """Read sensor JSON file(s) and load into a database via init_asset_data's helpers."""
    docs = []
    for fp in _asset_files(spec):
        if not os.path.isfile(fp):
            logger.warning("data file not found: %s", fp)
            continue
        with open(fp) as f:
            data = json.load(f)
        docs += data if isinstance(data, list) else [data]
    if docs:
        init_asset_data._ensure_db(db, drop)
        init_asset_data._bulk_insert(db, docs)
        init_asset_data._create_indexes(db)
    return db, len(docs)


def _load_iot(spec, drop) -> tuple:
    return _load_asset(spec, drop, init_asset_data.IOT_DBNAME)


def _load_vibration(spec, drop) -> tuple:
    return _load_asset(spec, drop, VIBRATION_DBNAME)


LOADERS = {
    "work_order": _load_work_order,
    "iot": _load_iot,
    "vibration": _load_vibration,
}


# --------------------------------------------------------------------------- #
# Scenario → manifest
# --------------------------------------------------------------------------- #
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
        return DEFAULT_MANIFEST
    row = _scenario_row(scenario_id, scenarios_path)
    dataset_value = row.get(WO_SCENARIO_FIELD)
    path = None if _is_default(dataset_value) else _manifest_path(scenario_id, dataset_value)
    if path is None:
        return DEFAULT_MANIFEST
    with open(path) as f:
        return json.load(f)


def init_data(scenario_id=None, scenarios_path: str = None, force: bool = True) -> dict:
    """Load a scenario's data (or the default) into CouchDB. Returns {collection: (db, n)}."""
    manifest = _resolve_manifest(scenario_id, scenarios_path)
    results = {}
    for key, spec in manifest.items():
        loader = LOADERS.get(key)
        if loader is None:
            logger.warning("No loader for collection '%s' — skipping.", key)
            continue
        results[key] = loader(spec, force)
        logger.info("Scenario %s: '%s' → %s (%d docs).", scenario_id, key, *results[key])
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Load CouchDB data for a scenario (default if omitted).")
    p.add_argument("scenario", nargs="?", default=None, help="Scenario id (omit to load default data).")
    p.add_argument("--scenarios", default=None, help="Scenarios .jsonl path (else WO_SCENARIOS_FILE).")
    p.add_argument("--reuse", action="store_true", help="Reuse instead of reloading from scratch.")
    a = p.parse_args()
    for key, (db, n) in init_data(a.scenario, scenarios_path=a.scenarios, force=not a.reuse).items():
        print(f"{key}\t{db}\t{n}")


if __name__ == "__main__":
    main()