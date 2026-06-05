"""Initialize CouchDB work-order databases from CSV files (Maximo-aligned schema).

Datasets are bound to scenarios: a scenario names a ``wo_dataset`` and that dataset
is loaded into its own database. Loading is idempotent and keyed by dataset id, so
scenarios that share a dataset reuse one database (load once), and scenarios that
bind different datasets get different databases.

Layout (one folder per dataset; each folder holds the dataset's CSVs):
    datasets/<dataset_id>/workorders.csv        (+ optional events.csv, alert_events.csv, ...)

Database naming:  workorder_<dataset_id>   (sanitized to CouchDB rules)

Each CSV row becomes a CouchDB document tagged with a ``dataset`` discriminator
(AssetOpsBench pattern). Documents use Maximo ``mxwo`` field names; the server only
reads — it never loads.

CLI:
    # load a dataset by id (idempotent; reuses the DB if already populated)
    python -m couchdb.init_wo --dataset chiller6_2017
    python -m couchdb.init_wo --dataset chiller6_2017 --force      # drop + reload
    # or point at an explicit dir / db (single-corpus mode)
    python -m couchdb.init_wo --data-dir <path> --db workorder --drop

Programmatic (for the scenario harness):
    from couchdb.init_wo import ensure_dataset
    db = ensure_dataset("chiller6_2017")     # returns the DB name to set as WO_DBNAME

Environment (or .env):
    COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD
    WO_DBNAME           default DB for single-corpus mode (default: workorder)
    WO_DATASETS_ROOT    root holding dataset folders (default: ./datasets)
"""

import argparse
import json
import logging
import math
import os
import re
import sys

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(__file__)

COUCHDB_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME", "admin")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD", "password")
WO_DBNAME = os.environ.get("WO_DBNAME", "workorder")
WO_DATASETS_ROOT = os.environ.get("WO_DATASETS_ROOT", os.path.join(_SCRIPT_DIR, "datasets"))
# Base dataset loaded under every scenario; scenario docs override it on _id overlap.
WO_DEFAULT_DATASET = os.environ.get("WO_DEFAULT_DATASET", "default")
_AUTH = (COUCHDB_USERNAME, COUCHDB_PASSWORD)

# (csv_filename, dataset key). Work orders are primary; add more CSVs per dataset here.
_DATASETS = [
    ("workorders.csv", "wo_events"),
]

_INDEXES = [
    ["type", "siteid", "status"],
    ["type", "assetnum", "reportdate"],
    ["type", "worktype", "wopriority"],
    ["type", "wonum"],
    ["type", "aob_source.scenario_id"],
]

_INT_COLS = {"wopriority", "taskid"}
_FLOAT_COLS = {
    "estlabhrs", "actlabhrs", "estlabcost", "actlabcost", "estmatcost", "actmatcost",
    "estservcost", "actservcost", "esttoolcost", "acttoolcost", "estatapprtotalcost",
    "esttotalcost", "acttotalcost", "aob_source.evidence.anomaly_score",
    "aob_source.evidence.threshold", "aob_source.evidence.observed_value",
}
_JSON_COLS = {"wplabor"}


# --------------------------------------------------------------------------- #
# Dataset <-> database name
# --------------------------------------------------------------------------- #
def dataset_db_name(dataset_id: str) -> str:
    """CouchDB-legal database name for a dataset id (start lowercase; a-z0-9_$()+-).

    '/' is intentionally excluded (it breaks the URL path) even though CouchDB allows it.
    """
    name = re.sub(r"[^a-z0-9_$()+-]", "_", f"workorder_{dataset_id}".lower())
    return name if name[:1].isalpha() else "wo_" + name


def dataset_dir(dataset_id: str) -> str:
    return os.path.join(WO_DATASETS_ROOT, dataset_id)


# --------------------------------------------------------------------------- #
# Pure CSV -> docs (no network; unit-testable)
# --------------------------------------------------------------------------- #
def _coerce(col: str, val):
    if col in _INT_COLS:
        return int(float(val))
    if col in _FLOAT_COLS:
        return float(val)
    if col in _JSON_COLS:
        return json.loads(val)
    return val


def _nest(doc: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    d = doc
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def build_docs(csv_path: str, dataset: str = "wo_events") -> list:
    """Read a CSV of Maximo work orders and return CouchDB documents."""
    df = pd.read_csv(csv_path, dtype=str)
    docs = []
    for row in df.to_dict(orient="records"):
        doc = {"dataset": dataset, "type": "workorder", "schema_version": "1.0.0"}
        for col, val in row.items():
            if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
                continue
            v = _coerce(col, val)
            if "." in col:
                _nest(doc, col, v)
            else:
                doc[col] = v
        if "wonum" in doc and "siteid" in doc:
            doc["_id"] = f"wo:{str(doc['siteid']).upper()}:{doc['wonum']}"
        docs.append(doc)
    return docs


# --------------------------------------------------------------------------- #
# CouchDB I/O
# --------------------------------------------------------------------------- #
def _db_url(db: str, *parts: str) -> str:
    return "/".join([COUCHDB_URL.rstrip("/"), db] + list(parts))


def _doc_count(db_name: str) -> int:
    """Number of non-design docs, or -1 if the database doesn't exist."""
    resp = requests.get(_db_url(db_name), auth=_AUTH, timeout=10)
    if resp.status_code != 200:
        return -1
    info = resp.json()
    return int(info.get("doc_count", 0))  # includes _design docs; good enough for an emptiness check


def _ensure_db(db_name: str, drop: bool) -> None:
    url = _db_url(db_name)
    resp = requests.head(url, auth=_AUTH, timeout=10)
    if resp.status_code == 200:
        if drop:
            logger.info("Dropping existing database '%s'…", db_name)
            requests.delete(url, auth=_AUTH, timeout=10).raise_for_status()
        else:
            return
    logger.info("Creating database '%s'…", db_name)
    requests.put(url, auth=_AUTH, timeout=10).raise_for_status()


def _install_design_doc(db_name: str) -> None:
    path = os.path.join(_SCRIPT_DIR, "_design_workorders.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        design = json.load(f)
    url = _db_url(db_name, "_design", "workorders")
    existing = requests.get(url, auth=_AUTH, timeout=10)
    if existing.status_code == 200:
        design["_rev"] = existing.json()["_rev"]
    requests.put(url, json=design, auth=_AUTH, timeout=10).raise_for_status()


def _create_indexes(db_name: str) -> None:
    url = _db_url(db_name, "_index")
    for fields in _INDEXES:
        requests.post(url, json={"index": {"fields": fields}, "type": "json"}, auth=_AUTH, timeout=10).raise_for_status()


def _bulk_insert(db_name: str, docs: list, batch_size: int = 500) -> None:
    url = _db_url(db_name, "_bulk_docs")
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i:i + batch_size]
        resp = requests.post(url, json={"docs": batch}, auth=_AUTH, timeout=60)
        resp.raise_for_status()
        errors = [r for r in resp.json() if r.get("error")]
        if errors:
            logger.warning("%d bulk-insert errors in batch %d", len(errors), i // batch_size)
        logger.info("Inserted batch %d/%d (%d docs)", i // batch_size + 1, math.ceil(total / batch_size), len(batch))


def _collect_docs(data_dir: str) -> list:
    """Build all docs from a dataset directory's CSVs (no DB I/O)."""
    docs: list = []
    for csv_file, dataset in _DATASETS:
        path = os.path.join(data_dir, csv_file)
        if not os.path.exists(path):
            logger.warning("CSV not found, skipping: %s", path)
            continue
        rows = build_docs(path, dataset)
        logger.info("Loaded %d rows from '%s' → dataset '%s'", len(rows), csv_file, dataset)
        docs.extend(rows)
    return docs


def merge_by_id(*doclists: list) -> list:
    """Merge doc lists by ``_id``; later lists win (scenario overrides default).

    On overlap the earlier (default) document is replaced entirely by the later
    (scenario) one — the old version is not kept.
    """
    merged: dict = {}
    for docs in doclists:
        for d in docs:
            merged[d.get("_id", id(d))] = d
    return list(merged.values())


def _write_db(db_name: str, docs: list, drop: bool) -> int:
    if not docs:
        return 0
    _ensure_db(db_name, drop=drop)
    _install_design_doc(db_name)
    _bulk_insert(db_name, docs)
    _create_indexes(db_name)
    return len(docs)


def _load_dir_into_db(data_dir: str, db_name: str, drop: bool = False) -> int:
    return _write_db(db_name, _collect_docs(data_dir), drop=drop)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def ensure_dataset(dataset_id: str, force: bool = False) -> str:
    """Ensure DB = default dataset + scenario dataset (scenario overrides on _id overlap).

    The effective database for a scenario is the base ``WO_DEFAULT_DATASET`` with the
    scenario's docs layered on top: any work order whose _id appears in both is taken
    from the scenario (the default's version is dropped). Idempotent and keyed by
    dataset id — set the returned name as WO_DBNAME for the spawned server.
    """
    db_name = dataset_db_name(dataset_id)
    if not force and _doc_count(db_name) > 1:  # >1 ⇒ already populated
        logger.info("Dataset '%s' already loaded in '%s' — reusing.", dataset_id, db_name)
        return db_name

    layers: list = []
    # base default layer (skipped when the scenario *is* the default, or no default dir)
    if dataset_id != WO_DEFAULT_DATASET and os.path.isdir(dataset_dir(WO_DEFAULT_DATASET)):
        base = _collect_docs(dataset_dir(WO_DEFAULT_DATASET))
        logger.info("Default layer '%s': %d docs", WO_DEFAULT_DATASET, len(base))
        layers.append(base)
    # scenario layer (wins on overlap)
    scen_dir = dataset_dir(dataset_id)
    if not os.path.isdir(scen_dir):
        raise FileNotFoundError(f"dataset '{dataset_id}' not found at {scen_dir}")
    scen = _collect_docs(scen_dir)
    logger.info("Scenario layer '%s': %d docs", dataset_id, len(scen))
    layers.append(scen)

    merged = merge_by_id(*layers)
    if not merged:
        raise ValueError(f"dataset '{dataset_id}' produced no documents")
    overlap = sum(len(x) for x in layers) - len(merged)
    n = _write_db(db_name, merged, drop=True)
    logger.info("Dataset '%s' → '%s': %d docs (%d overridden from default).",
                dataset_id, db_name, n, overlap)
    return db_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize CouchDB work-order database(s) from CSVs.")
    parser.add_argument("--dataset", help="Dataset id (loads datasets/<id>/ into workorder_<id>)")
    parser.add_argument("--force", action="store_true", help="With --dataset: drop + reload even if present")
    parser.add_argument("--data-dir", help="Explicit CSV dir (single-corpus mode)")
    parser.add_argument("--db", default=WO_DBNAME, help="DB name for single-corpus mode")
    parser.add_argument("--drop", action="store_true", help="Single-corpus mode: drop + recreate")
    args = parser.parse_args()

    logger.info("CouchDB URL: %s", COUCHDB_URL)
    if args.dataset:
        db = ensure_dataset(args.dataset, force=args.force)
        print(db)  # so a harness can capture it: WO_DBNAME=$(python -m couchdb.init_wo --dataset X)
        return

    data_dir = args.data_dir or os.path.join(_SCRIPT_DIR, "sample_data", "work_order")
    _ensure_db(args.db, drop=args.drop)
    n = _load_dir_into_db(data_dir, args.db)
    if n == 0:
        logger.error("No documents to insert — check --data-dir path.")
        sys.exit(1)
    logger.info("Done. Database '%s' is ready (%d docs).", args.db, n)


if __name__ == "__main__":
    main()