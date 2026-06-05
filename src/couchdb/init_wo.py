"""Low-level CouchDB work-order loader (Maximo-aligned schema).

Building blocks shared by init_scenario.py and usable directly for single-corpus
loading. Reads CSVs with pandas, turns each row into a CouchDB document tagged with
a ``dataset`` discriminator (AssetOpsBench pattern), batched ``_bulk_docs`` insert,
then Mango indexes. Documents use Maximo ``mxwo`` field names; the server only reads.

Scenario-bound, per-dataset initialization lives in init_scenario.py.

CLI (single-corpus mode — load one CSV dir into one DB):
    python -m couchdb.init_wo --data-dir <path> --db workorder --drop

Environment (or .env):
    COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD, WO_DBNAME
"""

import argparse
import json
import logging
import math
import os
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
_AUTH = (COUCHDB_USERNAME, COUCHDB_PASSWORD)

# (csv_filename, dataset key). Work orders are primary; add more CSVs here.
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


def doc_count(db_name: str) -> int:
    """Number of docs (incl. design docs), or -1 if the database doesn't exist."""
    resp = requests.get(_db_url(db_name), auth=_AUTH, timeout=10)
    if resp.status_code != 200:
        return -1
    return int(resp.json().get("doc_count", 0))


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


def collect_docs(data_dir: str) -> list:
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


def write_docs(db_name: str, docs: list, drop: bool = False) -> int:
    """Write a list of work-order docs to a database (design doc + indexes)."""
    if not docs:
        return 0
    _ensure_db(db_name, drop=drop)
    _install_design_doc(db_name)
    _bulk_insert(db_name, docs)
    _create_indexes(db_name)
    return len(docs)


def load_dir_into_db(data_dir: str, db_name: str, drop: bool = False) -> int:
    """Build docs from a CSV dir and write them to a database."""
    return write_docs(db_name, collect_docs(data_dir), drop=drop)


def _normalize(doc: dict, dataset: str) -> dict:
    """Ensure a doc has dataset/type/_id (so JSON-supplied docs match CSV-built ones)."""
    doc = dict(doc)
    doc.setdefault("dataset", dataset)
    doc.setdefault("type", "workorder")
    doc.setdefault("schema_version", "1.0.0")
    if "_id" not in doc and doc.get("wonum") and doc.get("siteid"):
        doc["_id"] = f"wo:{str(doc['siteid']).upper()}:{doc['wonum']}"
    return doc


# Accepted keys for the work-order collection inside a scenario JSON object.
_WO_KEYS = ("work_order", "workorders", "wo_events", "wo")


def docs_from_json(path: str, dataset: str = "wo_events") -> list:
    """Load work-order docs from a scenario JSON file.

    Accepts either a bare list of docs, or an object like
    ``{"work_order": [ ...docs... ], "events": [...]}`` (the work-order collection is
    pulled from any of: work_order / workorders / wo_events / wo).
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = next((data[k] for k in _WO_KEYS if k in data), [])
    else:
        rows = []
    return [_normalize(d, dataset) for d in rows]


# Default WO data directory (used when a manifest's work_order value is "default").
DEFAULT_WO_DIR = os.path.join(_SCRIPT_DIR, "sample_data", "work_order")


def _resolve_source(src: str, dataset: str = "wo_events") -> list:
    """Resolve one work_order source string to docs: 'default', a dir, a .csv, or a .json."""
    if src.strip().lower() == "default":
        return collect_docs(DEFAULT_WO_DIR)
    target = src if os.path.isabs(src) else os.path.join(_SCRIPT_DIR, src)  # relative to couchdb/
    if os.path.isdir(target):
        return collect_docs(target)
    if target.endswith(".csv") and os.path.isfile(target):
        return build_docs(target, dataset)
    if target.endswith(".json") and os.path.isfile(target):
        return docs_from_json(target, dataset)
    raise FileNotFoundError(f"work_order source not found: {src!r}")


def docs_from_spec(spec, dataset: str = "wo_events") -> list:
    """A manifest's ``work_order`` value → docs.

    ``spec`` may be a path string ("default", a .csv/.json file, or a dir), a list of
    such paths (concatenated), or a list of inline document objects.
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        return _resolve_source(spec, dataset)
    if isinstance(spec, list):
        docs = []
        for item in spec:
            if isinstance(item, dict):
                docs.append(_normalize(item, dataset))
            elif isinstance(item, str):
                docs += _resolve_source(item, dataset)
        return docs
    return []


def load_work_order(spec, db_name: str = None, drop: bool = True) -> tuple:
    """Load a manifest's ``work_order`` data into the WO database. Returns (db_name, n_docs)."""
    db_name = db_name or WO_DBNAME
    n = write_docs(db_name, docs_from_spec(spec), drop=drop)
    return db_name, n


def main() -> None:
    parser = argparse.ArgumentParser(description="Load one CSV directory into one CouchDB work-order database.")
    parser.add_argument("--data-dir", help="CSV directory (default: sample_data/work_order)")
    parser.add_argument("--db", default=WO_DBNAME, help="Target database name")
    parser.add_argument("--drop", action="store_true", help="Drop + recreate the database first")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(_SCRIPT_DIR, "sample_data", "work_order")
    logger.info("CouchDB URL: %s | DB: %s | Data dir: %s", COUCHDB_URL, args.db, data_dir)
    n = load_dir_into_db(data_dir, args.db, drop=args.drop)
    if n == 0:
        logger.error("No documents to insert — check --data-dir path.")
        sys.exit(1)
    logger.info("Done. Database '%s' is ready (%d docs).", args.db, n)


if __name__ == "__main__":
    main()