"""Generic, config-driven CouchDB collection loader.

One loader for every collection. ``collections.json`` says, per collection:
  format       : "csv" | "json"
  primary_key  : fields whose values form the deterministic _id (the CouchDB primary key)
  id_prefix    : optional _id prefix (defaults to the collection key)
  doc_type     : optional value written to each doc's ``type`` field
  design_doc   : optional design-doc JSON to install (validation + views)
  int_fields / float_fields / json_fields : declarative CSV typing
  indexes      : Mango indexes to create

Generic parsing rules (no per-collection code):
  - dotted CSV headers (a.b.c) nest into objects
  - json_fields parse JSON-valued cells (e.g. "[{...}]" -> list)
  - int_fields / float_fields coerce numerics; everything else stays a string
  - empty cells are dropped (missing columns simply don't appear)

Escape hatch: for anything the rules can't express, define a function named after the
collection in transforms.py — it's applied to each doc before the _id is computed.

The database name is always the collection key.
"""

import glob
import json
import logging
import math
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("loader")

_HERE = os.path.dirname(os.path.abspath(__file__))

COUCHDB_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
_AUTH = (os.environ.get("COUCHDB_USERNAME", "admin"), os.environ.get("COUCHDB_PASSWORD", "password"))
COLLECTIONS_FILE = os.environ.get("COLLECTIONS_CONFIG", os.path.join(_HERE, "collections.json"))
SAMPLE_DATA_DIR = os.path.join(_HERE, "sample_data")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    with open(COLLECTIONS_FILE) as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_")}  # drop "_notes" etc.


def collection_config(key: str) -> dict:
    """Config for a collection; unknown keys default to generic JSON with no primary key."""
    return load_config().get(key, {"format": "json"})


# --------------------------------------------------------------------------- #
# Parsing (generic rules)
# --------------------------------------------------------------------------- #
def _coerce(col, val, int_f, float_f, json_f):
    if col in int_f:
        return int(float(val))
    if col in float_f:
        return float(val)
    if col in json_f:
        return json.loads(val)
    return val


def _nest(doc, dotted_key, value):
    parts = dotted_key.split(".")
    d = doc
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def parse_csv(path, cfg) -> list:
    int_f = set(cfg.get("int_fields", []))
    float_f = set(cfg.get("float_fields", []))
    json_f = set(cfg.get("json_fields", []))
    df = pd.read_csv(path, dtype=str)
    rows = []
    for row in df.to_dict(orient="records"):
        doc = {}
        for col, val in row.items():
            if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
                continue
            v = _coerce(col, val, int_f, float_f, json_f)
            if "." in col:
                _nest(doc, col, v)
            else:
                doc[col] = v
        rows.append(doc)
    return rows


def parse_json(path) -> list:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _parse_file(path, cfg) -> list:
    return parse_csv(path, cfg) if cfg.get("format") == "csv" else parse_json(path)


def _collect_docs(key, source, cfg) -> list:
    """Resolve a manifest source ("default"/path/dir/list/inline docs) to parsed docs."""
    ext = ".csv" if cfg.get("format") == "csv" else ".json"

    def files_from(s):
        if s.strip().lower() == "default":
            return sorted(glob.glob(os.path.join(SAMPLE_DATA_DIR, key, "*" + ext)))
        p = s if os.path.isabs(s) else os.path.join(_HERE, s)
        if os.path.isdir(p):
            return sorted(glob.glob(os.path.join(p, "*" + ext)))
        return [p]

    docs = []
    for item in (source if isinstance(source, list) else [source]):
        if isinstance(item, dict):
            docs.append(item)                      # inline document
        elif isinstance(item, str):
            for fp in files_from(item):
                if not os.path.isfile(fp):
                    logger.warning("data file not found: %s", fp)
                    continue
                docs += _parse_file(fp, cfg)
    return docs


# --------------------------------------------------------------------------- #
# Normalisation (_id, type, dataset, transform hook)
# --------------------------------------------------------------------------- #
def _transform_for(key):
    """Optional per-collection transform: a function named <key> in transforms.py."""
    try:
        try:
            from . import transforms          # package context
        except ImportError:
            import transforms                  # script context
        return getattr(transforms, key, None)
    except Exception:
        return None


def _make_id(key, cfg, doc):
    pk = cfg.get("primary_key")
    if not pk:
        return None
    if any(doc.get(f) in (None, "") for f in pk):
        return None
    prefix = cfg.get("id_prefix", key)
    return prefix + ":" + ":".join(str(doc[f]) for f in pk)


def _normalise(doc, key, cfg, transform):
    doc.setdefault("dataset", key)
    if cfg.get("doc_type"):
        doc.setdefault("type", cfg["doc_type"])
    if transform:
        doc = transform(doc) or doc
    if "_id" not in doc:
        _id = _make_id(key, cfg, doc)
        if _id is not None:
            doc["_id"] = _id
    return doc


# --------------------------------------------------------------------------- #
# CouchDB I/O
# --------------------------------------------------------------------------- #
def _db_url(db, *parts):
    return "/".join([COUCHDB_URL.rstrip("/"), db] + list(parts))


def list_databases(include_system=False) -> list:
    r = requests.get(_db_url("_all_dbs"), auth=_AUTH, timeout=10)
    r.raise_for_status()
    dbs = r.json()
    return dbs if include_system else [d for d in dbs if not d.startswith("_")]


def drop_database(db) -> int:
    r = requests.delete(_db_url(db), auth=_AUTH, timeout=10)
    if r.status_code not in (200, 202, 404):
        r.raise_for_status()
    return r.status_code


def _ensure_db(db, drop):
    if requests.head(_db_url(db), auth=_AUTH, timeout=10).status_code == 200:
        if drop:
            requests.delete(_db_url(db), auth=_AUTH, timeout=10).raise_for_status()
        else:
            return
    requests.put(_db_url(db), auth=_AUTH, timeout=10).raise_for_status()


def _install_design(db, design_doc):
    path = design_doc if os.path.isabs(design_doc) else os.path.join(_HERE, design_doc)
    if not os.path.isfile(path):
        logger.warning("design doc not found: %s", path)
        return
    with open(path) as f:
        design = json.load(f)
    name = design.get("_id", "_design/workorders").split("/")[-1]
    url = _db_url(db, "_design", name)
    existing = requests.get(url, auth=_AUTH, timeout=10)
    if existing.status_code == 200:
        design["_rev"] = existing.json()["_rev"]
    resp = requests.put(url, json=design, auth=_AUTH, timeout=10)
    if not resp.ok:                      # surface CouchDB's actual reason (e.g. compilation_error)
        raise RuntimeError(f"design doc install failed for '{db}' ({resp.status_code}): {resp.text}")


def _create_indexes(db, indexes):
    for fields in indexes or []:
        requests.post(_db_url(db, "_index"), json={"index": {"fields": fields}, "type": "json"},
                      auth=_AUTH, timeout=10).raise_for_status()


def _bulk_insert(db, docs, batch_size=500):
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i:i + batch_size]
        r = requests.post(_db_url(db, "_bulk_docs"), json={"docs": batch}, auth=_AUTH, timeout=60)
        r.raise_for_status()
        errors = [x for x in r.json() if x.get("error")]
        if errors:
            logger.warning("%d bulk-insert errors in batch %d", len(errors), i // batch_size)
        logger.info("Inserted batch %d/%d (%d docs)", i // batch_size + 1, math.ceil(total / batch_size), len(batch))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_collection(key, source, drop=True) -> tuple:
    """Load one collection's data into a database named after the key. Returns (db, n)."""
    cfg = collection_config(key)
    transform = _transform_for(key)
    docs = [_normalise(d, key, cfg, transform) for d in _collect_docs(key, source, cfg)]
    db = key
    if docs:
        _ensure_db(db, drop=drop)
        if cfg.get("design_doc"):
            _install_design(db, cfg["design_doc"])
        _bulk_insert(db, docs)
        _create_indexes(db, cfg.get("indexes"))
    return db, len(docs)