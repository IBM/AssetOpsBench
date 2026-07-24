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
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("loader")

_HERE = os.path.dirname(os.path.abspath(__file__))

COUCHDB_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
_AUTH = (
    os.environ.get("COUCHDB_USERNAME", "admin"),
    os.environ.get("COUCHDB_PASSWORD", "password"),
)
COLLECTIONS_FILE = os.environ.get(
    "COLLECTIONS_CONFIG", os.path.join(_HERE, "collections.json")
)
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
            if (
                val is None
                or (isinstance(val, float) and pd.isna(val))
                or str(val).strip() == ""
            ):
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


def _collect_docs(key, source, cfg, base_dir=None) -> list:
    """Resolve a manifest source ("default"/path/dir/list/inline docs) to parsed docs.

    Relative data paths resolve against the scenario folder (``base_dir``), then its parent
    (so a sibling ``shared/`` corpus is reachable as ``shared/...``), then ``_HERE``. With
    base_dir=None (legacy flat manifests) only ``_HERE`` is used, preserving behaviour.
    """
    roots = ([base_dir, os.path.dirname(base_dir)] if base_dir else []) + [_HERE]
    ext = ".csv" if cfg.get("format") == "csv" else ".json"

    def _resolve(s):
        for root in roots:
            p = os.path.join(root, s)
            if os.path.exists(p):
                return p
        return os.path.join(roots[0], s)

    def files_from(s):
        if s.strip().lower() == "default":
            return sorted(glob.glob(os.path.join(SAMPLE_DATA_DIR, key, "*" + ext)))
        p = s if os.path.isabs(s) else _resolve(s)
        if os.path.isdir(p):
            return sorted(glob.glob(os.path.join(p, "*" + ext)))
        return [p]

    docs = []
    for item in source if isinstance(source, list) else [source]:
        if isinstance(item, dict):
            docs.append(item)  # inline document
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
            from . import transforms  # package context
        except ImportError:
            import transforms  # script context
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


_IOT_RESERVED_FIELDS = {
    "_id",
    "_rev",
    "asset_id",
    "timestamp",
    "dataset",
    "type",
    "doctype",
}


def _parse_iot_timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _iot_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _empty_iot_summary(asset_id):
    return {
        "_id": f"iot_summary:{asset_id}",
        "dataset": "iot",
        "doctype": "iot_asset_summary",
        "summary_asset_id": asset_id,
        "timestamped_records": 0,
        "invalid_timestamp_count": 0,
        "mixed_timezone_awareness": False,
        "start_time": None,
        "end_time": None,
        "sensors": set(),
        "coverage": {},
        "stats": {},
        "latest": None,
    }


def _add_iot_stat(stat, value, timestamp, timestamp_dt):
    number = _iot_number(value)
    if number is None:
        stat["null_count"] += 1
        return
    stat["count"] += 1
    stat["sum"] += number
    stat["sumsq"] += number * number
    stat["min"] = number if stat["min"] is None else min(stat["min"], number)
    stat["max"] = number if stat["max"] is None else max(stat["max"], number)
    if stat.get("_first_dt") is None or timestamp_dt < stat["_first_dt"]:
        stat["_first_dt"] = timestamp_dt
        stat["first_timestamp"] = timestamp
    if stat.get("_last_dt") is None or timestamp_dt > stat["_last_dt"]:
        stat["_last_dt"] = timestamp_dt
        stat["last_timestamp"] = timestamp


def _finalise_iot_stat(stat):
    count = int(stat["count"])
    mean = None
    stddev = None
    if count:
        mean = stat["sum"] / count
        variance = (stat["sumsq"] / count) - (mean * mean)
        if math.isfinite(variance):
            stddev = math.sqrt(max(variance, 0.0))
    return {
        "count": count,
        "null_count": int(stat["null_count"]),
        "min": stat["min"] if count else None,
        "max": stat["max"] if count else None,
        "mean": mean,
        "stddev": stddev,
        "first_timestamp": stat["first_timestamp"],
        "last_timestamp": stat["last_timestamp"],
    }


def _make_iot_summary_docs(docs):
    summaries = {}
    for doc in docs:
        if doc.get("doctype") == "iot_asset_summary":
            continue
        asset_id = doc.get("asset_id")
        timestamp = doc.get("timestamp")
        if not asset_id or timestamp is None:
            continue
        summary = summaries.setdefault(asset_id, _empty_iot_summary(asset_id))
        timestamp_dt = _parse_iot_timestamp(timestamp)
        if timestamp_dt is None:
            summary["invalid_timestamp_count"] += 1
            continue
        timestamp_is_aware = timestamp_dt.utcoffset() is not None
        if summary.get("_timestamp_is_aware") is None:
            summary["_timestamp_is_aware"] = timestamp_is_aware
        elif summary["_timestamp_is_aware"] != timestamp_is_aware:
            summary["mixed_timezone_awareness"] = True
            continue

        summary["timestamped_records"] += 1
        if summary.get("_first_dt") is None or timestamp_dt < summary["_first_dt"]:
            summary["_first_dt"] = timestamp_dt
            summary["start_time"] = timestamp
        if summary.get("_last_dt") is None or timestamp_dt > summary["_last_dt"]:
            summary["_last_dt"] = timestamp_dt
            summary["end_time"] = timestamp
            summary["latest"] = {
                "timestamp": timestamp,
                "values": {
                    field: value
                    for field, value in doc.items()
                    if field not in _IOT_RESERVED_FIELDS
                },
            }

        for field, value in doc.items():
            if field in _IOT_RESERVED_FIELDS:
                continue
            summary["sensors"].add(field)
            coverage = summary["coverage"].setdefault(
                field,
                {
                    "non_null_count": 0,
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "_first_dt": None,
                    "_last_dt": None,
                    "latest_timestamp": None,
                    "latest_value": None,
                    "_latest_dt": None,
                },
            )
            if value is not None:
                coverage["non_null_count"] += 1
                if coverage["_first_dt"] is None or timestamp_dt < coverage["_first_dt"]:
                    coverage["_first_dt"] = timestamp_dt
                    coverage["first_timestamp"] = timestamp
                if coverage["_last_dt"] is None or timestamp_dt > coverage["_last_dt"]:
                    coverage["_last_dt"] = timestamp_dt
                    coverage["last_timestamp"] = timestamp
                if coverage["_latest_dt"] is None or timestamp_dt > coverage["_latest_dt"]:
                    coverage["_latest_dt"] = timestamp_dt
                    coverage["latest_timestamp"] = timestamp
                    coverage["latest_value"] = value

            stat = summary["stats"].setdefault(
                field,
                {
                    "count": 0,
                    "null_count": 0,
                    "min": None,
                    "max": None,
                    "sum": 0.0,
                    "sumsq": 0.0,
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "_first_dt": None,
                    "_last_dt": None,
                },
            )
            _add_iot_stat(stat, value, timestamp, timestamp_dt)

    out = []
    for summary in summaries.values():
        summary["sensors"] = sorted(summary["sensors"])
        summary["coverage"] = {
            sensor: {
                key: value
                for key, value in coverage.items()
                if not key.startswith("_")
            }
            for sensor, coverage in sorted(summary["coverage"].items())
        }
        summary["stats"] = {
            sensor: _finalise_iot_stat(stat)
            for sensor, stat in sorted(summary["stats"].items())
        }
        summary.pop("_first_dt", None)
        summary.pop("_last_dt", None)
        summary.pop("_timestamp_is_aware", None)
        out.append(summary)
    return out


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
    if not resp.ok:  # surface CouchDB's actual reason (e.g. compilation_error)
        raise RuntimeError(
            f"design doc install failed for '{db}' ({resp.status_code}): {resp.text}"
        )


def _create_indexes(db, indexes):
    for fields in indexes or []:
        requests.post(
            _db_url(db, "_index"),
            json={"index": {"fields": fields}, "type": "json"},
            auth=_AUTH,
            timeout=10,
        ).raise_for_status()


def _bulk_insert(db, docs, batch_size=500):
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        r = requests.post(
            _db_url(db, "_bulk_docs"), json={"docs": batch}, auth=_AUTH, timeout=60
        )
        r.raise_for_status()
        errors = [x for x in r.json() if x.get("error")]
        if errors:
            logger.warning(
                "%d bulk-insert errors in batch %d", len(errors), i // batch_size
            )
        logger.info(
            "Inserted batch %d/%d (%d docs)",
            i // batch_size + 1,
            math.ceil(total / batch_size),
            len(batch),
        )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_collection(key, source, drop=True, base_dir=None) -> tuple:
    """Load one collection's data into a database named after the key. Returns (db, n)."""
    cfg = collection_config(key)
    transform = _transform_for(key)
    docs = [
        _normalise(d, key, cfg, transform)
        for d in _collect_docs(key, source, cfg, base_dir)
    ]
    docs_to_insert = list(docs)
    if key == "iot" and docs:
        docs_to_insert.extend(_make_iot_summary_docs(docs))
    db = key
    if docs:
        _ensure_db(db, drop=drop)
        if cfg.get("design_doc"):
            _install_design(db, cfg["design_doc"])
        _bulk_insert(db, docs_to_insert)
        _create_indexes(db, cfg.get("indexes"))
    return db, len(docs)
