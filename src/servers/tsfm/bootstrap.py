"""Seed loader — populate the model & feature catalogs into a Store."""

from __future__ import annotations

import json
import os

from .core import store as store_mod
from .stores import model_store
from .stores import feature_store

# Single source of truth: the catalog lives with the other AssetOpsBench CouchDB collection data
# at src/couchdb/scenarios_data/shared/tsfm/ (the package no longer ships its own seeds/ copy).
# Override with TSFM_SEEDS_DIR when the package is relocated or for tests.
_SEEDS = os.environ.get("TSFM_SEEDS_DIR") or os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "couchdb",
        "scenarios_data",
        "shared",
        "tsfm",
    )
)


def load_seeds(store) -> dict:
    """Load the two collection files — one file == one collection, exactly like the other
    AssetOpsBench servers. model_catalog.json holds every model card (foundation + classical +
    anomaly/clustering components); feature_catalog.json holds the feature transforms + the full
    FLOps extractor library (baked). Both are CouchDB-ready (each doc carries `_id`). Source:
    src/couchdb/scenarios_data/shared/tsfm/ (or $TSFM_SEEDS_DIR)."""
    counts = {}
    mc = json.load(open(os.path.join(_SEEDS, "model_catalog.json")))
    for doc in mc:
        doc.setdefault("_id", f"model:{doc['model_id']}")
        store.put(model_store.COLLECTION, doc)
    counts["model_catalog"] = len(mc)
    fc = json.load(open(os.path.join(_SEEDS, "feature_catalog.json")))
    for doc in fc:
        doc.setdefault(
            "_id",
            f"feature:{doc.get('feature_id') or doc.get('extractor_name') or doc.get('name')}",
        )
        doc.setdefault("kind", "transform")
        store.put(feature_store.COLLECTION, doc)
    counts["feature_catalog"] = len(fc)
    return counts


def fresh_store(load: bool = True):
    """Make the configured store; seed the default catalog ONLY when it's empty.

    This is what makes the catalog a per-scenario dataset: a JSON store dir (or CouchDB) that
    already carries a catalog — even a partial one — is used as-is and never overwritten by the
    packaged seeds. An empty/new store (incl. the in-memory test store) gets the curated defaults.
    """
    s = store_mod.make_store()
    if load and not s.find(model_store.COLLECTION, limit=1):
        load_seeds(s)
    return s
