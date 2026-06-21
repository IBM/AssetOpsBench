"""Seed loader — populate the model & feature catalogs into a Store."""

from __future__ import annotations

import glob
import json
import os

from tsfm.core import store as store_mod
from tsfm.stores import model_store
from tsfm.stores import feature_store

_SEEDS = os.path.join(os.path.dirname(__file__), "seeds")


def load_seeds(store) -> dict:
    counts = {"model_catalog": 0, "feature_catalog": 0}
    for path in glob.glob(os.path.join(_SEEDS, "model_*.json")) + \
                glob.glob(os.path.join(_SEEDS, "anomalykits_*.json")):
        for doc in json.load(open(path)):
            doc.setdefault("_id", f"model:{doc['model_id']}")
            store.put(model_store.COLLECTION, doc)
            counts["model_catalog"] += 1
    for path in glob.glob(os.path.join(_SEEDS, "feature_*.json")):
        for doc in json.load(open(path)):
            doc.setdefault("_id", f"feature:{doc['feature_id']}")
            doc.setdefault("kind", "transform")
            store.put(feature_store.COLLECTION, doc)
            counts["feature_catalog"] += 1
    counts["extractors"] = feature_store.register_extractor_library(store)   # FLOps library
    return counts


def fresh_store(load: bool = True):
    s = store_mod.make_store()
    if load:
        load_seeds(s)
    return s
