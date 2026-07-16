"""Test config — keep the suite hermetic.

Production uses CouchDB (TSFM_STORE=couch, like the other AssetOpsBench servers). Tests force the
in-memory backend so every run starts from the curated seeds with no service dependency — the
same skipif-on-missing-service pattern the sibling servers use for their CouchDB integration tests.
"""

import json
import os

import pytest

os.environ["TSFM_STORE"] = "memory"

# Production reads the catalogs from CouchDB (databases model_catalog / feature_catalog, loaded by
# src/couchdb/init_data.py). The server no longer seeds itself, so the suite keeps its own hermetic
# copy of the shipped seeds and needs no running service.
_SEEDS = os.environ.get("TSFM_SEEDS_DIR") or os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "couchdb", "scenarios_data", "shared", "tsfm",
    )
)


def _seed_into(s):
    """Load the shipped catalog seeds into a store (test-only)."""
    from ..stores import feature_store, model_store

    for doc in json.load(open(os.path.join(_SEEDS, "model_catalog.json"))):
        doc.setdefault("_id", f"model:{doc['model_id']}")
        s.put(model_store.COLLECTION, doc)
    for doc in json.load(open(os.path.join(_SEEDS, "feature_catalog.json"))):
        doc.setdefault(
            "_id",
            f"feature:{doc.get('feature_id') or doc.get('extractor_name') or doc.get('name')}",
        )
        doc.setdefault("kind", "transform")
        s.put(feature_store.COLLECTION, doc)
    return s


def seeded_store():
    """A MemoryStore preloaded with the shipped catalog seeds (test-only)."""
    from ..core import store as store_mod

    return _seed_into(store_mod.make_store())


@pytest.fixture(scope="session", autouse=True)
def _seed_main_store():
    """The server reads its catalogs from CouchDB and no longer seeds itself. The tool surface
    reads the module-level main._STORE, so fill that same store for the hermetic memory run."""
    from .. import main

    _seed_into(main._STORE)