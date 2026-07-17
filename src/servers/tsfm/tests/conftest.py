"""Test config — keep the suite hermetic.

Production reads the catalog from CouchDB (the `model_catalog` collection, loaded by
src/couchdb/init_data.py). Tests force the in-memory backend and seed it from the same shipped
seed file, so they need no running service.
"""

import json
import os

import pytest

os.environ["TSFM_STORE"] = "memory"

_SEEDS = os.environ.get("TSFM_SEEDS_DIR") or os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "couchdb", "scenarios_data", "shared", "tsfm",
    )
)


def _seed_into(store):
    """Load the shipped model_catalog seed into a store (test-only)."""
    from ..stores import model_store

    for doc in json.load(open(os.path.join(_SEEDS, "model_catalog.json"))):
        doc.setdefault("_id", f"model:{doc['model_id']}")
        store.put(model_store.COLLECTION, doc)
    return store


def seeded_store():
    """A MemoryStore preloaded with the shipped model_catalog seed."""
    from ..core import store as store_mod

    return _seed_into(store_mod.make_store())


@pytest.fixture(scope="session", autouse=True)
def _seed_main_store():
    """The tool surface reads the module-level main._STORE; fill it for the hermetic run."""
    from .. import main

    _seed_into(main._STORE)
