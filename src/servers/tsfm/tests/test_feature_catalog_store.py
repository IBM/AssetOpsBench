"""Feature catalog store tests.

These cover the database-backed store logic through the in-memory test double, so
they do not require a live database.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from servers.tsfm import main as tsfm_main
from servers.tsfm.core.store import MemoryStore
from servers.tsfm.stores import feature_store

from .conftest import call_tool


_CATALOG = (
    Path(__file__).resolve().parents[3]
    / "couchdb"
    / "scenarios_data"
    / "shared"
    / "tsfm"
    / "feature_catalog.json"
)


def _feature_doc(doc: dict) -> dict:
    out = copy.deepcopy(doc)
    out.setdefault("_id", f"feature:{out['feature_id']}")
    return out


def seeded_feature_store() -> MemoryStore:
    store = MemoryStore()
    with _CATALOG.open() as fh:
        for doc in json.load(fh):
            store.put(feature_store.collection_name(), _feature_doc(doc))
    return store


def test_feature_store_lists_searches_and_versions_catalog_cards():
    store = seeded_feature_store()

    transforms = feature_store.find_features(store, kind="transform")
    assert [doc["feature_id"] for doc in transforms] == ["efe_time_robust_norm_v1"]

    extractors = feature_store.list_extractors(store)
    assert [doc["feature_id"] for doc in extractors] == ["abs_2nd_diff_mean"]

    matches = feature_store.search(store, "normalization")
    assert [doc["feature_id"] for doc in matches] == ["efe_time_robust_norm_v1"]

    successor = feature_store.new_version(
        store,
        "efe_time_robust_norm_v1",
        {"description": "updated normalization"},
    )
    assert successor["parent_feature_id"] == "efe_time_robust_norm_v1"
    assert successor["version"] == "2"
    assert feature_store.get_feature(store, "efe_time_robust_norm_v1")[
        "status"
    ] == "superseded"
    assert feature_store.get_lineage(store, successor["feature_id"])[
        "ancestors"
    ] == ["efe_time_robust_norm_v1"]


def test_feature_store_reads_collection_name_from_env(monkeypatch):
    monkeypatch.setenv("FEATURE_CATALOG_DBNAME", "custom_feature_catalog")
    store = MemoryStore()
    with _CATALOG.open() as fh:
        doc = _feature_doc(json.load(fh)[0])
    store.put("custom_feature_catalog", doc)

    assert feature_store.collection_name() == "custom_feature_catalog"
    assert feature_store.get_feature(store, doc["feature_id"])["feature_id"] == doc[
        "feature_id"
    ]
    assert feature_store.find_features(store, kind="transform")


def test_register_feature_rejects_in_place_transform():
    store = MemoryStore()
    bad = {
        "feature_id": "bad_inplace",
        "interface": "fit_transform",
        "code": (
            "class Transformation:\n"
            "    def fit(self, X, metadata):\n"
            "        return {}\n"
            "    def transform(self, X, state):\n"
            "        X[:] = 0\n"
            "        return X\n"
        ),
    }

    with pytest.raises(Exception, match="mutated"):
        feature_store.register_feature(store, bad)


@pytest.mark.anyio
async def test_feature_catalog_mcp_tools(monkeypatch):
    monkeypatch.setattr(tsfm_main, "_FEATURE_STORE", seeded_feature_store())

    listed = await call_tool(
        tsfm_main.mcp, "list_features", {"kind": "transform"}
    )
    assert listed["message"] == "listed 1 transform feature with status active."
    assert listed["features"][0]["feature_id"] == "efe_time_robust_norm_v1"

    found = await call_tool(
        tsfm_main.mcp, "search_features", {"text": "second difference"}
    )
    assert (
        found["message"]
        == "found 1 feature matching text 'second difference' with status active."
    )
    assert found["features"][0]["feature_id"] == "abs_2nd_diff_mean"

    card = await call_tool(
        tsfm_main.mcp,
        "get_feature",
        {"feature_id": "efe_time_robust_norm_v1"},
    )
    assert (
        card["message"]
        == "found transform feature efe_time_robust_norm_v1 with status active."
    )

    lineage = await call_tool(
        tsfm_main.mcp,
        "get_feature_lineage",
        {"feature_id": "efe_time_robust_norm_v1"},
    )
    assert (
        lineage["message"]
        == "lineage for feature efe_time_robust_norm_v1 has 0 ancestors and "
        "0 descendants; root is efe_time_robust_norm_v1."
    )
    assert lineage["feature_id"] == "efe_time_robust_norm_v1"

    feature = {
        "feature_id": "smoke_identity",
        "interface": "fit_transform",
        "code": (
            "class Transformation:\n"
            "    def fit(self, X, metadata):\n"
            "        return {}\n"
            "    def transform(self, X, state):\n"
            "        return X.copy()\n"
        ),
    }
    registered = await call_tool(
        tsfm_main.mcp,
        "register_feature",
        {"feature": feature},
    )
    assert registered["message"] == "registered feature smoke_identity."
    assert registered["card"]["feature_id"] == "smoke_identity"

    updated = await call_tool(
        tsfm_main.mcp,
        "update_feature",
        {
            "feature_id": "smoke_identity",
            "fields": {"description": "updated identity transform"},
        },
    )
    assert updated["message"] == "updated feature smoke_identity with 1 field."
    assert updated["description"] == "updated identity transform"

    versioned = await call_tool(
        tsfm_main.mcp,
        "new_feature_version",
        {
            "feature_id": "smoke_identity",
            "fields": {"description": "successor identity transform"},
            "new_feature_id": "smoke_identity_v2",
        },
    )
    assert (
        versioned["message"]
        == "created feature version smoke_identity_v2 from smoke_identity."
    )
    assert versioned["feature_id"] == "smoke_identity_v2"

    deprecated = await call_tool(
        tsfm_main.mcp,
        "deprecate_feature",
        {"feature_id": "smoke_identity_v2", "reason": "covered by successor"},
    )
    assert (
        deprecated["message"]
        == "deprecated feature smoke_identity_v2. Reason: covered by successor."
    )
    assert deprecated["status"] == "deprecated"
