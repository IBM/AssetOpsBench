"""Tests for Utilities MCP server tools."""

import pytest
from servers.utilities import main as utilities
from servers.utilities.main import mcp
from .conftest import call_tool


class FakeCatalogDB:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def find(self, selector, fields=None, limit=200):
        self.calls.append({"selector": selector, "fields": fields, "limit": limit})
        matched = []
        for doc in self.docs:
            if self._matches(doc, selector):
                if fields:
                    matched.append(
                        {field: doc[field] for field in fields if field in doc}
                    )
                else:
                    matched.append(doc)
        return {"docs": matched[:limit]}

    def _matches(self, doc, selector):
        for field, expected in selector.items():
            if isinstance(expected, dict) and "$exists" in expected:
                if (field in doc) != expected["$exists"]:
                    return False
            elif doc.get(field) != expected:
                return False
        return True


@pytest.fixture
def fake_catalog_db(monkeypatch):
    fake = FakeCatalogDB(
        [
            {"sensor": "air flow", "description": "Airflow sensor"},
            {
                "category": "rotating equipment",
                "category_description": "Rotating equipment",
                "asset": "Electric motor",
                "description": "Converts electricity into motion.",
            },
            {
                "category": "rotating equipment",
                "failure_mode": "Air inlet blockage",
                "description": "Air intake is blocked.",
            },
        ]
    )
    monkeypatch.setattr(utilities, "catalog_db", fake)
    return fake


# ---------------------------------------------------------------------------
# catalog tools
# ---------------------------------------------------------------------------


class TestCatalogTools:
    @pytest.mark.anyio
    async def test_get_sensor_catalog_lists_sensor_entries(self, fake_catalog_db):
        data = await call_tool(mcp, "get_sensor_catalog", {})

        assert data["catalog_type"] == "sensor"
        assert data["total"] == 1
        assert data["entries"] == [
            {"sensor": "air flow", "description": "Airflow sensor"}
        ]
        assert fake_catalog_db.calls[-1]["selector"] == {
            "sensor": {"$exists": True}
        }
        assert fake_catalog_db.calls[-1]["limit"] == utilities.CATALOG_QUERY_LIMIT

    @pytest.mark.anyio
    async def test_get_asset_catalog_filters_asset_and_category(self, fake_catalog_db):
        data = await call_tool(
            mcp,
            "get_asset_catalog",
            {"asset": "Electric motor", "category": "rotating equipment"},
        )

        assert data["catalog_type"] == "asset"
        assert data["query"] == "Electric motor, category=rotating equipment"
        assert data["total"] == 1
        assert data["entries"][0]["asset"] == "Electric motor"
        assert fake_catalog_db.calls[-1]["selector"] == {
            "asset": "Electric motor",
            "category": "rotating equipment",
        }

    @pytest.mark.anyio
    async def test_get_failure_mode_catalog_queries_failure_mode(
        self, fake_catalog_db
    ):
        data = await call_tool(
            mcp,
            "get_failure_mode_catalog",
            {"failure_mode": "Air inlet blockage"},
        )

        assert data["catalog_type"] == "failure_mode"
        assert data["total"] == 1
        assert data["entries"] == [
            {
                "category": "rotating equipment",
                "failure_mode": "Air inlet blockage",
                "description": "Air intake is blocked.",
            }
        ]
        assert fake_catalog_db.calls[-1]["selector"] == {
            "failure_mode": "Air inlet blockage"
        }
