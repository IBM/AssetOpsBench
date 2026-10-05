"""Tests for Utilities MCP server tools."""

import pytest
from servers.db_errors import DATA_UNAVAILABLE
from servers.utilities import main as utilities
from servers.utilities.main import mcp
from .conftest import call_tool


class FakeCatalogDB:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def check(self):
        return True

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
    async def test_missing_catalog_reports_unavailable(self, monkeypatch):
        monkeypatch.setattr(utilities, "catalog_db", None)

        data = await call_tool(mcp, "get_sensor_catalog", {})

        assert data == {"error": DATA_UNAVAILABLE}

    @pytest.mark.anyio
    async def test_unreachable_couchdb_hides_database_name_and_host(
        self, monkeypatch
    ):
        import couchdb3

        monkeypatch.setattr(
            utilities,
            "catalog_db",
            couchdb3.Database("secret_catalog", url="http://127.0.0.1:9"),
        )

        for tool in (
            "get_sensor_catalog",
            "get_asset_catalog",
            "get_failure_mode_catalog",
        ):
            data = await call_tool(mcp, tool, {})
            assert data == {"error": DATA_UNAVAILABLE}, tool

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

    @pytest.mark.anyio
    async def test_zero_result_names_a_filter_the_catalog_cannot_serve(self):
        """A catalog whose `category` column was blank for every row keeps no
        category key, because the CSV loader drops empty cells. Filtering on it
        then returns 0 entries with a success message, which reads the same as a
        category that simply is not catalogued. The aa_v1 run spent 220 calls
        and one turn-cap failure on that ambiguity."""
        utilities.catalog_db = FakeCatalogDB(
            [
                {"failure_mode": "Bearing Failure"},
                {"failure_mode": "Stator Damage"},
            ]
        )

        data = await call_tool(
            mcp,
            "get_failure_mode_catalog",
            {"category": "Rotating equipment"},
        )

        assert data["total"] == 0
        assert "`category`" in data["message"]
        assert "cannot match" in data["message"]

    @pytest.mark.anyio
    async def test_zero_result_stays_quiet_when_the_filter_is_serviceable(
        self, fake_catalog_db
    ):
        """A populated field that merely has no matching value must not be
        reported as unserviceable."""
        data = await call_tool(
            mcp,
            "get_failure_mode_catalog",
            {"category": "no such category"},
        )

        assert data["total"] == 0
        assert data["message"] == "found 0 failure_mode catalog entries"
