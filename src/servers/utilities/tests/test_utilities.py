"""Tests for Utilities MCP server tools."""

import json
import os
import tempfile

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
# current_date_time
# ---------------------------------------------------------------------------


class TestCurrentDateTime:
    @pytest.mark.anyio
    async def test_response_structure(self):
        data = await call_tool(mcp, "current_date_time", {})
        assert "currentDateTime" in data
        assert "currentDateTimeDescription" in data

    @pytest.mark.anyio
    async def test_description_format(self):
        data = await call_tool(mcp, "current_date_time", {})
        desc = data["currentDateTimeDescription"]
        assert "Today's date is" in desc
        assert "time is" in desc

    @pytest.mark.anyio
    async def test_iso_format(self):
        data = await call_tool(mcp, "current_date_time", {})
        # Should contain a T separator (ISO 8601)
        assert "T" in data["currentDateTime"]


# ---------------------------------------------------------------------------
# current_time_english
# ---------------------------------------------------------------------------


class TestCurrentTimeEnglish:
    @pytest.mark.anyio
    async def test_response_structure(self):
        data = await call_tool(mcp, "current_time_english", {})
        assert "english" in data
        assert "iso" in data

    @pytest.mark.anyio
    async def test_english_is_readable(self):
        data = await call_tool(mcp, "current_time_english", {})
        # pendulum's to_datetime_string returns "YYYY-MM-DD HH:MM:SS"
        parts = data["english"].split(" ")
        assert len(parts) == 2  # date + time


# ---------------------------------------------------------------------------
# json_reader
# ---------------------------------------------------------------------------


class TestJsonReader:
    @pytest.mark.anyio
    async def test_reads_valid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump({"test": "data"}, tmp)
            tmp_name = tmp.name

        try:
            data = await call_tool(mcp, "json_reader", {"file_name": tmp_name})
            assert data == {"test": "data"}
        finally:
            os.remove(tmp_name)

    @pytest.mark.anyio
    async def test_reads_nested_json(self):
        payload = {"a": [1, 2, 3], "b": {"nested": True}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_name = tmp.name

        try:
            data = await call_tool(mcp, "json_reader", {"file_name": tmp_name})
            assert data == payload
        finally:
            os.remove(tmp_name)

    @pytest.mark.anyio
    async def test_nonexistent_file(self):
        data = await call_tool(
            mcp, "json_reader", {"file_name": "/tmp/does_not_exist_12345.json"}
        )
        assert "error" in data

    @pytest.mark.anyio
    async def test_invalid_json_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write("not valid json {{{")
            tmp_name = tmp.name

        try:
            data = await call_tool(mcp, "json_reader", {"file_name": tmp_name})
            assert "error" in data
        finally:
            os.remove(tmp_name)


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
    async def test_get_failure_model_catalog_queries_failure_mode(
        self, fake_catalog_db
    ):
        data = await call_tool(
            mcp,
            "get_failure_model_catalog",
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
