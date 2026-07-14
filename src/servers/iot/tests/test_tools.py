"""Tests for IoT MCP server tools.

Unit tests use a mocked CouchDB; integration tests require a live instance
(skipped unless COUCHDB_URL is set).
"""

import pytest
from servers.iot.main import mcp
from .conftest import requires_couchdb, call_tool


# ---------------------------------------------------------------------------
# sites
# ---------------------------------------------------------------------------


class TestSites:
    @pytest.mark.anyio
    async def test_returns_main(self):
        data = await call_tool(mcp, "sites", {})
        assert "MAIN" in data["sites"]


# ---------------------------------------------------------------------------
# asset_ids
# ---------------------------------------------------------------------------


class TestAssetIds:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(mcp, "asset_ids", {"site_name": "INVALID"})
        assert "error" in data
        assert "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_with_mock_asset_db(self, mock_asset_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {
                "docs": [
                    {"assetnum": "CHILLER6", "iot_asset_id": "Chiller 6"},
                    {"assetnum": "PUMP3", "iot_asset_id": None},
                ]
            },
        ]
        data = await call_tool(mcp, "asset_ids", {"site_name": "MAIN"})

        assert data["total_assets"] == 2
        assert data["assets"] == ["Chiller 6", "PUMP3"]

    @pytest.mark.anyio
    async def test_db_disconnected(self, no_asset_db):
        data = await call_tool(mcp, "asset_ids", {"site_name": "MAIN"})
        assert "error" in data
        assert "not connected" in data["error"].lower()

    @requires_couchdb
    @pytest.mark.anyio
    async def test_discovery_integration(self):
        data = await call_tool(mcp, "asset_ids", {"site_name": "MAIN"})
        assert "assets" in data
        assert "Chiller 6" in data["assets"]
        assert data["total_assets"] > 0


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------


class TestAssets:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(mcp, "assets", {"site_name": "INVALID"})
        assert "error" in data
        assert "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_with_mock_asset_db(self, mock_asset_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {
                "docs": [
                    {
                        "assetnum": "CHILLER6",
                        "assettype": "CHILLER",
                        "description": "Chiller 6",
                        "vintage": None,
                        "sensors": ["Supply Temperature", "Return Temperature"],
                    },
                    {
                        "assetnum": "PUMP3",
                        "assettype": "PUMP",
                        "description": "Pump 3",
                        "vintage": "new",
                        "sensors": [],
                    },
                ]
            },
        ]
        data = await call_tool(mcp, "assets", {"site_name": "MAIN"})

        assert data["total_assets"] == 2
        assert data["assets"][0] == {
            "asset_id": "CHILLER6",
            "description": "Chiller 6",
            "assettype": "CHILLER",
            "vintage": None,
            "n_sensors": 2,
        }
        assert data["assets"][1]["asset_id"] == "PUMP3"

    @pytest.mark.anyio
    async def test_filters_by_assettype(self, mock_asset_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {
                "docs": [
                    {
                        "assetnum": "PUMP3",
                        "assettype": "PUMP",
                        "description": "Pump 3",
                        "vintage": None,
                        "sensors": [],
                    }
                ]
            },
        ]
        data = await call_tool(
            mcp, "assets", {"site_name": "MAIN", "assettype": "PUMP"}
        )

        assert data["total_assets"] == 1
        assert data["assets"][0]["assettype"] == "PUMP"
        selector = mock_asset_db.find.call_args_list[1].args[0]
        assert selector["assettype"] == "PUMP"

    @pytest.mark.anyio
    async def test_db_disconnected(self, no_asset_db):
        data = await call_tool(mcp, "assets", {"site_name": "MAIN"})
        assert "error" in data
        assert "not connected" in data["error"].lower()

    @requires_couchdb
    @pytest.mark.anyio
    async def test_discovery_integration(self):
        data = await call_tool(mcp, "assets", {"site_name": "MAIN"})
        assert "assets" in data
        assert any(asset["asset_id"] == "CHILLER6" for asset in data["assets"])
        assert data["total_assets"] > 0


# ---------------------------------------------------------------------------
# sensors
# ---------------------------------------------------------------------------


class TestSensors:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(
            mcp, "sensors", {"site_name": "INVALID", "asset_id": "Chiller 6"}
        )
        assert "error" in data

    @pytest.mark.anyio
    async def test_unknown_asset(self):
        data = await call_tool(
            mcp, "sensors", {"site_name": "MAIN", "asset_id": "INVALID"}
        )
        assert "error" in data
        assert "no sensors found" in data["error"]

    @pytest.mark.anyio
    async def test_with_mock_db(self, mock_db):
        mock_db.find.return_value = {
            "docs": [
                {
                    "asset_id": "Chiller 1",
                    "timestamp": "2024-01-01T00:00:00",
                    "Temp": 25.5,
                    "Pressure": 10.2,
                    "_id": "doc1",
                    "_rev": "rev1",
                }
            ]
        }
        data = await call_tool(
            mcp, "sensors", {"site_name": "MAIN", "asset_id": "Chiller 1"}
        )

        assert data["total_sensors"] == 2
        assert "Temp" in data["sensors"]
        assert "Pressure" in data["sensors"]
        # Internal fields must be excluded
        assert "_id" not in data["sensors"]
        assert "_rev" not in data["sensors"]

    @requires_couchdb
    @pytest.mark.anyio
    async def test_success_integration(self):
        data = await call_tool(
            mcp, "sensors", {"site_name": "MAIN", "asset_id": "Chiller 6"}
        )
        assert "sensors" in data
        assert len(data["sensors"]) > 0
        assert any("Power" in s or "Efficiency" in s for s in data["sensors"])


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


class TestHistory:
    @pytest.mark.anyio
    async def test_invalid_date_range(self):
        data = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Chiller 6",
                "start": "2020-06-01T00:00:00",
                "final": "2020-05-01T00:00:00",
            },
        )
        assert "error" in data
        assert "start >= final" in data["error"]

    @pytest.mark.anyio
    async def test_malformed_date(self):
        data = await call_tool(
            mcp,
            "history",
            {"site_name": "MAIN", "asset_id": "Chiller 6", "start": "not-a-date"},
        )
        assert "error" in data

    @pytest.mark.anyio
    async def test_db_disconnected(self, no_db):
        data = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Chiller 6",
                "start": "2020-06-01T00:00:00",
            },
        )
        assert "error" in data
        assert "not connected" in data["error"].lower()

    @pytest.mark.anyio
    async def test_with_mock_db(self, mock_db):
        mock_db.find.return_value = {
            "docs": [
                {"timestamp": "2024-01-01T00:00:00", "Temp": 20.0},
                {"timestamp": "2024-01-01T00:15:00", "Temp": 21.0},
            ]
        }
        data = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Chiller 1",
                "start": "2024-01-01T00:00:00",
            },
        )

        assert data["total_observations"] == 2
        assert len(data["observations"]) == 2
        assert data["observations"][0]["Temp"] == 20.0

    @requires_couchdb
    @pytest.mark.anyio
    async def test_open_ended_integration(self):
        data = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Chiller 6",
                "start": "2020-06-01T00:00:00",
            },
        )
        assert "observations" in data
        assert "total_observations" in data

    @requires_couchdb
    @pytest.mark.anyio
    async def test_bounded_range_integration(self):
        data = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Chiller 6",
                "start": "2020-06-01T00:00:00",
                "final": "2020-06-01T01:00:00",
            },
        )
        assert "observations" in data
        for obs in data["observations"]:
            assert obs["timestamp"] >= "2020-06-01T00:00:00"
            assert obs["timestamp"] < "2020-06-01T01:00:00"
