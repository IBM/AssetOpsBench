"""Tests for IoT MCP server tools."""

import pytest

from servers.iot.main import mcp
from .conftest import call_tool, requires_couchdb


class TestToolRegistration:
    @pytest.mark.anyio
    async def test_registry_tools_are_registered(self):
        tools = await mcp.list_tools()
        assert sorted(tool.name for tool in tools) == ["asset_ids", "assets", "sites"]


class TestSites:
    @pytest.mark.anyio
    async def test_returns_known_sites(self, mock_asset_db):
        mock_asset_db.find.return_value = {
            "docs": [{"siteid": "MAIN"}, {"siteid": "NORTH"}, {"siteid": "MAIN"}]
        }
        data = await call_tool(mcp, "sites", {})

        assert data["sites"] == ["MAIN", "NORTH"]

    @pytest.mark.anyio
    async def test_falls_back_to_default_site(self, no_asset_db):
        data = await call_tool(mcp, "sites", {})
        assert data["sites"] == ["MAIN"]


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
                    {"assetnum": "Chiller 6"},
                    {"assetnum": "PUMP3"},
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
                        "assetnum": "Chiller 6",
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
            "asset_id": "Chiller 6",
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
        assert any(asset["asset_id"] == "Chiller 6" for asset in data["assets"])
        assert data["total_assets"] > 0
