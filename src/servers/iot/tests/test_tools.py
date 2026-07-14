"""Tests for IoT MCP server tools."""

import pytest

from servers.iot.main import mcp
from .conftest import call_tool, requires_couchdb, requires_iot_db


class TestToolRegistration:
    @pytest.mark.anyio
    async def test_registry_tools_are_registered(self):
        tools = await mcp.list_tools()
        assert sorted(tool.name for tool in tools) == [
            "asset_detail",
            "asset_ids",
            "assets",
            "find_assets_by_sensors",
            "installed_sensors",
            "measured_sensors",
            "sites",
            "stream_extent",
        ]

    @pytest.mark.anyio
    async def test_stream_extent_description_is_storage_neutral(self):
        tools = await mcp.list_tools()
        stream_extent_tool = next(tool for tool in tools if tool.name == "stream_extent")

        assert "couchdb" not in stream_extent_tool.description.lower()


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


class TestAssetDetail:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(
            mcp, "asset_detail", {"site_name": "INVALID", "asset_id": "Pump-1"}
        )
        assert "error" in data
        assert "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_with_mock_asset_db(self, mock_asset_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {
                "docs": [
                    {
                        "assetnum": "Pump-1",
                        "description": "Main pump",
                        "assettype": "PUMP",
                        "status": "OPERATING",
                        "location": "PUMP-HOUSE",
                        "installdate": "2024-01-01",
                        "vintage": "new",
                        "sensors": ["Pressure", "Temperature"],
                    }
                ]
            },
        ]

        data = await call_tool(
            mcp, "asset_detail", {"site_name": "MAIN", "asset_id": "Pump-1"}
        )

        assert data == {
            "site_name": "MAIN",
            "asset_id": "Pump-1",
            "description": "Main pump",
            "assettype": "PUMP",
            "status": "OPERATING",
            "location": "PUMP-HOUSE",
            "installdate": "2024-01-01",
            "vintage": "new",
            "n_installed_sensors": 2,
            "message": "asset Pump-1 is a PUMP (new vintage) at PUMP-HOUSE with 2 installed sensors.",
        }

    @pytest.mark.anyio
    async def test_reads_asset_registry_not_iot_db(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {
                "docs": [
                    {
                        "assetnum": "Pump-1",
                        "description": "Registry pump",
                        "assettype": "PUMP",
                        "status": "OPERATING",
                        "location": None,
                        "installdate": None,
                        "vintage": None,
                        "sensors": [],
                    }
                ]
            },
        ]
        mock_iot_db.find.return_value = {
            "docs": [
                {
                    "asset_id": "Pump-1",
                    "timestamp": "2024-01-01T00:00:00",
                    "Telemetry Sensor": 42,
                }
            ]
        }

        data = await call_tool(
            mcp, "asset_detail", {"site_name": "MAIN", "asset_id": "Pump-1"}
        )

        assert data["description"] == "Registry pump"
        assert data["n_installed_sensors"] == 0
        mock_iot_db.find.assert_not_called()

    @pytest.mark.anyio
    async def test_db_disconnected(self, no_asset_db):
        data = await call_tool(
            mcp, "asset_detail", {"site_name": "MAIN", "asset_id": "Pump-1"}
        )
        assert "error" in data
        assert "not connected" in data["error"].lower()

    @requires_couchdb
    @pytest.mark.anyio
    async def test_discovery_integration(self):
        data = await call_tool(
            mcp, "asset_detail", {"site_name": "MAIN", "asset_id": "Chiller 6"}
        )
        assert data["asset_id"] == "Chiller 6"
        assert data["assettype"] == "CHILLER"
        assert data["status"] == "OPERATING"
        assert data["n_installed_sensors"] > 0


class TestMeasuredSensors:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(
            mcp, "measured_sensors", {"site_name": "INVALID", "asset_id": "Pump-1"}
        )
        assert "error" in data
        assert "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_with_mock_iot_db(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {
                    "_id": "iot:Pump-1:1",
                    "_rev": "1-abc",
                    "asset_id": "Pump-1",
                    "timestamp": "2024-01-01T00:00:00",
                    "dataset": "iot",
                    "Pressure": 10,
                },
                {
                    "asset_id": "Pump-1",
                    "timestamp": "2024-01-01T00:01:00",
                    "dataset": "iot",
                    "Temperature": 30,
                    "Pressure": 11,
                },
            ]
        }

        data = await call_tool(
            mcp, "measured_sensors", {"site_name": "MAIN", "asset_id": "Pump-1"}
        )

        assert data["site_name"] == "MAIN"
        assert data["asset_id"] == "Pump-1"
        assert data["total_sensors"] == 2
        assert data["sensors"] == ["Pressure", "Temperature"]

    @pytest.mark.anyio
    async def test_reads_iot_db_not_asset_registry(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {
            "docs": [
                {
                    "siteid": "MAIN",
                    "assetnum": "Pump-1",
                    "sensors": ["Registry Sensor"],
                }
            ]
        }
        mock_iot_db.find.return_value = {
            "docs": [
                {
                    "asset_id": "Pump-1",
                    "timestamp": "2024-01-01T00:00:00",
                    "Telemetry Sensor": 42,
                }
            ]
        }

        data = await call_tool(
            mcp, "measured_sensors", {"site_name": "MAIN", "asset_id": "Pump-1"}
        )

        assert data["sensors"] == ["Telemetry Sensor"]
        assert mock_asset_db.find.call_count == 1
        mock_asset_db.find.assert_called_once_with(
            {"siteid": {"$exists": True}},
            fields=["siteid"],
            limit=100000,
        )
        mock_iot_db.find.assert_called_once()

    @pytest.mark.anyio
    async def test_db_disconnected(self, mock_asset_db, no_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        data = await call_tool(
            mcp, "measured_sensors", {"site_name": "MAIN", "asset_id": "Pump-1"}
        )
        assert "error" in data
        assert "not connected" in data["error"].lower()

    @requires_iot_db
    @pytest.mark.anyio
    async def test_discovery_integration(self):
        data = await call_tool(
            mcp, "measured_sensors", {"site_name": "MAIN", "asset_id": "Chiller 6"}
        )
        assert "sensors" in data
        assert "Chiller 6 Supply Temperature" in data["sensors"]
        assert data["total_sensors"] > 0


class TestInstalledSensors:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(
            mcp, "installed_sensors", {"site_name": "INVALID", "asset_id": "Pump-1"}
        )
        assert "error" in data
        assert "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_with_mock_asset_db(self, mock_asset_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {
                "docs": [
                    {
                        "assetnum": "Pump-1",
                        "sensors": ["Registry Pressure", "Registry Temperature"],
                    }
                ]
            },
        ]

        data = await call_tool(
            mcp, "installed_sensors", {"site_name": "MAIN", "asset_id": "Pump-1"}
        )

        assert data["site_name"] == "MAIN"
        assert data["asset_id"] == "Pump-1"
        assert data["total_sensors"] == 2
        assert data["sensors"] == ["Registry Pressure", "Registry Temperature"]
        mock_asset_db.find.assert_called_with(
            {"siteid": "MAIN", "assetnum": "Pump-1"},
            fields=["assetnum", "sensors"],
            limit=1,
        )

    @pytest.mark.anyio
    async def test_reads_asset_registry_not_iot_db(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {
                "docs": [
                    {
                        "assetnum": "Pump-1",
                        "sensors": ["Registry Sensor"],
                    }
                ]
            },
        ]
        mock_iot_db.find.return_value = {
            "docs": [
                {
                    "asset_id": "Pump-1",
                    "timestamp": "2024-01-01T00:00:00",
                    "Telemetry Sensor": 42,
                }
            ]
        }

        data = await call_tool(
            mcp, "installed_sensors", {"site_name": "MAIN", "asset_id": "Pump-1"}
        )

        assert data["sensors"] == ["Registry Sensor"]
        mock_iot_db.find.assert_not_called()

    @pytest.mark.anyio
    async def test_db_disconnected(self, no_asset_db):
        data = await call_tool(
            mcp, "installed_sensors", {"site_name": "MAIN", "asset_id": "Pump-1"}
        )
        assert "error" in data
        assert "not connected" in data["error"].lower()

    @requires_couchdb
    @pytest.mark.anyio
    async def test_discovery_integration(self):
        data = await call_tool(
            mcp, "installed_sensors", {"site_name": "MAIN", "asset_id": "Chiller 6"}
        )
        assert "sensors" in data
        assert "Chiller 6 Oil Pressure" in data["sensors"]
        assert data["total_sensors"] > 0


class TestFindAssetsBySensors:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(
            mcp,
            "find_assets_by_sensors",
            {"site_name": "INVALID", "sensors": ["Pressure"]},
        )
        assert "error" in data
        assert "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_installed_source_exact_match(self, mock_asset_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {"docs": [{"assetnum": "Fan-2"}, {"assetnum": "Pump-1"}]},
            {"docs": [{"sensors": ["Temperature"]}]},
            {"docs": [{"sensors": ["Pressure", "Temperature"]}]},
        ]

        data = await call_tool(
            mcp,
            "find_assets_by_sensors",
            {
                "site_name": "MAIN",
                "sensors": ["Pressure"],
                "source": "installed",
            },
        )

        assert data["site_name"] == "MAIN"
        assert data["query_sensors"] == ["Pressure"]
        assert data["match"] == "all"
        assert data["source"] == "installed"
        assert data["total_assets"] == 1
        assert data["assets"] == [
            {"asset_id": "Pump-1", "matched_sensors": ["Pressure"]}
        ]

    @pytest.mark.anyio
    async def test_deduplicates_query_sensors_for_all_match(self, mock_asset_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {"docs": [{"assetnum": "Pump-1"}]},
            {"docs": [{"sensors": ["Pressure", "Temperature"]}]},
        ]

        data = await call_tool(
            mcp,
            "find_assets_by_sensors",
            {
                "site_name": "MAIN",
                "sensors": ["Pressure", "Pressure"],
                "source": "installed",
            },
        )

        assert data["query_sensors"] == ["Pressure"]
        assert data["total_assets"] == 1
        assert data["assets"] == [
            {"asset_id": "Pump-1", "matched_sensors": ["Pressure"]}
        ]

    @pytest.mark.anyio
    async def test_measured_source_substring_match(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {"docs": [{"assetnum": "Compressor-1"}, {"assetnum": "Pump-1"}]},
        ]

        def find_records(selector, **kwargs):
            asset_id = selector["asset_id"]
            if asset_id == "Compressor-1":
                return {
                    "docs": [
                        {
                            "asset_id": "Compressor-1",
                            "timestamp": "2024-01-01T00:00:00",
                            "Oil Pressure": 12,
                        }
                    ]
                }
            if asset_id == "Pump-1":
                return {
                    "docs": [
                        {
                            "asset_id": "Pump-1",
                            "timestamp": "2024-01-01T00:00:00",
                            "Discharge Pressure": 42,
                            "Flow": 4,
                        }
                    ]
                }
            return {"docs": []}

        mock_iot_db.find.side_effect = find_records

        data = await call_tool(
            mcp,
            "find_assets_by_sensors",
            {
                "site_name": "MAIN",
                "sensors": ["pressure"],
                "substring": True,
            },
        )

        assert data["total_assets"] == 2
        assert data["assets"] == [
            {"asset_id": "Compressor-1", "matched_sensors": ["Oil Pressure"]},
            {"asset_id": "Pump-1", "matched_sensors": ["Discharge Pressure"]},
        ]


class TestStreamExtent:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(
            mcp,
            "stream_extent",
            {"site_name": "INVALID", "asset_id": "Pump-1"},
        )

        assert "error" in data
        assert "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_invalid_date(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "stream_extent",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "start": "not-a-date",
            },
        )

        assert "error" in data
        assert "Invalid date format" in data["error"]

    @pytest.mark.anyio
    async def test_start_must_precede_end(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "stream_extent",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "start": "2024-01-02T00:00:00",
                "end": "2024-01-01T00:00:00",
            },
        )

        assert data == {"error": "start >= end"}

    @pytest.mark.anyio
    async def test_db_disconnected(self, mock_asset_db, no_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "stream_extent",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert "error" in data
        assert "not connected" in data["error"].lower()

    @pytest.mark.anyio
    async def test_rejects_reserved_sensor_field(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "stream_extent",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "sensor": "asset_id",
            },
        )

        assert data == {
            "error": (
                "sensor must be a telemetry field, "
                "not reserved metadata field asset_id"
            )
        }
        mock_iot_db.find.assert_not_called()

    @pytest.mark.anyio
    async def test_with_mock_iot_db(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {"timestamp": "2024-01-01T00:00:00"},
                {"timestamp": "2024-01-01T00:01:00"},
                {"timestamp": "2024-01-01T00:02:00"},
            ]
        }

        data = await call_tool(
            mcp,
            "stream_extent",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data == {
            "site_name": "MAIN",
            "asset_id": "Pump-1",
            "sensor": None,
            "start_time": "2024-01-01T00:00:00",
            "end_time": "2024-01-01T00:02:00",
            "total_records": 3,
            "exceeds_page_limit": False,
            "approx_interval_seconds": 60.0,
            "message": (
                "3 record(s) for asset_id Pump-1 from "
                "2024-01-01T00:00:00 to 2024-01-01T00:02:00."
            ),
        }
        mock_iot_db.find.assert_called_once_with(
            {
                "asset_id": "Pump-1",
                "timestamp": {"$exists": True, "$ne": None},
            },
            limit=1000,
            sort=[{"asset_id": "asc"}, {"timestamp": "asc"}],
            fields=["timestamp"],
        )

    @pytest.mark.anyio
    async def test_sensor_and_window_selector(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {"timestamp": "2023-12-31T23:59:00"},
                {"timestamp": "2024-01-01T00:01:00"},
                {"timestamp": "2024-01-01T00:05:00"},
            ]
        }

        data = await call_tool(
            mcp,
            "stream_extent",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "sensor": "Pressure",
                "start": "2024-01-01T00:00:00",
                "end": "2024-01-01T00:05:00",
            },
        )

        assert data["sensor"] == "Pressure"
        assert data["total_records"] == 1
        assert data["start_time"] == "2024-01-01T00:01:00"
        assert data["end_time"] == "2024-01-01T00:01:00"
        selector = mock_iot_db.find.call_args.args[0]
        assert selector == {
            "asset_id": "Pump-1",
            "timestamp": {"$exists": True, "$ne": None},
            "Pressure": {"$exists": True, "$ne": None},
        }

    @pytest.mark.anyio
    async def test_date_only_window_selector_preserves_input(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {"timestamp": "2023-12-31T23:59:00"},
                {"timestamp": "2024-01-01T12:00:00"},
                {"timestamp": "2024-01-02T00:00:00"},
            ]
        }

        data = await call_tool(
            mcp,
            "stream_extent",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "start": "2024-01-01",
                "end": "2024-01-02",
            },
        )

        assert data["total_records"] == 1
        assert data["start_time"] == "2024-01-01T12:00:00"
        assert data["end_time"] == "2024-01-01T12:00:00"
        selector = mock_iot_db.find.call_args.args[0]
        assert selector["timestamp"] == {"$exists": True, "$ne": None}

    @pytest.mark.anyio
    async def test_bounds_must_match_stream_timezone_awareness(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [{"timestamp": "2024-01-01T00:00:00"}]
        }

        data = await call_tool(
            mcp,
            "stream_extent",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "start": "2024-01-01T00:00:00+00:00",
            },
        )

        assert data == {
            "error": (
                "timestamp bounds must use the same timezone awareness "
                "as telemetry timestamps"
            )
        }

    @pytest.mark.anyio
    async def test_compares_explicit_offsets_chronologically(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {"timestamp": "2023-12-31T23:00:00+00:00"},
                {"timestamp": "2024-01-01T00:30:00+02:00"},
                {"timestamp": "2024-01-01T00:00:00+00:00"},
            ]
        }

        data = await call_tool(
            mcp,
            "stream_extent",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "start": "2023-12-31T22:15:00+00:00",
                "end": "2023-12-31T23:30:00+00:00",
            },
        )

        assert data["total_records"] == 2
        assert data["start_time"] == "2024-01-01T00:30:00+02:00"
        assert data["end_time"] == "2023-12-31T23:00:00+00:00"
        assert data["approx_interval_seconds"] == 1800.0

    @pytest.mark.anyio
    async def test_mixed_stream_timezone_awareness_returns_error(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {"timestamp": "2024-01-01T00:00:00"},
                {"timestamp": "2024-01-01T01:00:00+00:00"},
            ]
        }

        data = await call_tool(
            mcp,
            "stream_extent",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data == {"error": "telemetry timestamps use mixed timezone awareness"}

    @pytest.mark.anyio
    async def test_invalid_stream_timestamp_returns_error(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {"docs": [{"timestamp": "not-a-date"}]}

        data = await call_tool(
            mcp,
            "stream_extent",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data == {
            "error": "telemetry record has an invalid ISO 8601 timestamp"
        }

    @pytest.mark.anyio
    async def test_records_without_timestamps_are_not_counted(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {},
                {"timestamp": "2024-01-01T00:00:00"},
                {"timestamp": None},
            ]
        }

        data = await call_tool(
            mcp,
            "stream_extent",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data["total_records"] == 1
        assert data["start_time"] == "2024-01-01T00:00:00"
        assert data["end_time"] == "2024-01-01T00:00:00"

    @pytest.mark.anyio
    async def test_no_records(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {"docs": []}

        data = await call_tool(
            mcp,
            "stream_extent",
            {"site_name": "MAIN", "asset_id": "Pump-1", "sensor": "Pressure"},
        )

        assert "error" in data
        assert data["error"] == "no records for asset_id Pump-1, sensor Pressure"

    @pytest.mark.anyio
    async def test_query_error_does_not_expose_storage_backend(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.side_effect = RuntimeError("CouchDB unavailable")

        data = await call_tool(
            mcp,
            "stream_extent",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data == {"error": "unable to inspect telemetry stream extent"}
        assert "couchdb" not in data["error"].lower()

    @requires_iot_db
    @pytest.mark.anyio
    async def test_discovery_integration(self):
        data = await call_tool(
            mcp,
            "stream_extent",
            {"site_name": "MAIN", "asset_id": "Chiller 6"},
        )

        assert data["asset_id"] == "Chiller 6"
        assert data["total_records"] > 0
        assert data["start_time"] is not None
        assert data["end_time"] is not None


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
