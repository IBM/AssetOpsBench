"""Tests for IoT MCP server tools."""

import json
from unittest.mock import patch

import pytest

from servers.iot.main import mcp
from .conftest import call_tool, requires_couchdb, requires_iot_db


class ViewBackedIotDb:
    def __init__(self, rows_by_view):
        self.rows_by_view = rows_by_view
        self.view_calls = []
        self.find_calls = []

    def view(self, ddoc, view_name, **kwargs):
        self.view_calls.append((ddoc, view_name, kwargs))
        rows = self.rows_by_view.get(view_name, [])
        return {"rows": rows(view_name, kwargs) if callable(rows) else rows}

    def find(self, selector, **kwargs):
        self.find_calls.append((selector, kwargs))
        return {"docs": []}


class SummaryBackedIotDb:
    def __init__(self, summaries):
        self.summaries = summaries
        self.get_calls = []
        self.view_calls = []
        self.find_calls = []

    def get(self, doc_id, **kwargs):
        self.get_calls.append((doc_id, kwargs))
        return self.summaries.get(doc_id)

    def view(self, ddoc, view_name, **kwargs):
        self.view_calls.append((ddoc, view_name, kwargs))
        return {"rows": []}

    def find(self, selector, **kwargs):
        self.find_calls.append((selector, kwargs))
        return {"docs": []}


class AllDocsBackedIotDb:
    def __init__(self, docs):
        self.docs = {doc["_id"]: doc for doc in docs}
        self.all_docs_calls = []
        self.find_calls = []
        self.view_calls = []

    def all_docs(self, **kwargs):
        self.all_docs_calls.append(kwargs)
        startkey = kwargs.get("startkey")
        endkey = kwargs.get("endkey")
        if isinstance(startkey, str) and startkey.startswith('"'):
            startkey = json.loads(startkey)
        if isinstance(endkey, str) and endkey.startswith('"'):
            endkey = json.loads(endkey)
        limit = kwargs.get("limit", 1000)
        skip = kwargs.get("skip", 0)
        include_docs = kwargs.get("include_docs")
        inclusive_end = kwargs.get("inclusive_end", True)
        keys = sorted(self.docs)
        rows = []
        for key in keys:
            if startkey is not None and key < startkey:
                continue
            if endkey is not None:
                if inclusive_end and key > endkey:
                    continue
                if not inclusive_end and key >= endkey:
                    continue
            rows.append(
                {
                    "id": key,
                    "key": key,
                    "value": {"rev": "1-test"},
                    **({"doc": self.docs[key]} if include_docs else {}),
                }
            )
        return {"rows": rows[skip : skip + limit]}

    def find(self, selector, **kwargs):
        self.find_calls.append((selector, kwargs))
        return {"docs": []}

    def view(self, ddoc, view_name, **kwargs):
        self.view_calls.append((ddoc, view_name, kwargs))
        return {"rows": []}


def iot_summary_doc(asset_id="Pump-1"):
    return {
        "_id": f"iot_summary:{asset_id}",
        "dataset": "iot",
        "doctype": "iot_asset_summary",
        "summary_asset_id": asset_id,
        "timestamped_records": 3,
        "invalid_timestamp_count": 0,
        "mixed_timezone_awareness": False,
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T00:02:00",
        "sensors": ["Pressure", "Temp"],
        "coverage": {
            "Pressure": {
                "non_null_count": 2,
                "first_timestamp": "2024-01-01T00:00:00",
                "last_timestamp": "2024-01-01T00:01:00",
                "latest_timestamp": "2024-01-01T00:01:00",
                "latest_value": 6.0,
            },
            "Temp": {
                "non_null_count": 3,
                "first_timestamp": "2024-01-01T00:00:00",
                "last_timestamp": "2024-01-01T00:02:00",
                "latest_timestamp": "2024-01-01T00:02:00",
                "latest_value": 12.0,
            },
        },
        "stats": {
            "Pressure": {
                "count": 2,
                "null_count": 0,
                "min": 5.0,
                "max": 6.0,
                "mean": 5.5,
                "stddev": 0.5,
                "first_timestamp": "2024-01-01T00:00:00",
                "last_timestamp": "2024-01-01T00:01:00",
            },
            "Temp": {
                "count": 3,
                "null_count": 0,
                "min": 10.0,
                "max": 12.0,
                "mean": 11.0,
                "stddev": 0.816496580927726,
                "first_timestamp": "2024-01-01T00:00:00",
                "last_timestamp": "2024-01-01T00:02:00",
            },
        },
        "latest": {
            "timestamp": "2024-01-01T00:02:00",
            "values": {"Pressure": None, "Temp": 12.0},
        },
    }


class TestToolRegistration:
    @pytest.mark.anyio
    async def test_registry_tools_are_registered(self):
        tools = await mcp.list_tools()
        assert sorted(tool.name for tool in tools) == [
            "asset_detail",
            "asset_ids",
            "assets",
            "find_assets_by_sensors",
            "history",
            "installed_sensors",
            "latest_reading",
            "measured_sensors",
            "sensor_coverage",
            "sensor_stats",
            "sites",
            "stream_extent",
        ]

    @pytest.mark.anyio
    async def test_stream_extent_description_is_storage_neutral(self):
        tools = await mcp.list_tools()
        descriptions = {
            tool.name: tool.description
            for tool in tools
            if tool.name
            in {
                "history",
                "latest_reading",
                "sensor_coverage",
                "sensor_stats",
                "stream_extent",
            }
        }

        assert all(
            "couchdb" not in description.lower()
            for description in descriptions.values()
        )


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
    async def test_uses_materialized_summary_when_available(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = SummaryBackedIotDb(
            {"iot_summary:Pump-1": iot_summary_doc()}
        )

        with patch("servers.iot.main.iot_db", fake_iot_db):
            data = await call_tool(
                mcp,
                "measured_sensors",
                {"site_name": "MAIN", "asset_id": "Pump-1"},
            )

        assert data["sensors"] == ["Pressure", "Temp"]
        assert fake_iot_db.get_calls == [
            ("iot_summary:Pump-1", {"default_value": None})
        ]
        assert fake_iot_db.find_calls == []
        assert fake_iot_db.view_calls == []

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
    async def test_uses_sensor_coverage_view_when_available(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = ViewBackedIotDb(
            {
                "coverage_by_asset_sensor": [
                    {"key": ["Pump-1", "Pressure"], "value": {"non_null_count": 2}},
                    {"key": ["Pump-1", "Temperature"], "value": {"non_null_count": 1}},
                ]
            }
        )

        with patch("servers.iot.main.iot_db", fake_iot_db), patch(
            "servers.iot.main.IOT_ENABLE_VIEW_FAST_PATH", True
        ):
            data = await call_tool(
                mcp,
                "measured_sensors",
                {"site_name": "MAIN", "asset_id": "Pump-1"},
            )

        assert data["sensors"] == ["Pressure", "Temperature"]
        assert fake_iot_db.find_calls == []
        assert [call[1] for call in fake_iot_db.view_calls] == [
            "coverage_by_asset_sensor"
        ]

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

    @pytest.mark.anyio
    async def test_measured_source_exact_match_uses_reverse_view(self, mock_asset_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {"docs": [{"assetnum": "Fan-2"}, {"assetnum": "Pump-1"}]},
        ]
        fake_iot_db = ViewBackedIotDb(
            {
                "assets_by_sensor": lambda _view, kwargs: (
                    [
                        {"key": ["Pressure", "Pump-1"], "value": 3},
                        {"key": ["Pressure", "Other-Site"], "value": 1},
                    ]
                    if "Pressure" in kwargs["startkey"]
                    else [
                        {"key": ["Temperature", "Pump-1"], "value": 2},
                        {"key": ["Temperature", "Fan-2"], "value": 1},
                    ]
                )
            }
        )

        with patch("servers.iot.main.iot_db", fake_iot_db), patch(
            "servers.iot.main.IOT_ENABLE_VIEW_FAST_PATH", True
        ):
            data = await call_tool(
                mcp,
                "find_assets_by_sensors",
                {
                    "site_name": "MAIN",
                    "sensors": ["Pressure", "Temperature"],
                    "source": "measured",
                },
            )

        assert data["total_assets"] == 1
        assert data["assets"] == [
            {"asset_id": "Pump-1", "matched_sensors": ["Pressure", "Temperature"]}
        ]
        assert fake_iot_db.find_calls == []
        assert [call[1] for call in fake_iot_db.view_calls] == [
            "assets_by_sensor",
            "assets_by_sensor",
        ]

    @pytest.mark.anyio
    async def test_measured_source_uses_summaries_by_default(self, mock_asset_db):
        mock_asset_db.find.side_effect = [
            {"docs": [{"siteid": "MAIN"}]},
            {"docs": [{"assetnum": "Fan-2"}, {"assetnum": "Pump-1"}]},
        ]
        fake_iot_db = SummaryBackedIotDb(
            {
                "iot_summary:Fan-2": iot_summary_doc("Fan-2")
                | {"sensors": ["Temp"]},
                "iot_summary:Pump-1": iot_summary_doc("Pump-1"),
            }
        )

        with patch("servers.iot.main.iot_db", fake_iot_db):
            data = await call_tool(
                mcp,
                "find_assets_by_sensors",
                {
                    "site_name": "MAIN",
                    "sensors": ["Pressure", "Temp"],
                    "source": "measured",
                },
            )

        assert data["total_assets"] == 1
        assert data["assets"] == [
            {"asset_id": "Pump-1", "matched_sensors": ["Pressure", "Temp"]}
        ]
        assert fake_iot_db.find_calls == []
        assert fake_iot_db.view_calls == []


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
                "timestamp": {"$gt": None},
            },
            limit=1000,
            sort=[{"asset_id": "asc"}, {"timestamp": "asc"}],
            fields=["timestamp"],
        )

    @pytest.mark.anyio
    async def test_full_stream_uses_materialized_summary(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = SummaryBackedIotDb(
            {"iot_summary:Pump-1": iot_summary_doc()}
        )

        with patch("servers.iot.main.iot_db", fake_iot_db):
            data = await call_tool(
                mcp,
                "stream_extent",
                {"site_name": "MAIN", "asset_id": "Pump-1"},
            )

        assert data["total_records"] == 3
        assert data["start_time"] == "2024-01-01T00:00:00"
        assert data["end_time"] == "2024-01-01T00:02:00"
        assert data["approx_interval_seconds"] == 60.0
        assert fake_iot_db.find_calls == []
        assert fake_iot_db.view_calls == []

    @pytest.mark.anyio
    async def test_sensor_extent_uses_materialized_summary(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = SummaryBackedIotDb(
            {"iot_summary:Pump-1": iot_summary_doc()}
        )

        with patch("servers.iot.main.iot_db", fake_iot_db):
            data = await call_tool(
                mcp,
                "stream_extent",
                {
                    "site_name": "MAIN",
                    "asset_id": "Pump-1",
                    "sensor": "Pressure",
                },
            )

        assert data["sensor"] == "Pressure"
        assert data["total_records"] == 2
        assert data["start_time"] == "2024-01-01T00:00:00"
        assert data["end_time"] == "2024-01-01T00:01:00"
        assert fake_iot_db.find_calls == []
        assert fake_iot_db.view_calls == []

    @pytest.mark.anyio
    async def test_windowed_stream_uses_all_docs_when_summary_not_applicable(
        self, mock_asset_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = AllDocsBackedIotDb(
            [
                {
                    "_id": "iot:Pump-1:2023-12-31T23:59:00",
                    "asset_id": "Pump-1",
                    "timestamp": "2023-12-31T23:59:00",
                    "Temp": 0.0,
                },
                {
                    "_id": "iot:Pump-1:2024-01-01T00:01:00",
                    "asset_id": "Pump-1",
                    "timestamp": "2024-01-01T00:01:00",
                    "Temp": 1.0,
                },
                {
                    "_id": "iot:Pump-1:2024-01-01T00:02:00",
                    "asset_id": "Pump-1",
                    "timestamp": "2024-01-01T00:02:00",
                    "Temp": 2.0,
                },
            ]
        )

        with patch("servers.iot.main.iot_db", fake_iot_db):
            data = await call_tool(
                mcp,
                "stream_extent",
                {
                    "site_name": "MAIN",
                    "asset_id": "Pump-1",
                    "start": "2024-01-01T00:00:00",
                },
            )

        assert data["total_records"] == 2
        assert fake_iot_db.find_calls == []
        assert fake_iot_db.view_calls == []
        first_call = fake_iot_db.all_docs_calls[0]
        assert json.loads(first_call["startkey"]) == "iot:Pump-1:2024-01-01T00:00:00"
        assert json.loads(first_call["endkey"]) == "iot:Pump-1;"

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
            "timestamp": {
                "$gte": "2024-01-01T00:00:00",
                "$lt": "2024-01-01T00:05:00",
            },
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
        assert selector["timestamp"] == {"$gte": "2024-01-01", "$lt": "2024-01-02"}

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


class TestHistory:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(
            mcp,
            "history",
            {"site_name": "INVALID", "asset_id": "Pump-1"},
        )

        assert "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_invalid_date(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "start": "not-a-date",
            },
        )

        assert "Invalid date format" in data["error"]

    @pytest.mark.anyio
    async def test_db_disconnected(self, mock_asset_db, no_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "history",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert "not connected" in data["error"].lower()

    @pytest.mark.anyio
    async def test_rejects_limits_outside_page_range(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        for limit in (0, 1001):
            data = await call_tool(
                mcp,
                "history",
                {
                    "site_name": "MAIN",
                    "asset_id": "Pump-1",
                    "limit": limit,
                },
            )
            assert data == {"error": "limit must be between 1 and 1000"}
        mock_iot_db.find.assert_not_called()

    @pytest.mark.anyio
    async def test_rejects_empty_sensor_list(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "history",
            {"site_name": "MAIN", "asset_id": "Pump-1", "sensors": []},
        )

        assert data == {"error": "sensors must not be empty when provided"}
        mock_iot_db.find.assert_not_called()

    @pytest.mark.anyio
    async def test_rejects_unknown_sensor(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [{"timestamp": "2024-01-01T00:00:00", "Temp": 1.0}]
        }

        data = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "sensors": ["Pressure"],
            },
        )

        assert data == {"error": "unknown sensors ['Pressure'] for asset_id Pump-1"}

    @pytest.mark.anyio
    async def test_exact_cursor_pagination_and_metadata_exclusion(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {
                    "_id": "a",
                    "_rev": "1-a",
                    "asset_id": "Pump-1",
                    "timestamp": "2024-01-01T00:00:00",
                    "Temp": 1.0,
                },
                {
                    "_id": "b",
                    "asset_id": "Pump-1",
                    "timestamp": "2024-01-01T00:01:00",
                    "Temp": 2.0,
                },
                {
                    "_id": "c",
                    "asset_id": "Pump-1",
                    "timestamp": "2024-01-01T00:02:00",
                    "Temp": 3.0,
                },
            ]
        }

        first = await call_tool(
            mcp,
            "history",
            {"site_name": "MAIN", "asset_id": "Pump-1", "limit": 2},
        )

        assert first["returned"] == 2
        assert first["has_more"] is True
        assert first["next_cursor"] is not None
        assert first["observations"] == [
            {
                "asset_id": "Pump-1",
                "timestamp": "2024-01-01T00:00:00",
                "Temp": 1.0,
            },
            {
                "asset_id": "Pump-1",
                "timestamp": "2024-01-01T00:01:00",
                "Temp": 2.0,
            },
        ]

        second = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "limit": 2,
                "cursor": first["next_cursor"],
            },
        )

        assert second["returned"] == 1
        assert second["has_more"] is False
        assert second["next_cursor"] is None
        assert second["observations"][0]["timestamp"] == "2024-01-01T00:02:00"

    @pytest.mark.anyio
    async def test_cursor_is_bound_to_query(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {"timestamp": "2024-01-01T00:00:00", "Temp": 1.0},
                {"timestamp": "2024-01-01T00:01:00", "Temp": 2.0},
            ]
        }
        first = await call_tool(
            mcp,
            "history",
            {"site_name": "MAIN", "asset_id": "Pump-1", "limit": 1},
        )

        data = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "start": "2024-01-01T00:00:00",
                "limit": 1,
                "cursor": first["next_cursor"],
            },
        )

        assert data == {"error": "cursor does not match history query"}

    @pytest.mark.anyio
    async def test_invalid_cursor(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "cursor": "not-a-cursor",
            },
        )

        assert data == {"error": "invalid cursor"}
        mock_iot_db.find.assert_not_called()

    @pytest.mark.anyio
    async def test_sensor_projection_deduplicates_fields(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "Temp": 1.0,
                    "Pressure": 5.0,
                }
            ]
        }

        data = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "sensors": ["Temp", "Temp"],
            },
        )

        assert data["observations"] == [
            {
                "asset_id": "Pump-1",
                "timestamp": "2024-01-01T00:00:00",
                "Temp": 1.0,
            }
        ]
        history_query = mock_iot_db.find.call_args_list[1]
        assert history_query.kwargs["fields"] == ["timestamp", "Temp"]

    @pytest.mark.anyio
    async def test_half_open_window(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {"timestamp": "2024-01-01T00:00:00", "Temp": 0.0},
                {"timestamp": "2024-01-01T00:01:00", "Temp": 1.0},
                {"timestamp": "2024-01-01T00:02:00", "Temp": 2.0},
            ]
        }

        data = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "start": "2024-01-01T00:01:00",
                "end": "2024-01-01T00:02:00",
            },
        )

        assert data["returned"] == 1
        assert data["observations"][0]["timestamp"] == "2024-01-01T00:01:00"

    @pytest.mark.anyio
    async def test_rejects_non_chronological_timestamp_representations(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {"timestamp": "2023-12-31T23:00:00+00:00", "Temp": 1.0},
                {"timestamp": "2024-01-01T00:30:00+02:00", "Temp": 2.0},
            ]
        }

        data = await call_tool(
            mcp,
            "history",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data == {
            "error": "telemetry timestamps cannot be returned in chronological order"
        }

    @requires_iot_db
    @pytest.mark.anyio
    async def test_discovery_integration(self):
        first = await call_tool(
            mcp,
            "history",
            {"site_name": "MAIN", "asset_id": "Chiller 6", "limit": 2},
        )

        assert first["returned"] == 2
        assert first["has_more"] is True
        second = await call_tool(
            mcp,
            "history",
            {
                "site_name": "MAIN",
                "asset_id": "Chiller 6",
                "limit": 2,
                "cursor": first["next_cursor"],
            },
        )
        assert second["returned"] == 2
        assert (
            first["observations"][-1]["timestamp"]
            < second["observations"][0]["timestamp"]
        )


class TestLatestReading:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(
            mcp,
            "latest_reading",
            {"site_name": "INVALID", "asset_id": "Pump-1"},
        )

        assert "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_db_disconnected(self, mock_asset_db, no_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "latest_reading",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert "not connected" in data["error"].lower()

    @pytest.mark.anyio
    async def test_rejects_reserved_sensor(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "latest_reading",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "sensor": "timestamp",
            },
        )

        assert "reserved metadata field timestamp" in data["error"]
        mock_iot_db.find.assert_not_called()

    @pytest.mark.anyio
    async def test_rejects_unknown_sensor(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [{"timestamp": "2024-01-01T00:00:00", "Temp": 1.0}]
        }

        data = await call_tool(
            mcp,
            "latest_reading",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "sensor": "Pressure",
            },
        )

        assert data == {"error": "unknown sensor Pressure for asset_id Pump-1"}

    @pytest.mark.anyio
    async def test_selects_latest_parsed_timestamp_and_excludes_metadata(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {
                    "_id": "a",
                    "timestamp": "2023-12-31T23:00:00+00:00",
                    "Temp": 1.0,
                },
                {
                    "_rev": "1-b",
                    "timestamp": "2024-01-01T00:30:00+02:00",
                    "Temp": 2.0,
                },
                {
                    "dataset": "sample",
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "Temp": 3.0,
                },
            ]
        }

        data = await call_tool(
            mcp,
            "latest_reading",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data["timestamp"] == "2024-01-01T00:00:00+00:00"
        assert data["values"] == {"Temp": 3.0}
        assert data["age_seconds"] > 0

    @pytest.mark.anyio
    async def test_uses_materialized_summary_for_latest(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = SummaryBackedIotDb(
            {"iot_summary:Pump-1": iot_summary_doc()}
        )

        with patch("servers.iot.main.iot_db", fake_iot_db):
            data = await call_tool(
                mcp,
                "latest_reading",
                {"site_name": "MAIN", "asset_id": "Pump-1"},
            )

        assert data["timestamp"] == "2024-01-01T00:02:00"
        assert data["values"] == {"Pressure": None, "Temp": 12.0}
        assert fake_iot_db.find_calls == []
        assert fake_iot_db.view_calls == []

    @pytest.mark.anyio
    async def test_uses_materialized_summary_for_sensor_latest(
        self, mock_asset_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = SummaryBackedIotDb(
            {"iot_summary:Pump-1": iot_summary_doc()}
        )

        with patch("servers.iot.main.iot_db", fake_iot_db):
            data = await call_tool(
                mcp,
                "latest_reading",
                {"site_name": "MAIN", "asset_id": "Pump-1", "sensor": "Pressure"},
            )

        assert data["timestamp"] == "2024-01-01T00:01:00"
        assert data["values"] == {"Pressure": 6.0}
        assert fake_iot_db.find_calls == []
        assert fake_iot_db.view_calls == []

    @pytest.mark.anyio
    async def test_uses_latest_stream_view_when_available(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = ViewBackedIotDb(
            {
                "stream_by_asset_time": [
                    {
                        "key": ["Pump-1", "2024-01-01T00:02:00"],
                        "doc": {
                            "_id": "iot:Pump-1:2024-01-01T00:02:00",
                            "asset_id": "Pump-1",
                            "timestamp": "2024-01-01T00:02:00",
                            "Temp": 3.0,
                            "Pressure": 12.0,
                        },
                    }
                ]
            }
        )

        with patch("servers.iot.main.iot_db", fake_iot_db), patch(
            "servers.iot.main.IOT_ENABLE_VIEW_FAST_PATH", True
        ):
            data = await call_tool(
                mcp,
                "latest_reading",
                {"site_name": "MAIN", "asset_id": "Pump-1"},
            )

        assert data["timestamp"] == "2024-01-01T00:02:00"
        assert data["values"] == {"Temp": 3.0, "Pressure": 12.0}
        assert fake_iot_db.find_calls == []
        assert fake_iot_db.view_calls[0][1] == "stream_by_asset_time"
        assert fake_iot_db.view_calls[0][2]["include_docs"] is True

    @pytest.mark.anyio
    async def test_sensor_latest_uses_sensor_stream_view(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = ViewBackedIotDb(
            {
                "coverage_by_asset_sensor": [
                    {"key": ["Pump-1", "Temp"], "value": {"non_null_count": 2}},
                    {"key": ["Pump-1", "Pressure"], "value": {"non_null_count": 2}},
                ],
                "stream_by_asset_sensor_time": [
                    {
                        "key": ["Pump-1", "Temp", "2024-01-01T00:02:00"],
                        "doc": {
                            "asset_id": "Pump-1",
                            "timestamp": "2024-01-01T00:02:00",
                            "Temp": 3.0,
                            "Pressure": 12.0,
                        },
                    }
                ],
            }
        )

        with patch("servers.iot.main.iot_db", fake_iot_db), patch(
            "servers.iot.main.IOT_ENABLE_VIEW_FAST_PATH", True
        ):
            data = await call_tool(
                mcp,
                "latest_reading",
                {"site_name": "MAIN", "asset_id": "Pump-1", "sensor": "Temp"},
            )

        assert data["timestamp"] == "2024-01-01T00:02:00"
        assert data["values"] == {"Temp": 3.0}
        assert fake_iot_db.find_calls == []
        assert [call[1] for call in fake_iot_db.view_calls] == [
            "coverage_by_asset_sensor",
            "stream_by_asset_sensor_time",
        ]

    @pytest.mark.anyio
    async def test_sensor_filter_returns_only_requested_value(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.side_effect = [
            {
                "docs": [
                    {
                        "timestamp": "2024-01-01T00:00:00",
                        "Temp": 1.0,
                        "Pressure": 5.0,
                    }
                ]
            },
            {
                "docs": [
                    {"timestamp": "2024-01-01T00:00:00", "Temp": 1.0},
                    {"timestamp": "2024-01-01T00:01:00", "Temp": 2.0},
                ]
            },
        ]

        data = await call_tool(
            mcp,
            "latest_reading",
            {"site_name": "MAIN", "asset_id": "Pump-1", "sensor": "Temp"},
        )

        assert data["timestamp"] == "2024-01-01T00:01:00"
        assert data["values"] == {"Temp": 2.0}

    @pytest.mark.anyio
    async def test_no_records(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {"docs": []}

        data = await call_tool(
            mcp,
            "latest_reading",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data == {"error": "no records for asset_id Pump-1"}

    @pytest.mark.anyio
    async def test_timestamp_error_is_returned(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [{"timestamp": "not-a-date", "Temp": 1.0}]
        }

        data = await call_tool(
            mcp,
            "latest_reading",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data == {
            "error": "telemetry record has an invalid ISO 8601 timestamp"
        }

    @requires_iot_db
    @pytest.mark.anyio
    async def test_discovery_integration(self):
        data = await call_tool(
            mcp,
            "latest_reading",
            {
                "site_name": "MAIN",
                "asset_id": "Chiller 6",
                "sensor": "Chiller 6 Supply Temperature",
            },
        )

        assert data["timestamp"] == "2020-06-30T23:45:00"
        assert set(data["values"]) == {"Chiller 6 Supply Temperature"}
        assert data["age_seconds"] > 0


class TestSensorCoverage:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(
            mcp,
            "sensor_coverage",
            {"site_name": "INVALID", "asset_id": "Pump-1"},
        )

        assert "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_db_disconnected(self, mock_asset_db, no_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "sensor_coverage",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert "not connected" in data["error"].lower()

    @pytest.mark.anyio
    async def test_counts_non_null_values_and_chronological_bounds(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {
                    "timestamp": "2024-01-01T00:02:00",
                    "Temp": 2.0,
                    "Pressure": None,
                },
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "Temp": None,
                    "Pressure": 1.0,
                    "Mode": None,
                },
                {
                    "timestamp": "2024-01-01T00:01:00",
                    "Temp": 1.0,
                    "Pressure": False,
                },
            ]
        }

        data = await call_tool(
            mcp,
            "sensor_coverage",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data["docs_scanned"] == 3
        coverage = {item["sensor"]: item for item in data["sensors"]}
        assert list(item["sensor"] for item in data["sensors"]) == [
            "Mode",
            "Pressure",
            "Temp",
        ]
        assert coverage["Mode"] == {
            "sensor": "Mode",
            "non_null_count": 0,
            "first_timestamp": None,
            "last_timestamp": None,
        }
        assert coverage["Pressure"]["non_null_count"] == 2
        assert coverage["Pressure"]["first_timestamp"] == "2024-01-01T00:00:00"
        assert coverage["Pressure"]["last_timestamp"] == "2024-01-01T00:01:00"
        assert coverage["Temp"]["non_null_count"] == 2
        assert coverage["Temp"]["first_timestamp"] == "2024-01-01T00:01:00"
        assert coverage["Temp"]["last_timestamp"] == "2024-01-01T00:02:00"

    @pytest.mark.anyio
    async def test_uses_materialized_summary(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = SummaryBackedIotDb(
            {"iot_summary:Pump-1": iot_summary_doc()}
        )

        with patch("servers.iot.main.iot_db", fake_iot_db):
            data = await call_tool(
                mcp,
                "sensor_coverage",
                {"site_name": "MAIN", "asset_id": "Pump-1"},
            )

        assert data["docs_scanned"] == 3
        assert data["sensors"] == [
            {
                "sensor": "Pressure",
                "non_null_count": 2,
                "first_timestamp": "2024-01-01T00:00:00",
                "last_timestamp": "2024-01-01T00:01:00",
            },
            {
                "sensor": "Temp",
                "non_null_count": 3,
                "first_timestamp": "2024-01-01T00:00:00",
                "last_timestamp": "2024-01-01T00:02:00",
            },
        ]
        assert fake_iot_db.find_calls == []
        assert fake_iot_db.view_calls == []

    @pytest.mark.anyio
    async def test_no_records(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {"docs": []}

        data = await call_tool(
            mcp,
            "sensor_coverage",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data == {"error": "unknown asset_id Pump-1 or no records found"}

    @pytest.mark.anyio
    async def test_timestamp_error_is_returned(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [{"timestamp": "not-a-date", "Temp": 1.0}]
        }

        data = await call_tool(
            mcp,
            "sensor_coverage",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert data == {
            "error": "telemetry record has an invalid ISO 8601 timestamp"
        }

    @requires_iot_db
    @pytest.mark.anyio
    async def test_discovery_integration(self):
        data = await call_tool(
            mcp,
            "sensor_coverage",
            {
                "site_name": "MAIN",
                "asset_id": "Chiller 6",
            },
        )

        assert data["docs_scanned"] > 0
        assert any(
            item["sensor"] == "Chiller 6 Supply Temperature"
            for item in data["sensors"]
        )


class TestSensorStats:
    @pytest.mark.anyio
    async def test_invalid_site(self):
        data = await call_tool(
            mcp,
            "sensor_stats",
            {"site_name": "INVALID", "asset_id": "Pump-1"},
        )

        assert "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_invalid_date(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "sensor_stats",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "start": "not-a-date",
            },
        )

        assert "Invalid date format" in data["error"]

    @pytest.mark.anyio
    async def test_db_disconnected(self, mock_asset_db, no_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "sensor_stats",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        assert "not connected" in data["error"].lower()

    @pytest.mark.anyio
    async def test_rejects_reserved_sensor_field(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}

        data = await call_tool(
            mcp,
            "sensor_stats",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "sensor": "timestamp",
            },
        )

        assert data == {
            "error": (
                "sensor must be a telemetry field, "
                "not reserved metadata field timestamp"
            )
        }
        mock_iot_db.find.assert_not_called()

    @pytest.mark.anyio
    async def test_rejects_unknown_sensor(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [{"timestamp": "2024-01-01T00:00:00", "Pressure": 1.0}]
        }

        data = await call_tool(
            mcp,
            "sensor_stats",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "sensor": "Temperature",
            },
        )

        assert data == {"error": "no records for asset_id Pump-1, sensor Temperature"}

    @pytest.mark.anyio
    async def test_single_sensor_numeric_summary(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {"timestamp": "2024-01-01T00:02:00", "Temp": None},
                {"timestamp": "2024-01-01T00:00:00", "Temp": 1.0},
                {"timestamp": "2024-01-01T00:04:00", "Temp": "bad"},
                {"timestamp": "2024-01-01T00:01:00", "Temp": "2.5"},
                {"timestamp": "2024-01-01T00:03:00", "Temp": True},
                {"timestamp": "2024-01-01T00:05:00", "Temp": float("inf")},
                {"timestamp": "2024-01-01T00:06:00"},
            ]
        }

        data = await call_tool(
            mcp,
            "sensor_stats",
            {"site_name": "MAIN", "asset_id": "Pump-1", "sensor": "Temp"},
        )

        assert data == {
            "site_name": "MAIN",
            "asset_id": "Pump-1",
            "stats": [
                {
                    "sensor": "Temp",
                    "count": 2,
                    "null_count": 4,
                    "min": 1.0,
                    "max": 2.5,
                    "mean": 1.75,
                    "stddev": 0.75,
                    "first_timestamp": "2024-01-01T00:00:00",
                    "last_timestamp": "2024-01-01T00:01:00",
                }
            ],
            "message": "numeric stats for 1 sensor(s) on asset_id Pump-1.",
        }
        stats_query = mock_iot_db.find.call_args_list[0]
        assert stats_query.args[0] == {
            "asset_id": "Pump-1",
            "timestamp": {"$gt": None},
            "Temp": {"$exists": True},
        }
        assert stats_query.kwargs["fields"] == ["timestamp", "Temp"]

    @pytest.mark.anyio
    async def test_all_measured_sensors_when_omitted(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "Temp": 10.0,
                    "Pressure": 1.0,
                    "Mode": "auto",
                },
                {
                    "timestamp": "2024-01-01T00:01:00",
                    "Temp": 20.0,
                    "Pressure": 3.0,
                    "Mode": "manual",
                },
            ]
        }

        data = await call_tool(
            mcp,
            "sensor_stats",
            {"site_name": "MAIN", "asset_id": "Pump-1"},
        )

        stats = {item["sensor"]: item for item in data["stats"]}
        assert list(item["sensor"] for item in data["stats"]) == [
            "Mode",
            "Pressure",
            "Temp",
        ]
        assert stats["Pressure"]["mean"] == 2.0
        assert stats["Temp"]["stddev"] == 5.0
        assert stats["Mode"]["count"] == 0
        assert stats["Mode"]["null_count"] == 2

    @pytest.mark.anyio
    async def test_full_stream_uses_materialized_summary(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = SummaryBackedIotDb(
            {"iot_summary:Pump-1": iot_summary_doc()}
        )

        with patch("servers.iot.main.iot_db", fake_iot_db):
            data = await call_tool(
                mcp,
                "sensor_stats",
                {"site_name": "MAIN", "asset_id": "Pump-1"},
            )

        assert [item["sensor"] for item in data["stats"]] == ["Pressure", "Temp"]
        assert data["stats"][0]["mean"] == 5.5
        assert fake_iot_db.find_calls == []
        assert fake_iot_db.view_calls == []

    @pytest.mark.anyio
    async def test_single_sensor_uses_materialized_summary(self, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        fake_iot_db = SummaryBackedIotDb(
            {"iot_summary:Pump-1": iot_summary_doc()}
        )

        with patch("servers.iot.main.iot_db", fake_iot_db):
            data = await call_tool(
                mcp,
                "sensor_stats",
                {"site_name": "MAIN", "asset_id": "Pump-1", "sensor": "Temp"},
            )

        assert data["stats"] == [
            {
                "sensor": "Temp",
                "count": 3,
                "null_count": 0,
                "min": 10.0,
                "max": 12.0,
                "mean": 11.0,
                "stddev": 0.816496580927726,
                "first_timestamp": "2024-01-01T00:00:00",
                "last_timestamp": "2024-01-01T00:02:00",
            }
        ]
        assert fake_iot_db.find_calls == []
        assert fake_iot_db.view_calls == []

    @pytest.mark.anyio
    async def test_half_open_window_uses_chronological_order(
        self, mock_asset_db, mock_iot_db
    ):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [
                {"timestamp": "2024-01-01T00:05:00", "Temp": 5.0},
                {"timestamp": "2024-01-01T00:02:00", "Temp": 2.0},
                {"timestamp": "2024-01-01T00:00:00", "Temp": 0.0},
                {"timestamp": "2024-01-01T00:04:00", "Temp": 4.0},
            ]
        }

        data = await call_tool(
            mcp,
            "sensor_stats",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "sensor": "Temp",
                "start": "2024-01-01T00:02:00",
                "end": "2024-01-01T00:05:00",
            },
        )

        stat = data["stats"][0]
        assert stat["count"] == 2
        assert stat["mean"] == 3.0
        assert stat["first_timestamp"] == "2024-01-01T00:02:00"
        assert stat["last_timestamp"] == "2024-01-01T00:04:00"

    @pytest.mark.anyio
    async def test_no_records_in_window(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [{"timestamp": "2024-01-01T00:00:00", "Temp": 1.0}]
        }

        data = await call_tool(
            mcp,
            "sensor_stats",
            {
                "site_name": "MAIN",
                "asset_id": "Pump-1",
                "sensor": "Temp",
                "start": "2024-01-02T00:00:00",
            },
        )

        assert data == {"error": "no records for asset_id Pump-1, sensor Temp"}

    @pytest.mark.anyio
    async def test_timestamp_error_is_returned(self, mock_asset_db, mock_iot_db):
        mock_asset_db.find.return_value = {"docs": [{"siteid": "MAIN"}]}
        mock_iot_db.find.return_value = {
            "docs": [{"timestamp": "not-a-date", "Temp": 1.0}]
        }

        data = await call_tool(
            mcp,
            "sensor_stats",
            {"site_name": "MAIN", "asset_id": "Pump-1", "sensor": "Temp"},
        )

        assert data == {
            "error": "telemetry record has an invalid ISO 8601 timestamp"
        }

    @requires_iot_db
    @pytest.mark.anyio
    async def test_discovery_integration(self):
        data = await call_tool(
            mcp,
            "sensor_stats",
            {
                "site_name": "MAIN",
                "asset_id": "Chiller 6",
                "sensor": "Chiller 6 Supply Temperature",
                "start": "2020-06-10",
                "end": "2020-06-11",
            },
        )

        stat = data["stats"][0]
        assert stat["sensor"] == "Chiller 6 Supply Temperature"
        assert stat["count"] > 0
        assert stat["min"] is not None
        assert stat["max"] is not None


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
