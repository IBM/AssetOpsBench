"""Tests for the IoT MCP server tools (12-tool redesign).

Two databases: `db` = telemetry readings, `asset_db` = asset registry (keyed by `assetnum`,
scoped by `siteid`). Unit tests mock the relevant DB(s); integration tests need a live CouchDB
(skipped unless COUCHDB_URL is set). `known_main` seeds the registry-sites cache so site checks
pass offline.

`get_asset_doc` now filters on BOTH `assetnum` and `siteid`, so registry mocks that must resolve
a specific (asset, site) pair use `_registry_find`, a fake that honours those selector keys like
CouchDB Mango would.
"""

import pytest
from servers.iot.main import mcp
from .conftest import requires_couchdb, call_tool


def _registry_find(docs):
    """A fake `asset_db.find` that filters on the selector's assetnum / siteid / assettype
    (like Mango), so site-scoped lookups (get_asset_doc) behave realistically in unit tests."""
    def _find(selector, limit=100000, **kwargs):
        out = [
            d for d in docs
            if all(
                selector.get(k) is None or d.get(k) == selector.get(k)
                for k in ("assetnum", "siteid", "assettype")
            )
        ]
        return {"docs": out[:limit] if limit else out}

    return _find


# ---------------------------------------------------------------------------
# sites
# ---------------------------------------------------------------------------
class TestSites:
    @pytest.mark.anyio
    async def test_fallback_to_default_when_registry_empty(self, no_asset_db):
        data = await call_tool(mcp, "sites", {})
        assert data["sites"] == ["MAIN"]

    @pytest.mark.anyio
    async def test_discovered_from_registry(self, mock_asset_db):
        mock_asset_db.find.return_value = {
            "docs": [{"siteid": "MAIN"}, {"siteid": "NORTH"}, {"siteid": "MAIN"}]
        }
        data = await call_tool(mcp, "sites", {})
        assert data["sites"] == ["MAIN", "NORTH"]


# ---------------------------------------------------------------------------
# asset_ids  (bare IDs from the registry, keyed by assetnum)
# ---------------------------------------------------------------------------
class TestAssetIds:
    @pytest.mark.anyio
    async def test_invalid_site(self, known_main):
        data = await call_tool(mcp, "asset_ids", {"site_name": "INVALID"})
        assert "error" in data and "unknown site" in data["error"]

    @pytest.mark.anyio
    async def test_returns_assetnums(self, known_main, mock_asset_db):
        mock_asset_db.find.return_value = {
            "docs": [{"assetnum": "PUMP3"}, {"assetnum": "CHILLER 6"}]
        }
        data = await call_tool(mcp, "asset_ids", {"site_name": "MAIN"})
        assert data["total_assets"] == 2
        assert data["assets"] == ["CHILLER 6", "PUMP3"]        # sorted, List[str]

    @requires_couchdb
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "asset_ids", {"site_name": "MAIN"})
        assert "assets" in data and data["total_assets"] > 0
        assert "CHILLER 6" in data["assets"]


# ---------------------------------------------------------------------------
# assets  (typed metadata rows — distinct from asset_ids)
# ---------------------------------------------------------------------------
class TestAssets:
    @pytest.mark.anyio
    async def test_typed_metadata_rows(self, known_main, mock_asset_db):
        mock_asset_db.find.return_value = {
            "docs": [
                {
                    "assetnum": "CHILLER 6",
                    "siteid": "MAIN",
                    "assettype": "CHILLER",
                    "description": "Chiller 6",
                    "vintage": "2015",
                    "sensors": ["a", "b", "c"],
                }
            ]
        }
        data = await call_tool(mcp, "assets", {"site_name": "MAIN"})
        assert data["total_assets"] == 1
        row = data["assets"][0]
        assert row["asset_id"] == "CHILLER 6"
        assert row["assettype"] == "CHILLER"
        assert row["vintage"] == "2015"
        assert row["n_sensors"] == 3          # metadata IS present (the reviewer's R2 fix)

    @pytest.mark.anyio
    async def test_assettype_filter_passthrough(self, known_main, mock_asset_db):
        mock_asset_db.find.return_value = {"docs": []}
        await call_tool(mcp, "assets", {"site_name": "MAIN", "assettype": "PUMP"})
        selector = mock_asset_db.find.call_args.args[0]
        assert selector.get("assettype") == "PUMP"
        assert selector.get("siteid") == "MAIN"


# ---------------------------------------------------------------------------
# measured_sensors  (telemetry — union of columns)
# ---------------------------------------------------------------------------
class TestMeasuredSensors:
    @pytest.mark.anyio
    async def test_unknown_asset(self, known_main, mock_db):
        mock_db.find.return_value = {"docs": []}
        data = await call_tool(
            mcp, "measured_sensors", {"site_name": "MAIN", "asset_id": "NOPE"}
        )
        assert "error" in data and "no sensors found" in data["error"]

    @pytest.mark.anyio
    async def test_excludes_reserved_fields(self, known_main, mock_db):
        mock_db.find.return_value = {
            "docs": [
                {"asset_id": "A", "timestamp": "t0", "Temp": 1.0, "_id": "x", "_rev": "y"},
                {"asset_id": "A", "timestamp": "t1", "Pressure": 2.0},
            ]
        }
        data = await call_tool(
            mcp, "measured_sensors", {"site_name": "MAIN", "asset_id": "A"}
        )
        assert data["total_sensors"] == 2
        assert set(data["sensors"]) == {"Temp", "Pressure"}
        assert "_id" not in data["sensors"] and "timestamp" not in data["sensors"]


# ---------------------------------------------------------------------------
# installed_sensors  (registry nameplate list — site-scoped)
# ---------------------------------------------------------------------------
class TestInstalledSensors:
    _DOC = {"assetnum": "CHILLER 6", "siteid": "MAIN", "sensors": ["s1", "s2"]}

    @pytest.mark.anyio
    async def test_from_registry(self, known_main, mock_asset_db):
        mock_asset_db.find.side_effect = _registry_find([self._DOC])
        data = await call_tool(
            mcp, "installed_sensors", {"site_name": "MAIN", "asset_id": "CHILLER 6"}
        )
        assert data["total_sensors"] == 2
        assert data["sensors"] == ["s1", "s2"]

    @pytest.mark.anyio
    async def test_asset_not_at_site(self, monkeypatch, mock_asset_db):
        # asset exists at MAIN; querying it via NORTH (a known site) must not resolve it
        monkeypatch.setattr("servers.iot.main._registry_sites_cache", ["MAIN", "NORTH"])
        mock_asset_db.find.side_effect = _registry_find([self._DOC])
        data = await call_tool(
            mcp, "installed_sensors", {"site_name": "NORTH", "asset_id": "CHILLER 6"}
        )
        assert "error" in data and "at site NORTH" in data["error"]


# ---------------------------------------------------------------------------
# get_asset_detail  (nameplate — site-scoped)
# ---------------------------------------------------------------------------
class TestGetAssetDetail:
    _DOC = {
        "assetnum": "CHILLER 6",
        "siteid": "MAIN",
        "assettype": "CHILLER",
        "status": "OPERATING",
        "location": "MAIN-MECH-CH6",
        "description": "Chiller 6",
        "sensors": ["a", "b"],
    }

    @pytest.mark.anyio
    async def test_unknown_asset(self, known_main, mock_asset_db):
        mock_asset_db.find.side_effect = _registry_find([])
        data = await call_tool(
            mcp, "get_asset_detail", {"site_name": "MAIN", "asset_id": "NOPE"}
        )
        assert "error" in data and "at site MAIN" in data["error"]

    @pytest.mark.anyio
    async def test_nameplate_fields(self, known_main, mock_asset_db):
        mock_asset_db.find.side_effect = _registry_find([self._DOC])
        data = await call_tool(
            mcp, "get_asset_detail", {"site_name": "MAIN", "asset_id": "CHILLER 6"}
        )
        assert data["asset_id"] == "CHILLER 6"
        assert data["assettype"] == "CHILLER"
        assert data["status"] == "OPERATING"
        assert data["n_sensors"] == 2

    @pytest.mark.anyio
    async def test_asset_not_at_site(self, monkeypatch, mock_asset_db):
        monkeypatch.setattr("servers.iot.main._registry_sites_cache", ["MAIN", "NORTH"])
        mock_asset_db.find.side_effect = _registry_find([self._DOC])
        data = await call_tool(
            mcp, "get_asset_detail", {"site_name": "NORTH", "asset_id": "CHILLER 6"}
        )
        assert "error" in data and "at site NORTH" in data["error"]


# ---------------------------------------------------------------------------
# find_assets_by_sensors
# ---------------------------------------------------------------------------
class TestFindAssetsBySensors:
    @pytest.mark.anyio
    async def test_bad_match(self, known_main):
        data = await call_tool(
            mcp,
            "find_assets_by_sensors",
            {"site_name": "MAIN", "sensors": ["Temp"], "match": "bogus"},
        )
        assert "error" in data and "match must be" in data["error"]

    @pytest.mark.anyio
    async def test_measured_source_substring(self, known_main, mock_db, mock_asset_db):
        # _site_asset_ids -> asset_db returns one asset; get_sensor_list -> db returns its columns
        mock_asset_db.find.return_value = {"docs": [{"assetnum": "CHILLER 6"}]}
        mock_db.find.return_value = {
            "docs": [{"asset_id": "CHILLER 6", "timestamp": "t0",
                      "Chiller 6 Chiller Efficiency": 0.8}]
        }
        data = await call_tool(
            mcp,
            "find_assets_by_sensors",
            {"site_name": "MAIN", "sensors": ["Efficiency"],
             "substring": True, "source": "measured"},
        )
        assert data["total_assets"] == 1
        assert data["assets"][0]["asset_id"] == "CHILLER 6"


# ---------------------------------------------------------------------------
# history  (merged + paged; start optional; end/limit/cursor)
# ---------------------------------------------------------------------------
class TestHistory:
    @pytest.mark.anyio
    async def test_malformed_date(self, known_main):
        data = await call_tool(
            mcp, "history",
            {"site_name": "MAIN", "asset_id": "A", "start": "not-a-date"},
        )
        assert "error" in data and "Invalid date" in data["error"]

    @pytest.mark.anyio
    async def test_start_after_end(self, known_main):
        data = await call_tool(
            mcp, "history",
            {"site_name": "MAIN", "asset_id": "A",
             "start": "2020-06-02T00:00:00", "end": "2020-06-01T00:00:00"},
        )
        assert "error" in data and "start >= end" in data["error"]

    @pytest.mark.anyio
    async def test_db_disconnected(self, known_main, no_db):
        data = await call_tool(
            mcp, "history", {"site_name": "MAIN", "asset_id": "A"}
        )
        assert "error" in data and "not connected" in data["error"].lower()

    @pytest.mark.anyio
    async def test_start_optional(self, known_main, mock_db):
        # start omitted → full range, no error
        mock_db.find.return_value = {"docs": [{"timestamp": "t0", "Temp": 1.0}]}
        data = await call_tool(mcp, "history", {"site_name": "MAIN", "asset_id": "A"})
        assert data["returned"] == 1
        assert data["has_more"] is False
        assert data["next_cursor"] is None

    @pytest.mark.anyio
    async def test_paging_has_more(self, known_main, mock_db):
        mock_db.find.return_value = {
            "docs": [{"timestamp": "t0"}, {"timestamp": "t1"}],
            "bookmark": "cursor-2",
        }
        data = await call_tool(
            mcp, "history", {"site_name": "MAIN", "asset_id": "A", "limit": 2}
        )
        assert data["returned"] == 2
        assert data["has_more"] is True            # full page => more may follow
        assert data["next_cursor"] == "cursor-2"

    @pytest.mark.anyio
    async def test_sensors_projection(self, known_main, mock_db):
        mock_db.find.return_value = {"docs": [{"timestamp": "t0", "Temp": 1.0}]}
        await call_tool(
            mcp, "history",
            {"site_name": "MAIN", "asset_id": "A", "sensors": ["Temp"]},
        )
        kwargs = mock_db.find.call_args.kwargs
        assert "Temp" in kwargs.get("fields", [])
        assert "timestamp" in kwargs.get("fields", [])

    @requires_couchdb
    @pytest.mark.anyio
    async def test_bounded_range_integration(self):
        data = await call_tool(
            mcp, "history",
            {"site_name": "MAIN", "asset_id": "CHILLER 6",
             "start": "2020-06-01T00:00:00", "end": "2020-06-01T01:00:00"},
        )
        assert "observations" in data
        for obs in data["observations"]:
            assert obs["timestamp"] >= "2020-06-01T00:00:00"
            assert obs["timestamp"] < "2020-06-01T01:00:00"


# ---------------------------------------------------------------------------
# stream_extent  (bounds + count pre-flight)
# ---------------------------------------------------------------------------
class TestStreamExtent:
    @pytest.mark.anyio
    async def test_bounds_and_count(self, known_main, mock_db):
        mock_db.find.return_value = {
            "docs": [{"timestamp": "t0"}, {"timestamp": "t1"}, {"timestamp": "t2"}]
        }
        data = await call_tool(
            mcp, "stream_extent", {"site_name": "MAIN", "asset_id": "A"}
        )
        assert data["total_records"] == 3
        assert data["start_time"] == "t0"
        assert data["end_time"] == "t2"
        assert data["exceeds_page_limit"] is False

    @pytest.mark.anyio
    async def test_no_records(self, known_main, mock_db):
        mock_db.find.return_value = {"docs": []}
        data = await call_tool(
            mcp, "stream_extent", {"site_name": "MAIN", "asset_id": "A"}
        )
        assert "error" in data and "no records" in data["error"]


# ---------------------------------------------------------------------------
# sensor_coverage  (per-sensor counts)
# ---------------------------------------------------------------------------
class TestSensorCoverage:
    @pytest.mark.anyio
    async def test_counts_per_sensor(self, known_main, mock_db):
        mock_db.find.return_value = {
            "docs": [
                {"timestamp": "t0", "Temp": 1.0},
                {"timestamp": "t1", "Temp": 2.0, "Pressure": 5.0},
            ]
        }
        data = await call_tool(
            mcp, "sensor_coverage", {"site_name": "MAIN", "asset_id": "A"}
        )
        cov = {c["sensor"]: c for c in data["sensors"]}
        assert cov["Temp"]["non_null_count"] == 2
        assert cov["Pressure"]["non_null_count"] == 1
        assert cov["Temp"]["first_timestamp"] == "t0"
        assert cov["Temp"]["last_timestamp"] == "t1"


# ---------------------------------------------------------------------------
# sensor_stats  (sensor optional → all sensors)
# ---------------------------------------------------------------------------
class TestSensorStats:
    @pytest.mark.anyio
    async def test_single_sensor(self, known_main, mock_db):
        mock_db.find.return_value = {
            "docs": [
                {"timestamp": "t0", "Temp": 10.0},
                {"timestamp": "t1", "Temp": 20.0},
            ]
        }
        data = await call_tool(
            mcp, "sensor_stats",
            {"site_name": "MAIN", "asset_id": "A", "sensor": "Temp"},
        )
        stats = {s["sensor"]: s for s in data["stats"]}
        assert stats["Temp"]["count"] == 2
        assert stats["Temp"]["min"] == 10.0
        assert stats["Temp"]["max"] == 20.0
        assert stats["Temp"]["mean"] == 15.0

    @pytest.mark.anyio
    async def test_all_sensors_when_omitted(self, known_main, mock_db):
        mock_db.find.return_value = {
            "docs": [
                {"timestamp": "t0", "Temp": 10.0, "Pressure": 1.0},
                {"timestamp": "t1", "Temp": 20.0, "Pressure": 3.0},
            ]
        }
        data = await call_tool(
            mcp, "sensor_stats", {"site_name": "MAIN", "asset_id": "A"}
        )
        names = {s["sensor"] for s in data["stats"]}
        assert names == {"Temp", "Pressure"}


# ---------------------------------------------------------------------------
# latest_reading
# ---------------------------------------------------------------------------
class TestLatestReading:
    @pytest.mark.anyio
    async def test_newest_values(self, known_main, mock_db):
        mock_db.find.return_value = {
            "docs": [{"asset_id": "A", "timestamp": "2024-01-01T00:00:00",
                      "Temp": 42.0, "_id": "x", "_rev": "y"}]
        }
        data = await call_tool(
            mcp, "latest_reading", {"site_name": "MAIN", "asset_id": "A"}
        )
        assert data["timestamp"] == "2024-01-01T00:00:00"
        assert data["values"]["Temp"] == 42.0
        assert "_id" not in data["values"]           # reserved fields excluded
        assert "age_seconds" in data