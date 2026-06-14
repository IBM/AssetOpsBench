"""Integration tests for robot asset profile documents in the iot CouchDB database.

Tests verify:
  - All 4 profile docs exist with correct shape
  - All 9 in-scope robot fields are present with correct types
  - gauge_value is NOT reachable via any existing IoT MCP tool
  - Profile documents are invisible to existing IoT server asset/sensor queries
  - idx_robot_never_read Mango index was created
  - Deferred fields (hazard_class, maintenance_slot, active_work_order) are absent
"""

import json
import os

import couchdb3
import pytest
from dotenv import load_dotenv

load_dotenv()

from .conftest import requires_couchdb

COUCHDB_URL      = os.environ.get("COUCHDB_URL", "")
COUCHDB_HOST     = COUCHDB_URL.replace("http://", "").replace("https://", "")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME", "")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD", "")
COUCHDB_DBNAME   = os.environ.get("IOT_DBNAME", "iot")

PROFILE_IDS = [
    "profile:chiller_6",
    "profile:metro_pump_1",
    "profile:hydraulic_pump_1",
    "profile:motor_01",
]

ROBOT_FIELDS = [
    "physical_location",
    "gauge_value",
    "gauge_range",
    "panel_stuck_prob",
    "human_present",
    "never_read",
    "real_gauge_images",
    "reading_consistency",
    "sensor_physical_gap",
]

DEFERRED_FIELDS = ["hazard_class", "maintenance_slot", "active_work_order"]


@pytest.fixture
def raw_db():
    return couchdb3.Server(
        f"http://{COUCHDB_HOST}",
        user=COUCHDB_USERNAME,
        password=COUCHDB_PASSWORD,
    )[COUCHDB_DBNAME]


# ---------------------------------------------------------------------------
# Profile document shape
# ---------------------------------------------------------------------------

@requires_couchdb
class TestRobotAssetProfiles:
    def test_profiles_exist(self, raw_db):
        for pid in PROFILE_IDS:
            doc = raw_db.get(pid)
            assert doc is not None, f"Profile document missing: {pid}"

    def test_doc_type_set(self, raw_db):
        for pid in PROFILE_IDS:
            doc = raw_db.get(pid)
            assert doc["doc_type"] == "asset_robot_profile", (
                f"{pid}: expected doc_type='asset_robot_profile', got {doc.get('doc_type')}"
            )

    def test_all_9_fields_present(self, raw_db):
        for pid in PROFILE_IDS:
            doc = raw_db.get(pid)
            missing = [f for f in ROBOT_FIELDS if f not in doc]
            assert not missing, f"{pid}: missing robot fields: {missing}"

    def test_field_types(self, raw_db):
        for pid in PROFILE_IDS:
            doc = raw_db.get(pid)
            assert isinstance(doc["gauge_value"],     float),           f"{pid}: gauge_value must be float"
            assert isinstance(doc["panel_stuck_prob"], float),          f"{pid}: panel_stuck_prob must be float"
            assert isinstance(doc["human_present"],    bool),           f"{pid}: human_present must be bool"
            assert isinstance(doc["never_read"],       bool),           f"{pid}: never_read must be bool"
            assert isinstance(doc["real_gauge_images"], list),          f"{pid}: real_gauge_images must be list"
            assert doc["physical_location"] is None or isinstance(doc["physical_location"], dict), (
                f"{pid}: physical_location must be dict or null"
            )

    def test_gauge_range_is_two_element_list(self, raw_db):
        for pid in PROFILE_IDS:
            doc = raw_db.get(pid)
            gr = doc["gauge_range"]
            assert isinstance(gr, list) and len(gr) == 2, (
                f"{pid}: gauge_range must be [min, max], got {gr}"
            )
            assert gr[0] < gr[1], f"{pid}: gauge_range[0] must be < gauge_range[1], got {gr}"

    def test_deferred_fields_absent(self, raw_db):
        for pid in PROFILE_IDS:
            doc = raw_db.get(pid)
            present = [f for f in DEFERRED_FIELDS if f in doc]
            assert not present, (
                f"{pid}: deferred fields must not be in DB yet: {present}"
            )

    def test_no_asset_id_field(self, raw_db):
        """Profiles must not have asset_id — that field is what IoT server queries scan for."""
        for pid in PROFILE_IDS:
            doc = raw_db.get(pid)
            assert "asset_id" not in doc, (
                f"{pid}: profile must NOT have asset_id field (would pollute IoT server queries)"
            )


# ---------------------------------------------------------------------------
# gauge_value protection — must not leak through any IoT MCP tool
# ---------------------------------------------------------------------------

@requires_couchdb
class TestGaugeValueProtection:
    """Verifies gauge_value can never reach the agent via existing IoT tools."""

    @pytest.mark.anyio
    async def test_gauge_value_not_in_sensor_list(self):
        from servers.iot.main import mcp, _asset_list_cache, _sensor_list_cache
        import servers.iot.main as iot_main

        # Clear caches so real DB is queried
        iot_main._asset_list_cache = None
        iot_main._sensor_list_cache = {}

        known_assets = ["Chiller 6", "Metro Pump 1", "Hydraulic Pump 1", "Motor 01"]
        for asset_id in known_assets:
            contents, _ = await mcp.call_tool("sensors", {"site_name": "MAIN", "asset_id": asset_id})
            result = json.loads(contents[0].text)
            if "sensors" in result:
                assert "gauge_value" not in result["sensors"], (
                    f"gauge_value leaked into sensor list for {asset_id}: {result['sensors']}"
                )

        iot_main._asset_list_cache = None
        iot_main._sensor_list_cache = {}

    @pytest.mark.anyio
    async def test_gauge_value_not_in_history(self):
        from servers.iot.main import mcp
        import servers.iot.main as iot_main

        iot_main._asset_list_cache = None
        iot_main._sensor_list_cache = {}

        contents, _ = await mcp.call_tool("history", {
            "site_name": "MAIN",
            "asset_id":  "Chiller 6",
            "start":     "2020-06-01T00:00:00",
            "final":     "2020-06-01T01:00:00",
        })
        result = json.loads(contents[0].text)

        if "observations" in result:
            for obs in result["observations"]:
                assert "gauge_value" not in obs, (
                    f"gauge_value leaked into history observation: {list(obs.keys())}"
                )

        iot_main._asset_list_cache = None
        iot_main._sensor_list_cache = {}

    @pytest.mark.anyio
    async def test_profile_docs_not_in_asset_list(self):
        """Profile documents must be invisible to the IoT server asset enumeration."""
        from servers.iot.main import mcp
        import servers.iot.main as iot_main

        iot_main._asset_list_cache = None
        iot_main._sensor_list_cache = {}

        contents, _ = await mcp.call_tool("assets", {"site_name": "MAIN"})
        result = json.loads(contents[0].text)

        if "assets" in result:
            profile_leaks = [a for a in result["assets"] if str(a).startswith("profile:")]
            assert not profile_leaks, (
                f"Profile document IDs appeared in asset list: {profile_leaks}"
            )

        iot_main._asset_list_cache = None
        iot_main._sensor_list_cache = {}


# ---------------------------------------------------------------------------
# Mango index verification
# ---------------------------------------------------------------------------

@requires_couchdb
class TestRobotIndexes:
    def test_never_read_index_created(self):
        import requests

        resp = requests.get(
            f"http://{COUCHDB_HOST}/{COUCHDB_DBNAME}/_index",
            auth=(COUCHDB_USERNAME, COUCHDB_PASSWORD),
        )
        assert resp.status_code == 200, f"Could not query _index endpoint: {resp.status_code}"

        index_names = [idx.get("name") for idx in resp.json().get("indexes", [])]
        assert "idx_robot_never_read" in index_names, (
            f"idx_robot_never_read not found. Existing indexes: {index_names}"
        )
