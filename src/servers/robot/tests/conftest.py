"""Shared fixtures and helpers for robot MCP server tests."""

import json
import os

import pytest
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

load_dotenv()


# ---------------------------------------------------------------------------
# CouchDB availability gate
# ---------------------------------------------------------------------------


def _couchdb_reachable() -> bool:
    url = os.environ.get("COUCHDB_URL")
    if not url:
        return False
    try:
        import requests
        requests.get(url, timeout=2)
        return True
    except Exception:
        return False


requires_couchdb = pytest.mark.skipif(
    not _couchdb_reachable(),
    reason="CouchDB not reachable (set COUCHDB_URL and ensure CouchDB is running)",
)


# ---------------------------------------------------------------------------
# CouchDB document fixtures — simulate the 3 doc types in the iot DB
# ---------------------------------------------------------------------------

ROBOT_ID = "spot_1"

NOMINAL_PROFILE = {
    "_id":             "profile:motor_01",
    "doc_type":        "asset_robot_profile",
    "display_name":    "Motor 01",
    "waypoint_id":     "wp_motor_01_panel",
    "physical_location": {"x": 7.2, "y": 44.8, "z": 0.0, "room_id": "motor_bay_1"},
    "gauge_path":      None,
    "gauge_range":     [0, 200],
    "gauge_description": "Motor temperature gauge, analog dial, 0–200°C",
    "panel_stuck":     False,
}

NOMINAL_ROBOT_STATE = {
    "_id":                         f"robot_state:{ROBOT_ID}",
    "doc_type":                    "robot_state",
    "robot_id":                    ROBOT_ID,
    "battery_charge_pct":          85.0,
    "battery_low_threshold":       20.0,
    "battery_estimated_runtime_s": 5400.0,
    "at_charge_station":           False,
    "pose": {"x": 0.0, "y": 0.0, "theta": 0.0, "frame": "map"},
    "localization_ok":             True,
    "pose_drift_m":                0.0,
    "fault_state":                 None,
    "power_state":                 "POWERED_ON",
    "stance_state":                "STANDING",
}

NOMINAL_WAYPOINTS = {
    "_id":      "waypoints",
    "doc_type": "waypoints",
    "waypoints": [
        {
            "waypoint_id": "wp_motor_01_panel", "waypoint_name": "Motor 01 — Panel M-01",
            "asset_id": "motor_01", "location_description": "Plant A / Drive Bay 1",
            "x": 7.2, "y": 44.8, "active": True,
        },
        {
            "waypoint_id": "wp_chiller_6_panel", "waypoint_name": "Chiller 6 — Panel C-06",
            "asset_id": "chiller_6", "location_description": "Plant B / Mech Room 2",
            "x": 52.3, "y": 18.1, "active": True,
        },
        {
            "waypoint_id": "wp_dock_main", "waypoint_name": "Main Charging Dock",
            "asset_id": None, "location_description": "Building A / Charging Bay 1",
            "x": 0.0, "y": 0.0, "active": True,
        },
    ],
}


def _make_mock_db(
    profile: dict = None,
    robot_state: dict = None,
    waypoints: dict = None,
) -> MagicMock:
    """Return a MagicMock CouchDB that returns the specified docs on .get()."""
    import copy
    profile_doc     = copy.deepcopy(profile or NOMINAL_PROFILE)
    robot_state_doc = copy.deepcopy(robot_state or NOMINAL_ROBOT_STATE)
    waypoints_doc   = copy.deepcopy(waypoints or NOMINAL_WAYPOINTS)

    doc_map = {
        profile_doc["_id"]:     profile_doc,
        robot_state_doc["_id"]: robot_state_doc,
        waypoints_doc["_id"]:   waypoints_doc,
    }

    mock = MagicMock()
    mock.get.side_effect = lambda doc_id: doc_map.get(doc_id)
    return mock


@pytest.fixture
def mock_db():
    """Nominal CouchDB mock — motor_01 profile, nominal robot state, all waypoints active."""
    with patch("servers.robot.main.db", _make_mock_db()):
        yield


@pytest.fixture
def no_db():
    """Simulate disconnected IoT CouchDB (db=None)."""
    with patch("servers.robot.main.db", None):
        yield


# ---------------------------------------------------------------------------
# Scenario-specific CouchDB mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_panel_stuck():
    """FM-1: panel_stuck=True in profile."""
    import copy
    profile = copy.deepcopy(NOMINAL_PROFILE)
    profile["panel_stuck"] = True
    with patch("servers.robot.main.db", _make_mock_db(profile=profile)):
        yield


@pytest.fixture
def mock_db_low_battery():
    """FM-9: battery_charge_pct below threshold."""
    import copy
    state = copy.deepcopy(NOMINAL_ROBOT_STATE)
    state["battery_charge_pct"]         = 18.0
    state["battery_estimated_runtime_s"] = 360.0
    with patch("servers.robot.main.db", _make_mock_db(robot_state=state)):
        yield


@pytest.fixture
def mock_db_loc_failure():
    """FM-10: localization_ok=False, pose_drift_m=1.8."""
    import copy
    state = copy.deepcopy(NOMINAL_ROBOT_STATE)
    state["localization_ok"] = False
    state["pose_drift_m"]    = 1.8
    state["pose"]            = {"x": 13.1, "y": 31.7, "theta": 0.3, "frame": "map"}
    with patch("servers.robot.main.db", _make_mock_db(robot_state=state)):
        yield


@pytest.fixture
def mock_db_waypoint_missing():
    """FM-11: motor_01 waypoint active=False."""
    import copy
    wps = copy.deepcopy(NOMINAL_WAYPOINTS)
    for wp in wps["waypoints"]:
        if wp["asset_id"] == "motor_01":
            wp["active"] = False
    with patch("servers.robot.main.db", _make_mock_db(waypoints=wps)):
        yield


@pytest.fixture
def mock_db_with_gauge_path():
    """Profile with gauge_path set (for capture_image image_available=True test)."""
    import copy
    profile = copy.deepcopy(NOMINAL_PROFILE)
    profile["gauge_path"] = "/data/gauges/motor_01/gauge_001.jpg"
    with patch("servers.robot.main.db", _make_mock_db(profile=profile)):
        yield


@pytest.fixture
def mock_db_powered_off():
    """Robot power_state=POWERED_OFF, stance_state=SITTING."""
    import copy
    state = copy.deepcopy(NOMINAL_ROBOT_STATE)
    state["power_state"]  = "POWERED_OFF"
    state["stance_state"] = "SITTING"
    with patch("servers.robot.main.db", _make_mock_db(robot_state=state)):
        yield


@pytest.fixture
def mock_db_sitting():
    """Robot powered on but stance_state=SITTING (before stand or after sit)."""
    import copy
    state = copy.deepcopy(NOMINAL_ROBOT_STATE)
    state["stance_state"] = "SITTING"
    with patch("servers.robot.main.db", _make_mock_db(robot_state=state)):
        yield


# ---------------------------------------------------------------------------
# MCP tool call helper
# ---------------------------------------------------------------------------


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Call an MCP tool and return the parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)
